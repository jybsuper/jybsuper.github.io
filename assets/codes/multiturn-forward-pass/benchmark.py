#!/usr/bin/env python
"""
Benchmark Script for Comparing Different Attention Mechanisms with Reasoning Tokens

This script evaluates different approaches to handle reasoning tokens in LLMs:
1. Naive approach - processes each assistant message independently
2. KV Cache approach - processes each assistant message separately, but uses key-value caching to avoid recomputing the same tokens
3. SDPA Attention - process all message at once with 2D custom attention masks
4. Flex Attention - process all message at once with 2D custom attention masks
5. Full Reasoning - original VeRL method that includes all reasoning tokens during forward pass

The benchmark measures both performance (time and memory) and accuracy (logit similarity).
"""

import json
import time

import numpy as np
import torch
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.functional import softmax, kl_div, log_softmax
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import OffloadedCache

# ============================================================================
# Configuration and Model Loading
# ============================================================================

# Load evaluation dataset
with open("/your/path/to/reasoning-func-call-valid.json") as f:
    data = json.load(f)
print(f"Data loaded: {len(data)} samples")

# Model configuration
model_id = "Qwen/Qwen3-4B"


tokenizer = AutoTokenizer.from_pretrained(model_id)
model0 = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda:0")
model1 = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda:1")
model2 = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda:2")
model3 = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, attn_implementation="flex_attention", device_map="cuda:3")
model4 = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda:4")

# Device for logit comparison (separate GPU to avoid memory conflicts)
compare_device = "cuda:5"

# ============================================================================
# Utilities for Reasoning Token Handling
# ============================================================================

# Create a dummy user turn to append after assistant messages
# This helps extract assistant responses without reasoning tokens
# The specific rules for when reasoning tokens are removed depend on the model
# (e.g., Qwen3 removes them from messages before the last user query)
dummy_turn = [{"role": "user", "content": "I am a user."}]
dummy_turn_len = len(tokenizer.apply_chat_template(dummy_turn, add_generation_prompt=False, tokenize=False))
dummy_turn_token_len = len(tokenizer.apply_chat_template(dummy_turn, add_generation_prompt=False, tokenize=True))
gen_prompt_len = len(tokenizer.apply_chat_template(dummy_turn, add_generation_prompt=True, tokenize=False)) - dummy_turn_len

# ============================================================================
# Accuracy Metrics Functions
# ============================================================================

def compare_logits(logits1, logits2):
    """
    Compare two sets of logits to measure how similar they are.
    
    Returns multiple metrics:
    - close_percent: Percentage of logits that are close (within tolerance)
    - rmse: Root Mean Square Error
    - kl/kl_reverse/kl_symmetric: KL divergence metrics
    - top1_overlap: How often the top prediction matches
    - top8_overlap: How often top-8 predictions overlap
    """
    def topk_overlap(a, b, k):
        """Calculate the overlap between top-k predictions"""
        ta = torch.topk(a, k, dim=-1).indices
        tb = torch.topk(b, k, dim=-1).indices
        return (ta.unsqueeze(-1) == tb.unsqueeze(-2)).any(dim=-1).float().mean().item()

    # Calculate percentage of logits that are close
    close_percent = torch.isclose(logits1, logits2, rtol=1e-1, atol=1e-2).sum().item() * 100 / logits1.numel()

    # Calculate RMSE
    flat1 = logits1.view(-1, logits1.size(-1))
    flat2 = logits2.view(-1, logits2.size(-1))
    rmse = (flat1 - flat2).pow(2).mean().sqrt().item()
    
    # Calculate KL divergences (both directions and symmetric)
    log_probs1 = log_softmax(logits1, dim=-1)
    log_probs2 = log_softmax(logits2, dim=-1)
    probs1 = softmax(logits1, dim=-1)
    probs2 = softmax(logits2, dim=-1)

    kl = kl_div(log_probs1, probs2, reduction='batchmean', log_target=False).item()
    kl_reverse = kl_div(log_probs2, probs1, reduction='batchmean', log_target=False).item()

    # Calculate top-k prediction overlaps
    top1_overlap = topk_overlap(logits1, logits2, 1)
    top8_overlap = topk_overlap(logits1, logits2, 8)
    
    return {
        "close_percent": close_percent,
        "rmse": rmse,
        "kl": kl,
        "kl_reverse": kl_reverse,
        "kl_symmetric": (kl + kl_reverse) / 2,
        "top1_overlap": top1_overlap,
        "top8_overlap": top8_overlap
    }

# ============================================================================
# Performance Measurement Wrapper
# ============================================================================

def measure_forward(forward_fn, model, tokenizer, messages, tools):
    """
    Wrapper function to measure performance of different forward methods.
    
    Tracks:
    - Total execution time
    - Model forward pass time
    - Peak memory usage
    """
    torch.cuda.reset_peak_memory_stats(device=model.device)
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    logits, forward_time = forward_fn(model, tokenizer, messages, tools)
    torch.cuda.synchronize()
    overall_time = time.perf_counter() - start_time

    max_memory_mb = torch.cuda.max_memory_allocated(device=model.device) / 1024 / 1024
    logits = logits.to(compare_device).detach()
    torch.cuda.reset_peak_memory_stats(device=model.device)

    return logits, {
        "total_time": overall_time,
        "model_forward_time": forward_time,
        "max_memory_mb": max_memory_mb
    }

# ============================================================================
# Method 1: Naive Approach
# ============================================================================

def forward_naive(model, tokenizer, messages, tools):
    """
    Naive approach: Process each assistant message independently.
    
    This is the baseline - simple but inefficient as it doesn't reuse
    any computation between messages.
    """
    model_forward_time = 0.0
    all_logits = []

    for idx, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue

        # Get prompt up to this message
        prompt = tokenizer.apply_chat_template(
            messages[:idx], tools=tools, add_generation_prompt=True, tokenize=True
        )
        
        # Get full conversation including this assistant message
        input_ids = tokenizer.apply_chat_template(
            messages[:idx+1], tools=tools, add_generation_prompt=False,
            return_tensors="pt", tokenize=True
        )

        # Forward pass
        torch.cuda.synchronize()
        model_start = time.perf_counter()
        logits = model(input_ids=input_ids.to(model.device)).logits
        torch.cuda.synchronize()
        model_forward_time += (time.perf_counter() - model_start)

        # Extract only the assistant message logits
        all_logits.append(logits[:, len(prompt):, :])

    return torch.cat(all_logits, dim=1), model_forward_time

# ============================================================================
# Method 2: KV Cache Approach
# ============================================================================

def forward_with_kv_cache(model, tokenizer, messages, tools):
    """
    KV Cache approach: Use key-value caching to avoid recomputation.
    
    Special handling for reasoning tokens:
    1. Process messages with reasoning tokens
    2. Crop cache back to prompt
    3. Process without reasoning tokens to update cache and use it as cached context for following messages
    
    Key insight: We batch together consecutive messages that share the same context.
    A new batch starts when reasoning tokens from previous messages need to be removed
    from the context (i.e., when the context changes from the perspective of later messages).
    
    Important: In models like Qwen3, reasoning tokens are ONLY removed from assistant 
    messages that appear BEFORE the last user query. Assistant messages after a user 
    query keep their reasoning tokens intact when they become context.
    
    Example with Qwen3-style reasoning removal (at user query boundaries):
    User: "Question 1"
    Assistant: "Answer 1" (with reasoning)  <-- Reasoning removed when context
    User: "Question 2" 
    Assistant: "Answer 2a" (with reasoning) <-- Reasoning kept when context for 2b
    Assistant: "Answer 2b" (with reasoning) <-- Reasoning kept when context for 2c
    Assistant: "Answer 2c" (with reasoning)
    User: "Question 3"
    Assistant: "Answer 3" (with reasoning)
    
    Batching works when consecutive messages maintain the same representation
    when becoming context (e.g., messages 2a, 2b, 2c in the example above).
    
    Processing triggers in this example:
    - When processing Answer 2a: new batch (Answer 1 loses reasoning in context)
    - When processing Answer 2b: same batch (Answer 2a keeps reasoning in context)
    - When processing Answer 2c: same batch (Answer 2a, 2b keep reasoning in context)
    - When processing Answer 3: new batch (Answer 2a, 2b, 2c lose reasoning in context)
    """
    model_forward_time = 0.0
    kv_cache = OffloadedCache()
    all_logits = []
    ai_msg_boundaries = []
    last_prompt_len = None
    last_ai_message = None
    last_ai_message_idx = None
    
    for idx, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue

        prompt = tokenizer.apply_chat_template(
            messages[:idx], tools=tools, add_generation_prompt=True, tokenize=True
        )
        input_ids_with_reasoning = tokenizer.apply_chat_template(
            messages[:idx+1], tools=tools, add_generation_prompt=False,
            return_tensors="pt", tokenize=True
        )

        # Detect when we cross a context change boundary
        # Check if we can continue batching with previous messages as context without changing their content
        if (last_ai_message is None or input_ids_with_reasoning[:, :last_ai_message.shape[-1]].equal(last_ai_message)):
            # Context representation unchanged - accumulate this message
            # These messages maintain the same form when used as context for subsequent messages in the current batch
            last_ai_message = input_ids_with_reasoning
            last_ai_message_idx = idx
            ai_msg_boundaries.append((len(prompt), input_ids_with_reasoning.shape[-1]))
            last_prompt_len = len(prompt) if last_prompt_len is None else last_prompt_len
            continue

        # Context has changed! We've crossed a boundary where reasoning removal occurs:
        # - Previous assistant messages now have a different representation in context
        # - This creates a different token sequence than what we've been accumulating
        # - Time to process the accumulated batch before continuing
        seen_tokens = kv_cache.get_seq_length()

        # Process with reasoning tokens
        torch.cuda.synchronize()
        model_start = time.perf_counter()
        logits = model(
            input_ids=last_ai_message[:, seen_tokens:].to(model.device),
            past_key_values=kv_cache
        ).logits
        torch.cuda.synchronize()
        model_forward_time += (time.perf_counter() - model_start)

        # Crop cache back to prompt and process without reasoning
        kv_cache.crop(last_prompt_len)
        input_ids_without_reasoning = tokenizer.apply_chat_template(
            [*messages[:last_ai_message_idx+1], *dummy_turn],
            tools=tools, add_generation_prompt=False,
            return_tensors="pt", tokenize=True
        )[:, :-dummy_turn_token_len]

        torch.cuda.synchronize()
        model_start = time.perf_counter()
        model(
            input_ids=input_ids_without_reasoning[:, last_prompt_len:].to(model.device),
            past_key_values=kv_cache
        )
        torch.cuda.synchronize()
        model_forward_time += (time.perf_counter() - model_start)

        # Extract logits for each assistant message
        all_logits.extend([
            logits[:, start - seen_tokens : end - seen_tokens, :]
            for start, end in ai_msg_boundaries
        ])
        
        # Reset for new context
        ai_msg_boundaries = [(len(prompt), input_ids_with_reasoning.shape[-1])]
        last_ai_message = input_ids_with_reasoning
        last_prompt_len = len(prompt)
        last_ai_message_idx = idx

    # Process final batch
    seen_tokens = kv_cache.get_seq_length()
    torch.cuda.synchronize()
    model_start = time.perf_counter()
    logits = model(
        input_ids=last_ai_message[:, seen_tokens:].to(model.device),
        past_key_values=kv_cache
    ).logits
    torch.cuda.synchronize()
    model_forward_time += (time.perf_counter() - model_start)

    all_logits.extend([
        logits[:, start - seen_tokens : end - seen_tokens, :]
        for start, end in ai_msg_boundaries
    ])
    
    return torch.cat(all_logits, dim=1), model_forward_time

# ============================================================================
# Shared Function for Custom Attention Methods
# ============================================================================

def prepare_custom_attention(tokenizer, messages, tools):
    """
    Prepare input sequences and attention masks for single forward pass with custom attention masks.
    
    This function handles the complex logic of:
    1. Creating a single sequence with strategic copying of turns:
        - A "turn" = messages that maintain same content whether as message or context
        - Messages BEFORE first assistant in each turn: shared (no copies needed)
          (includes user messages, system messages, etc.)
        - Messages FROM first assistant onward in each turn: two copies
            i) with reasoning tokens preserved
            ii) with reasoning tokens removed
        - Last turn: only one copy (won't be used as context)
    2. Building custom position IDs (different for each copy) and attention masks
    
    Example structure:
    Turn 1: [User1 (shared), Assistant1a (2 copies), Tool1 (2 copies), Assistant1b (2 copies)]
    Turn 2: [User2 (shared), Assistant2 (2 copies)]
    Turn 3: [User3 (shared), Assistant3 (1 copy only - last turn)]
    """
    prompt_token_ids = []
    token_ids_with_reasoning = []
    token_ids_without_reasoning = []

    input_ids = []
    position_ids = torch.empty(0, dtype=torch.long)
    ai_msg_boundaries = []
    turn_boundaries = []
    
    total_len_with_reasoning = total_len_without_reasoning = 0
    conv_till_last_msg = None

    for idx, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue

        # Tokenize with reasoning
        prompt_with_reasoning = tokenizer.apply_chat_template(
            messages[:idx], tools=tools, add_generation_prompt=True, tokenize=False
        )
        conv_with_reasoning = tokenizer.apply_chat_template(
            messages[:idx+1], tools=tools, add_generation_prompt=False, tokenize=False
        )
        ai_msg_with_reasoning = conv_with_reasoning[len(prompt_with_reasoning):]
        
        # Tokenize without reasoning (using dummy turn trick)
        prompt_without_reasoning = tokenizer.apply_chat_template(
            [*messages[:idx], *dummy_turn], tools=tools,
            add_generation_prompt=False, tokenize=False
        )[:-dummy_turn_len]
        conv_without_reasoning = tokenizer.apply_chat_template(
            [*messages[:idx+1], *dummy_turn], tools=tools,
            add_generation_prompt=False, tokenize=False
        )[:-dummy_turn_len]
        ai_msg_without_reasoning = conv_without_reasoning[len(prompt_without_reasoning) + gen_prompt_len:]

        # Check if we need to start a new turn
        total_len = total_len_with_reasoning
        if not (conv_till_last_msg is None or conv_till_last_msg == prompt_with_reasoning[:len(conv_till_last_msg)]):
            total_len = total_len_without_reasoning

            # Process accumulated messages
            turn_start_pos = len(input_ids) + len(prompt_token_ids[0])
            turn_without_reasoning = []
            
            # Add shared prefix (messages before first assistant in turn)
            input_ids.extend(prompt_token_ids[0])
            position_ids = torch.cat((
                position_ids,
                torch.arange(len(input_ids) - len(position_ids)) + ((position_ids[-1] + 1) if position_ids.numel() > 0 else 0)
            ))
            prompt_token_ids[0] = []
            
            # Add two copies of messages from first assistant onward
            # First copy: with reasoning tokens
            for prompt_tok, tok_with_r, tok_without_r in zip(
                prompt_token_ids, token_ids_with_reasoning, token_ids_without_reasoning
            ):
                ai_msg_boundaries.append((
                    len(input_ids) + len(prompt_tok),
                    len(input_ids) + len(prompt_tok) + len(tok_with_r)
                ))
                input_ids.extend((*prompt_tok, *tok_with_r))
                # Collect tokens for second copy (without reasoning)
                turn_without_reasoning.extend((*prompt_tok, *tok_without_r))

            turn_boundaries.append((turn_start_pos, len(input_ids)))
            
            # Add second copy (without reasoning) with its own position IDs
            position_ids = torch.cat((
                position_ids,
                torch.arange(len(input_ids) - len(position_ids)) + (position_ids[-1] + 1),
                torch.arange(len(turn_without_reasoning)) + (position_ids[-1] + 1)
            ))
            input_ids.extend(turn_without_reasoning)

            # Reset for new turn
            prompt_token_ids = []
            token_ids_with_reasoning = []
            token_ids_without_reasoning = []

        # Tokenize and store
        prompt_token_ids.append(tokenizer.encode(prompt_with_reasoning[total_len:]))
        token_ids_with_reasoning.append(tokenizer.encode(ai_msg_with_reasoning))
        token_ids_without_reasoning.append(tokenizer.encode(ai_msg_without_reasoning))

        conv_till_last_msg = conv_with_reasoning
        total_len_with_reasoning = len(conv_with_reasoning)
        total_len_without_reasoning = len(conv_without_reasoning)

    # Process final turn (last turn - only one copy needed as it won't be context)
    for prompt_tok, tok_with_r in zip(prompt_token_ids, token_ids_with_reasoning):
        ai_msg_boundaries.append((
            len(input_ids) + len(prompt_tok),
            len(input_ids) + len(prompt_tok) + len(tok_with_r)
        ))
        input_ids.extend((*prompt_tok, *tok_with_r))
    
    position_ids = torch.cat((
        position_ids,
        torch.arange(len(input_ids) - len(position_ids)) + ((position_ids[-1] + 1) if position_ids.numel() > 0 else 0)
    ))

    # Build attention mask
    attention_mask = torch.ones(len(input_ids), len(input_ids), dtype=torch.bool)
    for turn_start, turn_end in turn_boundaries:
        # Mask attention between the two copies of the same turn
        # This prevents the "without reasoning" copy from seeing the "with reasoning" copy
        attention_mask[turn_end:, turn_start:turn_end] = False
    attention_mask = attention_mask.tril(diagonal=0)  # Causal mask

    # Extract indices for assistant messages
    ai_messages_indices = torch.cat([
        torch.arange(start, end)
        for start, end, in ai_msg_boundaries
    ])

    return input_ids, position_ids, attention_mask, ai_messages_indices

# ============================================================================
# Method 3: SDPA Attention with custom attention masks
# ============================================================================

def forward_with_sdpa_attention(model, tokenizer, messages, tools):
    """
    SDPA approach: Use PyTorch's SDPA with custom attention masks.
    
    This method uses a boolean attention mask to control which tokens
    can attend to which other tokens.
    """
    # Prepare inputs with custom attention
    input_ids, position_ids, attention_mask, ai_messages_indices = \
        prepare_custom_attention(tokenizer, messages, tools)
    
    # Convert to tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.device).unsqueeze(0)
    position_ids = position_ids.unsqueeze(0).to(model.device)
    attention_mask = attention_mask[None, None].to(model.device)

    # Forward pass
    torch.cuda.synchronize()
    model_start = time.perf_counter()
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids
    ).logits
    torch.cuda.synchronize()
    model_forward_time = (time.perf_counter() - model_start)
    
    # Extract only assistant message logits
    return logits[:, ai_messages_indices, :], model_forward_time

# ============================================================================
# Method 4: Flex Attention
# ============================================================================

def forward_with_flex_attention(model, tokenizer, messages, tools):
    """
    Flex Attention approach: Use PyTorch's flexible attention implementation.
    
    This creates a block mask using a custom mask function, which can be
    more efficient than dense boolean masks for certain patterns.
    """
    # Prepare inputs with custom attention
    input_ids, position_ids, attention_mask, ai_messages_indices = \
        prepare_custom_attention(tokenizer, messages, tools)
    
    # Convert to tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.device).unsqueeze(0)
    position_ids = position_ids.unsqueeze(0).to(model.device)

    # Create block mask for flex attention
    attention_mask = attention_mask.to(model.device)
    def mask_mod(b, h, q_idx, kv_idx):
        """Define which query positions can attend to which key positions"""
        return attention_mask[q_idx, kv_idx]
    
    block_mask = create_block_mask(
        mask_mod, B=None, H=None,
        Q_LEN=input_ids.shape[-1],
        KV_LEN=input_ids.shape[-1],
        device=model.device
    )

    # Forward pass
    torch.cuda.synchronize()
    model_start = time.perf_counter()
    logits = model(
        input_ids=input_ids,
        attention_mask=block_mask,
        position_ids=position_ids
    ).logits
    torch.cuda.synchronize()
    model_forward_time = (time.perf_counter() - model_start)

    # Extract only assistant message logits
    return logits[:, ai_messages_indices, :], model_forward_time

# ============================================================================
# Method 5: Full Reasoning (Baseline with All Tokens)
# ============================================================================

# Base conversation for context separation
BASE_CHAT_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."}
]

def forward_with_full_reasoning(model, tokenizer, messages, tools):
    """
    Full Reasoning approach: Include all reasoning tokens without masking.
    
    This serves as a comparison point to see how much the reasoning tokens
    affect the final outputs when they're fully visible.
    """
    all_tokens = []
    message_boundaries = []
    turn_start_idx = 0

    # Calculate base context length for offset
    base_end_pos = len(tokenizer.apply_chat_template(
        BASE_CHAT_HISTORY, tools=tools, add_generation_prompt=False, tokenize=True
    ))
    
    for idx, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue

        # Add base context for turn separation if needed
        if turn_start_idx > 0:
            # Not the first turn - add base context for separation
            prompt_messages = [*BASE_CHAT_HISTORY, *messages[turn_start_idx:idx]]
            conv_messages = [*BASE_CHAT_HISTORY, *messages[turn_start_idx:idx+1]]
            offset = base_end_pos
        else:
            # First turn - no base context needed
            prompt_messages = messages[:idx]
            conv_messages = messages[:idx+1]
            offset = 0
        
        # Tokenize prompt and full conversation
        prompt = tokenizer.apply_chat_template(
            prompt_messages, tools=tools, add_generation_prompt=True, tokenize=True
        )[offset:]
        conv = tokenizer.apply_chat_template(
            conv_messages, tools=tools, add_generation_prompt=False, tokenize=True
        )[offset:]
        
        # Track message boundaries
        message_boundaries.append((len(all_tokens) + len(prompt), len(all_tokens) + len(conv)))
        all_tokens.extend(conv)
        turn_start_idx = idx + 1

    # Forward pass
    input_ids = torch.tensor(all_tokens, dtype=torch.long, device=model.device).unsqueeze(0)
    torch.cuda.synchronize()
    model_start = time.perf_counter()
    logits = model(input_ids=input_ids).logits
    torch.cuda.synchronize()
    model_forward_time = (time.perf_counter() - model_start)

    # Extract assistant message logits
    ai_messages_indices = torch.cat([
        torch.arange(start, end)
        for start, end in message_boundaries
    ])
    
    return logits[:, ai_messages_indices, :], model_forward_time

# ============================================================================
# Main Benchmark Loop
# ============================================================================

acc_metrics = []

# Process each sample in the dataset
for i, sample in tqdm(enumerate(data), total=len(data), desc="Processing"):
    # Run all methods
    logits_naive, naive_perf = measure_forward(
        forward_naive, model0, tokenizer, sample["messages"], sample["tools"]
    )
    logits_kv_cache, kv_cache_perf = measure_forward(
        forward_with_kv_cache, model1, tokenizer, sample["messages"], sample["tools"]
    )
    logits_sdpa, sdpa_perf = measure_forward(
        forward_with_sdpa_attention, model2, tokenizer, sample["messages"], sample["tools"]
    )
    logits_flex, flex_perf = measure_forward(
        forward_with_flex_attention, model3, tokenizer, sample["messages"], sample["tools"]
    )
    logits_full_reasoning, full_reasoning_perf = measure_forward(
        forward_with_full_reasoning, model4, tokenizer, sample["messages"], sample["tools"]
    )

    # Store results
    acc_metrics.append({
        "index": i,
        "kv_cache": {
            "accuracy": compare_logits(logits_naive, logits_kv_cache),
            "performance": kv_cache_perf
        },
        "sdpa": {
            "accuracy": compare_logits(logits_naive, logits_sdpa),
            "performance": sdpa_perf
        },
        "flex": {
            "accuracy": compare_logits(logits_naive, logits_flex),
            "performance": flex_perf
        },
        "full_reasoning": {
            "accuracy": compare_logits(logits_naive, logits_full_reasoning),
            "performance": full_reasoning_perf
        },
        "naive": {
            "performance": naive_perf
        }
    })

    # Clean up GPU memory
    del logits_naive, logits_kv_cache, logits_sdpa, logits_flex
    for model in [model0, model1, model2, model3, model4]:
        torch.cuda.reset_peak_memory_stats(device=model.device)

# Save detailed results
with open("/home/jobuser/acc_metrics.json", "w") as f:
    json.dump(acc_metrics, f, indent=4)

# ============================================================================
# Print Summary Results
# ============================================================================

print("\n=== Performance Summary ===")
methods = ["naive", "kv_cache", "sdpa", "flex", "full_reasoning"]

# Skip first 42 samples for warmup
for method in methods:
    metrics_list = [m[method]["performance"] for m in acc_metrics[42:]]

    total_times = [m["total_time"] for m in metrics_list]
    model_times = [m["model_forward_time"] for m in metrics_list]
    memories = [m["max_memory_mb"] for m in metrics_list]

    print(f"\n{method.upper()}:")
    print(f"  Total time AVG: {sum(total_times)/len(total_times):.4f}s")
    print(f"  Total time P99: {np.percentile(total_times, 99):.4f}s")
    print(f"  Model forward time AVG: {sum(model_times)/len(model_times):.4f}s ({sum(model_times)/sum(total_times)*100:.1f}%)")
    print(f"  Model forward time P99: {np.percentile(model_times, 99):.4f}s")
    print(f"  Other time AVG: {(sum(total_times)-sum(model_times))/len(total_times):.4f}s ({(sum(total_times)-sum(model_times))/sum(total_times)*100:.1f}%)")
    print(f"  Other time P99: {np.percentile(total_times, 99) - np.percentile(model_times, 99):.4f}s")
    print(f"  Max memory: {max(memories):.2f} MB")

print("\n=== Accuracy Summary ===")
for method in ["kv_cache", "sdpa", "flex", "full_reasoning"]:
    metrics = [m[method]["accuracy"] for m in acc_metrics]
    print(f"\n{method.upper()} vs NAIVE:")
    print(f"  Avg close %: {sum(m['close_percent'] for m in metrics)/len(metrics):.4f}%")
    print(f"  Avg RMSE: {sum(m['rmse'] for m in metrics)/len(metrics):.6f}")
    print(f"  Avg KL: {sum(m['kl_symmetric'] for m in metrics)/len(metrics):.6f}")
    print(f"  Avg Top-1: {sum(m['top1_overlap'] for m in metrics)/len(metrics)*100:.2f}%")
    print(f"  Avg Top-8: {sum(m['top8_overlap'] for m in metrics)/len(metrics)*100:.2f}%")
