---
title: "Efficient Forward Pass for Agent RL: Solving Multi-Turn Context Consistency (Part 1)"
date: 2025-06-29 09:00:00 -0700
categories: [LLM]
tags: [flash-attention, flex-attention, sdpa-attention, kv-cache, verl, agent-rl, llm-infra, multi-turn]
author: yanbin_jiang
toc: true
pin: true
---

After implementing correct and scalable tokenization and masking[^1] for multi-turn rollout, there remains a critical challenge to achieve full consistency between training and inference: the context discrepancy problem.

# The Training-Inference Context Mismatch

Let me recap the issue I highlighted at [the end of the multi-turn tokenization and masking blog](https://jybsuper.github.io/posts/multiturn_tokenization/#next-steps). Currently, we face two different chat template patterns during training and inference of reasoning models. 

Consider a simple conversation with alternating human queries and assistant responses, where each assistant message contains both reasoning and response components:

**Human Query 1** → **Assistant Message 1** (reasoning + response) → **Human Query 2** → **Assistant Message 2** (reasoning + response) → ...

During inference, reasoning models' chat templates **remove reasoning content from previous turns** when generating prompts for new turns. As illustrated below, each turn only retains the response portion of previous assistant messages while keeping the full reasoning + response for the current turn:

![Multi-turn Inference Context](/assets/img/posts/multiturn-forward-pass/inference-context.excalidraw.png)

However, during training, we must preserve the reasoning content from each assistant message to enable the model to learn reasoning capabilities. For each sample, training frameworks typically pack all turns within that sample together and perform a single forward pass for loss or log probability calculation, which means the computation includes the complete reasoning content for all assistant messages:

![Multi-turn Training Context](/assets/img/posts/multiturn-forward-pass/training-context.excalidraw.png)

This creates a fundamental discrepancy: inference-time models operate without previous reasoning content in their context, while training-time models have access to all previous reasoning content. In this blog, I'll explore different approaches to solve this problem efficiently and provide comprehensive comparisons.

# Approaches to Bridge the Training-Inference Gap

I've identified three potential solutions to address this context mismatch. To evaluate their feasibility for VeRL, I built prototypes using Qwen3-4B and analyzed their practicality.

## Base Solution: Turn-by-Turn Forward Passes

The most straightforward approach is to abandon turn packing entirely and run separate forward passes for each turn, mimicking the inference pattern exactly.

### Why Standard Multi-Turn Training Works

Before diving into this solution, let's understand why typical multi-turn training can pack all turns into a single forward pass. The key insight is that **standard models maintain content immutability**: once tokens are generated in a turn, they remain unchanged when included as context for future turns.

This immutability enables an important optimization:
- **Single-pass training**: Calculate loss for Turn 1, Turn 2, and Turn 3 messages all in one forward pass
- **Multi-pass inference**: Run three separate forward passes (one per turn) — this is the inherent nature of autoregressive generation, not a training-imposed requirement

These produce identical results because the tokens from earlier turns don't change based on their position in the conversation. The model sees the exact same token sequences whether processed together or separately.

### Why Reasoning Models Break This Assumption

Reasoning models violate this immutability principle. When an assistant message with reasoning content becomes part of the chat history, the model's chat template **removes the reasoning tokens**. This means:

- **Training forward pass**: Sees `Human 1 → Assistant 1 (reasoning + response) → Human 2 → Assistant 2 (reasoning + response)`
- **Inference forward passes**: 
  - Turn 2 sees: `Human 1 → Assistant 1 (response only) → Human 2 → Assistant 2 (reasoning + response)`
  - Turn 3 sees: `Human 1 → Assistant 1 (response only) → Human 2 → Assistant 2 (response only) → Human 3 → Assistant 3 (reasoning + response)`

The model is effectively trained on contexts it will never see during inference, creating a distribution mismatch.

### The Turn-by-Turn Approach

This solution restores correctness by mimicking inference behavior during training:
1. Process each turn individually with separate forward passes
2. For each turn, apply the same context modifications (reasoning removal) used during inference
3. Calculate loss only on the current turn's assistant response
4. Aggregate the losses from all turns to get the final loss for the sample

![Base Solution: Turn-by-Turn Forward Passes](/assets/img/posts/multiturn-forward-pass/unpacked-multiturn-forward.excalidraw.png)

This solution serves as our correctness baseline. While computationally expensive, it provides a reference point to validate the outputs of more optimized approaches.

### Reference Implementation

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Test conversation with reasoning in assistant messages
messages = [
    {"role": "user", "content": "What is 15 + 27?"},
    {"role": "assistant", "content": "<think>I need to add 15 and 27. 15 + 27 = 42.</think>The answer is 42."},
    {"role": "user", "content": "Now multiply that by 3."},
    {"role": "assistant", "content": "<think>The previous result was 42. 42 × 3 = 126.</think>42 times 3 equals 126."},
    {"role": "user", "content": "What's half of that?"},
    {"role": "assistant", "content": "<think>Half of 126 is 126 ÷ 2 = 63.</think>Half of 126 is 63."}
]

# Same messages without reasoning (for context building)
messages_wo_reasoning = [
    {"role": "user", "content": "What is 15 + 27?"},
    {"role": "assistant", "content": "The answer is 42."},
    {"role": "user", "content": "Now multiply that by 3."},
    {"role": "assistant", "content": "42 times 3 equals 126."},
    {"role": "user", "content": "What's half of that?"},
    {"role": "assistant", "content": "Half of 126 is 63."}
]

assistant_message_indices = [1, 3, 5]

# Initialize model
model_id = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda:0")

# Process each turn separately
all_logits = []
for idx in assistant_message_indices:
    prompt = tokenizer.apply_chat_template(
        messages[:idx], add_generation_prompt=True, tokenize=True
    )
    # Full conversation including current assistant response
    input_ids = tokenizer.apply_chat_template(
        messages[:idx+1],
        add_generation_prompt=False,
        return_tensors="pt",
        tokenize=True
    ).to(model.device)
    # Forward pass and extract assistant response logits
    all_logits.append(model(input_ids=input_ids).logits[:, len(prompt):, :])

final_logits_base = torch.cat(all_logits, dim=1)
```

Since optimizations like KV caching involve different computation paths and CUDA kernels, numerical differences are inevitable. Here's how I measure the magnitude of these differences:

```python
def compare_logits(logits1, logits2):
    """Compare two logit tensors using multiple metrics."""
    flat1 = logits1.view(-1, logits1.size(-1))
    flat2 = logits2.view(-1, logits2.size(-1))
    rmse = (flat1 - flat2).pow(2).mean().sqrt().item()
    print(f"RMSE Distance: {rmse}")

    from torch.nn.functional import softmax, kl_div
    # KL divergence: KL(P||Q) where P is the reference distribution
    # Computing KL(logits2||logits1) - how much information is lost when using logits1 to approximate logits2
    kl = kl_div(softmax(logits1, dim=-1).log(), softmax(logits2, dim=-1), reduction='batchmean').item()
    print(f"KL divergence (logits||expected): {kl}")
    # Also compute the reverse KL for completeness
    kl_reverse = kl_div(softmax(logits2, dim=-1).log(), softmax(logits1, dim=-1), reduction='batchmean').item()
    print(f"KL divergence (expected||logits): {kl_reverse}")
    # Symmetric KL divergence (average of both directions)
    print(f"Symmetric KL divergence: {(kl + kl_reverse) / 2}")

    def topk_overlap(a, b, k):
        ta = torch.topk(a, k, dim=-1).indices
        tb = torch.topk(b, k, dim=-1).indices
        return (ta.unsqueeze(-1) == tb.unsqueeze(-2)).any(dim=-1).float().mean().item()

    print(f"Top-1 overlap: {topk_overlap(logits1, logits2, 1) * 100:.2f}%")
    print(f"Top-8 overlap: {topk_overlap(logits1, logits2, 8) * 100:.2f}%")

    try:
        torch.testing.assert_close(logits2, logits1, rtol=1e-1, atol=1e-2)
    except Exception as e:
        print(e)
```

While this guarantees training-inference consistency, it comes with significant costs:
- **Computational overhead**: Redundant processing of shared chat history across turns
- **Implementation complexity**: Requires a different training loop pattern compared to standard single-turn or packed multi-turn training

## Optimized Solution 1: KV Cache Acceleration

Since the base solution mirrors the inference pattern, we can apply the same optimization technique used during inference: KV caching. This approach maintains correctness while significantly reducing computational redundancy.

HuggingFace models include built-in KV cache support via the `transformers` library[^2], allowing us to cache key-value projections of past messages for future turns:

### The KV Cache Strategy

1. **Process assistant message**: Run forward pass for the i-th assistant message (with reasoning)
2. **Extract KV cache**: Save the key-value projections from the forward pass output
3. **Crop to prompt**: Remove KV entries for the assistant's response, keeping only the prompt portion
4. **Rebuild without reasoning**: Process the same assistant message without reasoning content to update the KV cache
5. **Continue to next turn**: Use the updated KV cache for processing subsequent turns

This approach eliminates redundant attention computations on shared chat history:

![KV Cache Optimization](/assets/img/posts/multiturn-forward-pass/KV-cache.png)

### Implementation and Validation

Using the same conversation setup from the base solution, here's the optimized KV cache version:

```python
from transformers.cache_utils import OffloadedCache

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda:1")
# Initialize KV cache
kv_cache = OffloadedCache()

all_logits = []
for idx in assistant_message_indices:
    prompt = tokenizer.apply_chat_template(messages[:idx], add_generation_prompt=True, tokenize=True)
    input_ids_w_think = tokenizer.apply_chat_template(
        messages[:idx+1],
        add_generation_prompt=False,
        return_tensors="pt",
        tokenize=True
    ).to(model.device)
    # Forward pass using cached KV states
    all_logits.append(
        model(
            input_ids=input_ids_w_think[:, kv_cache.get_seq_length():],
            past_key_values=kv_cache
        ).logits[:, len(prompt) - input_ids_w_think.shape[-1]:, :]  # Extract logits for current assistant message
    )

    # Remove assistant response from KV cache
    kv_cache.crop(len(prompt))
    
    # Rebuild cache with reasoning-free version
    # Note: Qwen3 adds empty <think> tags for non-reasoning messages
    inputs_wo_think = tokenizer.apply_chat_template(
        messages_wo_reasoning[:idx+1], 
        add_generation_prompt=False, 
        tokenize=False
    ).replace("<think>\n\n</think>\n\n", '')  # Remove empty tags
    input_ids_wo_think = tokenizer.encode(inputs_wo_think, return_tensors="pt").to(model.device)

    # Update KV cache with reasoning-free context
    model(input_ids=input_ids_wo_think[:, kv_cache.get_seq_length():], past_key_values=kv_cache)

final_logits_kv_cache = torch.cat(all_logits, dim=1)
```

This optimization reduces the computational complexity from O(n²) to O(n) for processing n turns, making it much more efficient while maintaining exact consistency with inference behavior.

### Numerical Accuracy Analysis

When comparing the KV cache optimization against the reference implementation, I observe small but notable numerical differences:

```
RMSE Distance: 0.0791015625
KL divergence (logits||expected): 0.042236328125
KL divergence (expected||logits): 0.033203125
Symmetric KL divergence: 0.0377197265625
Top-1 overlap: 99.10%
Top-8 overlap: 99.66%
Tensor-likes are not close!

Mismatched elements: 1508533 / 16864896 (8.9%)
Greatest absolute difference: 0.90625 at index (0, 54, 4969) (up to 0.01 allowed)
Greatest relative difference: 610304.0 at index (0, 76, 52622) (up to 0.1 allowed)
```

These differences arise from different CUDA kernel dispatch patterns:

1. **Linear Layer Kernels**: When computing Q, K, V projections, the KV cache version processes only new tokens since previously computed projections are cached. This difference in sequence length causes PyTorch to dispatch different GEMM kernels:
   - Base solution → Segmented K GEMM kernel
   - KV cache → SplitK GEMM kernel
   
   The SplitK kernel is chosen for shorter sequences because when the M dimension (sequence length: 37 tokens in this example) is small relative to the K dimension (hidden dimension: 2560 for Qwen3-4B), it's more efficient to split the reduction work across multiple thread blocks. Longer sequences with larger M dimensions benefit from the segmented approach instead.

2. **Attention Kernels**: Qwen3 uses PyTorch's SDPA attention implementation[^3] by default, which automatically selects the most optimal backend among Flash Attention 2, xFormers, and PyTorch's native C++ implementation based on hardware and input characteristics. HuggingFace models are configured to enable Flash Attention 2 as the SDPA backend when possible. In this case, both base and KV cache solutions routed to Flash Attention 2, but the different query and key-value sequence lengths triggered different kernels within Flash Attention:
   - Base solution → `flash_fwd_kernel` (standard Flash Attention kernel)
   - KV cache → `flash_fwd_splitkv_kernel` (Flash Attention's kernel variant for handling different Q and KV sequence lengths, which occurs in KV caching scenarios)

These numerical differences show the following characteristics:
- 99%+ top-1 token prediction overlap
- Low KL divergence indicating similar probability distributions
- The differences can be eliminated by forcing the same kernel paths, confirming they're purely computational artifacts rather than algorithmic issues

## Optimized Solution 2: Custom 2D Attention Mask

Another approach leverages customized 2D attention masks to selectively control which tokens can attend to each other, enabling single-pass training while maintaining inference-like context visibility.

### Understanding Attention Masks

When tokenizing text with HuggingFace tokenizers, the returned attention mask is a 1D binary mask indicating valid tokens (1) versus padding (0). However, this is not the mask used in actual attention computation.

The attention mechanism uses a 2D mask of shape `[seq_len_q, seq_len_k]` that specifies which query positions can attend to which key positions. For causal language models, the `is_causal=True` flag generates a lower-triangular mask, ensuring each token only attends to previous tokens and itself:

![Attention Mask Mechanism](/assets/img/posts/multiturn-forward-pass/attention-mask.png)

This mask is converted to an attention bias by replacing:
- `True` or `1` (attend) → 0
- `False` or `0` (ignore) → -∞

After adding this bias to the QK dot product and applying softmax, ignored positions become 0, effectively removing their contribution.

### Duplicating Messages with Custom Masks

By crafting a custom 2D attention mask, we can make assistant messages attend only to their own reasoning content while ignoring reasoning from previous assistant messages — all within a single forward pass. This approach requires:

1. **Duplicate Assistant Messages**: Include each assistant message twice in the input sequence:
   - First copy: Without reasoning (for context)
   - Second copy: With reasoning (for learning)

2. **Custom Attention Patterns**: Design the mask so that:
   - All tokens attend to non-reasoning versions of previous assistant messages
   - Current assistant message tokens attend to their own reasoning content
   - Human and other messages follow standard causal attention

3. **Adjusted Position IDs**: Since we have duplicate content, position IDs no longer monotonically increase. Each duplicated token pair shares the same position ID to maintain positional consistency:

![Custom 2D Attention Mask and Position IDs](/assets/img/posts/multiturn-forward-pass/2D-custom-attention-mask.png)

In this visualization (shown at message level for clarity), you can see how:
- Green cells indicate where tokens can attend
- Red cells block attention to reasoning content from previous turns
- Position IDs properly handle the duplicate messages

### Attention Backend Support

Different attention implementations vary in their support for custom attention masks:

**Flash Attention 2** - No custom mask support:
```python
# Flash Attention only supports boolean causal flag, not custom masks (applies to both flash_attn_func and flash_attn_varlen_func)
flash_attn.flash_attn_func(
    q, k, v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,  # Boolean only - no custom masks
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
)
```

**PyTorch SDPA Attention** - Limited custom mask support:
```python
# SDPA accepts custom attention masks
torch.nn.functional.scaled_dot_product_attention(
    query, key, value, 
    attn_mask=custom_mask,  # Custom mask supported
    dropout_p=0.0, 
    is_causal=False,
    scale=None,
    enable_gqa=False
)
```
However, passing a custom `attn_mask` prevents SDPA from using the Flash Attention backend. It falls back to either:
- xFormers: Limited experimental support for GQA (Group Query Attention)
- PyTorch C++ Attention Implementation: Slower and more memory-intensive

For models using GQA (like Qwen3), this often means falling back to the native implementation, negating performance benefits.

**PyTorch FlexAttention**[^4] - Flexible mask and scoring support:
```python
# Flex Attention provides flexible mask and scoring options
torch.nn.attention.flex_attention.flex_attention(
    query, key, value,
    score_mod=custom_score_function,  # Custom attention score bias
    block_mask=custom_block_mask,     # Custom attention patterns
    scale=None,
    enable_gqa=False,
    return_lse=False,
    kernel_options=None
)
```

FlexAttention not only supports arbitrary attention patterns but can also [leverage sparsity in the attention mask for performance improvements](https://pytorch.org/blog/flexattention/). By analyzing which blocks need computation, it can skip unnecessary operations entirely.

Given these constraints, I evaluated both SDPA (with its limitations) and Flex Attention for implementing the custom 2D mask approach.

### Implementation and Validation

#### Shared Input Preparation

Both SDPA and FlexAttention implementations share the same input preparation logic for constructing token IDs, position IDs, and attention masks:

```python
# Initialize tracking variables
current_turn_start = 0
assistant_tokens_offset = 0
all_token_ids = []
all_position_ids = []

# Track boundaries for mask construction and logits extraction
# Each tuple: (start_idx_with_reasoning, start_idx_without_reasoning)
assistant_message_boundaries = []

# Process each assistant turn
for turn_idx in assistant_message_indices:
    # Tokenize prompt (everything before current assistant message)
    prompt_tokens = tokenizer.apply_chat_template(
        messages[:turn_idx], 
        add_generation_prompt=True, 
        tokenize=True
    )

    # Tokenize full conversation including current assistant (with reasoning)
    tokens_with_reasoning = tokenizer.apply_chat_template(
        messages[:turn_idx+1], 
        add_generation_prompt=False, 
        return_tensors="pt", 
        tokenize=True
    )

    # Tokenize conversation with reasoning removed
    conv_without_reasoning = tokenizer.apply_chat_template(
        messages_wo_reasoning[:turn_idx+1], 
        add_generation_prompt=False, 
        tokenize=False
    ).replace("<think>\n\n</think>\n\n", '')  # Remove empty think tags
    tokens_without_reasoning = tokenizer.encode(conv_without_reasoning, return_tensors="pt")

    # Concatenate: [previous_tokens][assistant_with_reasoning][assistant_without_reasoning]
    turn_tokens = torch.cat([
        tokens_with_reasoning[:, current_turn_start:],  # New tokens from this turn
        tokens_without_reasoning[:, len(prompt_tokens):]  # Assistant without reasoning
    ], dim=1)
    all_token_ids.append(turn_tokens)

    # Generate position IDs (duplicate positions for duplicate content)
    turn_position_ids = torch.cat([
        torch.arange(current_turn_start, tokens_with_reasoning.shape[-1], dtype=torch.long),
        torch.arange(len(prompt_tokens), tokens_without_reasoning.shape[-1], dtype=torch.long)
    ]).unsqueeze(0)
    all_position_ids.append(turn_position_ids)
    
    # Track boundaries for attention mask and logit extraction
    reasoning_start = assistant_tokens_offset + len(prompt_tokens)
    no_reasoning_start = assistant_tokens_offset + tokens_with_reasoning.shape[-1]
    assistant_message_boundaries.append((reasoning_start, no_reasoning_start))
    
    # Update tracking variables
    current_turn_start = tokens_without_reasoning.shape[-1]
    assistant_tokens_offset += tokens_with_reasoning.shape[-1] - len(prompt_tokens)

# Combine all inputs
input_ids = torch.cat(all_token_ids, dim=1)
position_ids = torch.cat(all_position_ids, dim=1)

# Build custom 2D attention mask
# Shape: [batch_size, num_heads=1, seq_length, seq_length]
seq_length = input_ids.shape[1]
attention_mask = torch.ones(
    input_ids.shape[0], 1, seq_length, seq_length, 
    dtype=torch.bool
)

# Apply masking rules: block attention from future tokens to past reasoning content
for reasoning_start, no_reasoning_start in assistant_message_boundaries:
    attention_mask[:, :, no_reasoning_start:, reasoning_start:no_reasoning_start] = False

# Apply causal mask (lower triangular)
attention_mask = attention_mask.tril(diagonal=0)
```

#### SDPA Implementation

```python
# Qwen3 uses SDPA attention by default
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda:2"
)

# Forward pass with custom attention mask
all_logits = model(
    input_ids=input_ids.to(model.device), 
    attention_mask=attention_mask.to(model.device), 
    position_ids=position_ids.to(model.device)
).logits

# Extract logits only for assistant messages with reasoning
assistant_logits_list = [
    all_logits[:, reasoning_start:no_reasoning_start, :] 
    for reasoning_start, no_reasoning_start in assistant_message_boundaries
]
final_logits_sdpa_custom_mask = torch.cat(assistant_logits_list, dim=1)
```

##### Numerical Accuracy Analysis

```
RMSE Distance: 0.0791015625
KL divergence (logits||expected): 0.058837890625
KL divergence (expected||logits): 0.0257568359375
Symmetric KL divergence: 0.04229736328125
Top-1 overlap: 98.20%
Top-8 overlap: 99.10%
Tensor-likes are not close!

Mismatched elements: 2035728 / 16864896 (12.1%)
Greatest absolute difference: 0.921875 at index (0, 89, 59151) (up to 0.01 allowed)
Greatest relative difference: 1384448.0 at index (0, 76, 52622) (up to 0.1 allowed)
```

The numerical differences are slightly higher than the KV cache approach. The linear layers use the same SegmentK GEMM kernel as the base implementation due to the longer sequence length. However, SDPA falls back to PyTorch's native C++ attention implementation instead of Flash Attention 2 when using custom attention masks and GQA, which likely contributes to the larger numerical discrepancies.

#### FlexAttention Implementation

```python
from torch.nn.attention.flex_attention import create_block_mask

# Load model with FlexAttention backend
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    attn_implementation="flex_attention", 
    device_map="cuda:3"
)

# Convert attention mask to FlexAttention's block mask format
# Extract the head-shared mask (all heads use the same pattern)
head_shared_mask = attention_mask[:, 0, :, :].to(model.device)

# Define mask modification function for FlexAttention
def mask_mod(b, h, q_idx, kv_idx):
    return head_shared_mask[b, q_idx, kv_idx]

# Create block mask optimized for sparse attention patterns
block_mask = create_block_mask(
    mask_mod, 
    B=input_ids.shape[0], 
    H=None,  # Broadcast across all heads
    Q_LEN=seq_length, 
    KV_LEN=seq_length, 
    device=model.device
)

# Forward pass with FlexAttention block mask
all_logits = model(
    input_ids=input_ids.to(model.device), 
    attention_mask=block_mask, 
    position_ids=position_ids.to(model.device)
).logits

# Extract logits only for assistant messages with reasoning
assistant_logits_list = [
    all_logits[:, reasoning_start:no_reasoning_start, :] 
    for reasoning_start, no_reasoning_start in assistant_message_boundaries
]
final_logits_flex_custom_mask = torch.cat(assistant_logits_list, dim=1)
```

##### Numerical Accuracy Analysis

```
RMSE Distance: 0.08740234375
KL divergence (logits||expected): 0.0966796875
KL divergence (expected||logits): 0.08837890625
Symmetric KL divergence: 0.092529296875
Top-1 overlap: 99.10%
Top-8 overlap: 99.21%
Tensor-likes are not close!

Mismatched elements: 2237592 / 16864896 (13.3%)
Greatest absolute difference: 1.0 at index (0, 89, 59151) (up to 0.01 allowed)
Greatest relative difference: 720896.0 at index (0, 0, 24300) (up to 0.1 allowed)
```

The numerical differences are comparable to the SDPA implementation. FlexAttention uses entirely different Triton-based kernels for both the attention computation and user-provided functions, which explains the similar magnitude of differences from the base implementation.

# Next Steps

Having verified the feasibility of these solutions, Part 2 will provide comprehensive benchmarks across multiple dimensions:

- **Performance Analysis**: Throughput and latency comparisons across varying sequence lengths and batch sizes
- **Numerical Accuracy**: Statistical analysis of differences across diverse multi-turn datasets
- **Memory Efficiency**: Peak memory usage and allocation patterns for each approach
- **Scalability**: How each solution performs with increasing conversation turns and model sizes

These benchmarks will help determine the optimal approach for different use cases in agent reinforcement learning scenarios.

<br>

---

[^1]: See my previous post on [When Reasoning Models Break Tokenization: The Hidden Complexity of Multiturn Training](https://jybsuper.github.io/posts/multiturn_tokenization/)

[^2]: [Transformers KV cache strategies](https://huggingface.co/docs/transformers/en/kv_cache)

[^3]: [PyTorch SDPA attention](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html)

[^4]: [PyTorch FlexAttention](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html#module-torch.nn.attention.flex_attention)
