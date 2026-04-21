---
title: "SGLang + LoRA Deep Dive — Qwen3-30B-A3B-Instruct-2507"
date: 2026-04-21 14:00:00 -0700
categories: [LLM]
tags: [sglang, lora, qwen3, moe, tensor-parallel, expert-parallel, cuda-graph, triton, llm-infra]
author: yanbin_jiang
toc: true
pin: true
math: false
---

> **Audited against SGLang main:** `1ebe1c57eddd0ea7915b408e35a1b9b33cd10c41` (2026-04-19)  
> **Audited against HF transformers main:** `631d45082cbd23f4146f1e79c37b3875a3dbc4f4` (2026-04-20)  
> **Model:** <a href="https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507" target="_blank" rel="noopener noreferrer">Qwen/Qwen3-30B-A3B-Instruct-2507</a> — bfloat16, 30.5 B total / ~3 B active, 48 layers, 128 experts per layer, top-8 routing, 262 144 native context. All line numbers, function names, and tensor shapes in this document are quoted directly from these commits. Where a claim depends on a value that would only appear at runtime (e.g. "KV cache: #tokens=X"), the number is clearly labeled as *derived* from the config rather than *observed*.
{: .prompt-info }

## 0 · Intro & how to read this
{: #intro }

This document traces what actually happens when you run:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --tp 4 \
    --context-length 32768 \
    --enable-lora \
    --lora-paths adapter0=/path/to/my-qwen3-lora \
    --max-lora-rank 64 \
    --lora-target-modules all
```

The goal is that a reader who understands how an LLM works (attention, MLP, MoE routing, LoRA math) can close the SGLang repo and still know what every component does, how it's wired, and what the intermediate tensor shapes are on the way. Every factual claim is backed by a code block pulled directly from the source. Links go to the `main` branch at the commit pinned above; if a function has moved by the time you read this, use the function name to grep.

Each section follows the same three-part shape:

1. **The code.** A code snippet with a header line that links to the exact lines on GitHub.
2. **What it does.** Prose walking through the snippet step by step.
3. **Example values.** What the variables hold for Qwen3-30B-A3B-Instruct-2507 at TP=4, bf16.

Where a design decision would surprise a reader, there's a *motivation* callout that cites the paper, blog post, PR, or code comment that justifies it.

<div class="callout info" markdown="1">

#### Badge conventions

<span class="badge badge-hf">HF</span> Hugging Face transformers code. <span class="badge badge-sg">SG</span> SGLang code. <span class="badge badge-tr">TR</span> The Qwen3 team's training-time checkpoint layout.

</div>

---

## System overview — the whole stack at a glance
{: #overview }

Before we zoom in, here's the full system you're about to explore. Keep this section open as a map while reading the rest; everything below expands on something you see here.

### The processes

A single SGLang server running our command (`--tp 4 --enable-lora`) forks into a tree of processes. Each box is its own OS process; arrows are ZMQ IPC sockets between them:

```text
           HTTP request
                │
                ▼
┌───────────────────────────────────┐
│  Main process                     │     ← launched by `python -m sglang.launch_server`
│  ├─ uvicorn / FastAPI             │       §2.5  (http_server)
│  └─ TokenizerManager              │       §3    (tokenizes, owns LoRA registry)
└────────────┬──────────────────────┘
             │ ZMQ PUSH  (scheduler_input_ipc_name)
             ▼
┌───────────────────────────────────┐
│  Scheduler subprocess  (tp=0)     │     ← one per TP rank; rank 0 is "leader"
│  ├─ Scheduler (event loop)        │       §4
│  ├─ TpModelWorker                 │       §5 (wraps ModelRunner)
│  │   └─ ModelRunner               │       §5 (weights, KV pool, graphs)
│  │       ├─ Qwen3MoeForCausalLM   │       §1 (the actual PyTorch model)
│  │       └─ LoRAManager           │       §6 (adapter wrappers + memory pool)
│  └─ (NCCL all-reduces with other ranks)   §8
└────────────┬──────────────────────┘
             │ ZMQ PUSH  (detokenizer_ipc_name)
             ▼
┌───────────────────────────────────┐
│  DetokenizerManager subprocess    │
│  └─ decodes token IDs → strings   │
└────────────┬──────────────────────┘
             │ ZMQ PULL back to TokenizerManager, then HTTP stream out
             ▼
         user sees tokens

Plus, in parallel for TP=4:
    Scheduler tp=1, tp=2, tp=3  — peers of tp=0, exchanging tensors via NCCL,
                                  not via ZMQ. Only tp=0 talks to TokenizerManager.
```

### The request lifecycle

Once the server is running, a single request traverses these stages:

```text
1. HTTP POST /generate  ──▶  TokenizerManager
2. Tokenize prompt  ──▶  input_ids (list of int)
3. Resolve lora_name → UUID via LoRARegistry   (§3.3)
4. ZMQ PUSH to Scheduler
5. Scheduler appends to waiting_queue
6. Scheduler's event loop: pick next batch
   ├─ Maybe schedule prefill (new tokens)
   └─ Maybe schedule decode (existing tokens)
7. Build ForwardBatch  (§5)
   ├─ LoRAManager.prepare_lora_batch → weight_indices, permutation  (§6.7)
   └─ Allocate KV slots from TokenToKVPoolAllocator
8. ModelRunner.forward()  ──▶  GPU
   ├─ Each layer: base QKV + LoRA delta; MoE routing; attention via FA3
   └─ Returns logits for last token of each request
9. Sampler picks token IDs
10. Scheduler: pop_and_process  (§4.5)
    ├─ Is this request finished?  (EOS / length / stop tokens)
    └─ Emit result to DetokenizerManager
11. Detokenizer: token IDs → text chunks → back to TokenizerManager
12. TokenizerManager: stream chunk over HTTP to caller

Decode tokens repeat steps 6-12 until the request finishes.
```

### The architecture by memory ownership

Understanding what lives where helps a lot. In one scheduler subprocess (one TP rank):

| Memory region                                     | Who owns it        | Approx. size (Qwen3 TP=4, H200)             | §          |
|---------------------------------------------------|--------------------|---------------------------------------------|------------|
| Model weights (sharded)                           | `ModelRunner`      | ~14.5 GB (of 60 GB total, sharded across 4) | §5.4, §5.5 |
| KV pool (flat, `page_size=1`)                     | `MHATokenToKVPool` | ~105 GB (dominates)                         | §5.9       |
| LoRA memory pool (A, B buffers per target module) | `LoRAMemoryPool`   | ~0.3 GB per 8 adapters × rank 16            | §6.3       |
| CUDA graph buffers (one graph per bucket bs)      | `CudaGraphRunner`  | ~1 GB (all buckets)                         | §5.11      |
| Activation scratch, CUBLAS/NCCL/FA workspaces     | PyTorch/CUDA       | ~0.3-0.5 GB                                 | —          |

### Parallelism vocabulary (referenced throughout, detailed in §8–9)

Five acronyms get thrown around a lot. Here's the short version so the rest of the doc reads without cross-referencing:

- **TP (Tensor Parallel).** Shard a single weight matrix across N GPUs; they all-reduce partial results. Our run has `--tp 4`. Cost: bandwidth for all-reduces, scales within a node. See §8.1–8.2.
- **EP (Expert Parallel).** MoE-specific. Shard *which experts* each GPU owns, instead of sharding each expert's weights. Dispatch tokens to the GPU owning the right expert. Enabled with `--enable-ep`. See §8.3–8.4.
- **PP (Pipeline Parallel).** Split layers across N GPU groups (e.g., layers 0-23 on group 0, 24-47 on group 1). Forward passes flow as a pipeline. Our run doesn't use it. See §9.1.
- **DP (Data Parallel).** *Two distinct things* in SGLang share this name. DP-replication runs N independent copies of the model on different request streams. "DP attention" uses data-parallelism only for the attention layers while keeping MoE in EP. See §9.3–9.4.
- **CP (Context Parallel).** Shard a single long prompt's attention computation across N GPUs during prefill. Used only for very long contexts. See §9.2.

### The roadmap

Each Part builds on the previous. One-paragraph guide:

1. **§1 — The model.** What's on disk for Qwen3-30B-A3B. `config.json`, safetensors shape, the HF class hierarchy, and an MoE step-by-step with the router, top-k, and expert weights laid out precisely. Read this to have the target layer structure in mind.
2. **§2 — Launching the server.** `launch_server.py` → `prepare_server_args` → `Engine._launch_subprocesses` → ZMQ socket setup. Where the process tree above comes from. Ends with how ZMQ (within a node) and NCCL/Gloo (across nodes) split the IPC responsibilities.
3. **§3 — TokenizerManager.** Lives in the main process. Tokenizer + IPC channels + LoRA naming registry. Short.
4. **§4 — Scheduler subprocess.** The event loop, the waiting/running queues, how the overlap scheduler keeps the GPU busy while Python does bookkeeping.
5. **§5 — ModelRunner.** The biggest part. Loads weights from safetensors, wires up the model registry, builds the KV pool and its allocator, picks the attention backend, captures CUDA graphs. The **core** of the GPU story.
6. **§6 — LoRA subsystem.** Physical module swaps at load time, the memory pool layout (3-D dense / 4-D MoE / shared-outer variants), `LoRABatchInfo` with its two segmentation layouts, the Triton SGEMM kernels, and CUDA-graph integration. Long but self-contained; skip if not using LoRA.
7. **§7 — A request, end to end.** Synthesis. Walks the request lifecycle above through the actual SGLang call stack, with line-level references.
8. **§8 — TP & EP.** Deep dive on the two parallelism dimensions most relevant to Qwen3-MoE. Includes DeepEP's Normal and Low-Latency dispatchers.
9. **§9 — PP, CP, DP & routers.** Less relevant to our run but important for scale-out: pipeline parallel, context parallel, DP attention, and the external sgl-router for multi-replica serving.
10. **§10 — Where to change things.** Practical entry points for common modifications.
11. **§11 — Reference index.** Shape, module, and symbol table lookups.

<div class="callout info" markdown="1">

#### Who should read this doc front-to-back?

If you're onboarding to SGLang and plan to change *any* part of it: read §1–7 in order, skim §8–9, then use §10–11 as lookup. If you're only working on a specific area (kernels, scheduling, parallelism, LoRA), skip to that Part and follow its internal subsection order — each Part is self-contained enough to be read standalone once you've seen §2 (launch) and §4 (scheduler).

</div>

---

<p class="bridge" markdown="span">*We start with the model on disk — because every runtime concern in the rest of this doc traces back to a shape, a parameter, or a layer decision visible in Qwen3's own files.*</p>

## 1 · The Qwen3-30B-A3B-Instruct-2507 model
{: #model }

Before we can talk about SGLang, we need to know what's on disk. A Hugging Face "model" is a folder with a handful of JSON files and a sharded tensor archive. Here's the tree for this checkpoint, with file sizes as shown on the HF page:

<div class="code-head" markdown="span">
Qwen/Qwen3-30B-A3B-Instruct-2507 · tree/main (61.1 GB total) <a href="https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/tree/main" target="_blank" rel="noopener noreferrer">view on HF</a>
</div>

```text
config.json                          963 B
config_1m.json                       77.3 KB   (optional 1M-context config)
generation_config.json              239 B
merges.txt                           1.67 MB
tokenizer.json                       ...
tokenizer_config.json                ...
vocab.json                           ...
model.safetensors.index.json        <weight shard index>
model-00001-of-00016.safetensors    ~4.0 GB
model-00002-of-00016.safetensors    ~4.0 GB
...
model-00016-of-00016.safetensors    ~4.0 GB    (16 shards total)
```

Only three of those files matter for SGLang's model loader: `config.json` (architecture + hyperparameters), `model.safetensors.index.json` (which tensor lives in which shard), and the `*.safetensors` files (the actual weights). Everything else (tokenizer, chat template, generation config) is consumed by the <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py#L215" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TokenizerManager</code></a>, not the model loader.

### 1.1 `config.json` line by line
{: #model-config }

Here is the authoritative `config.json` for this checkpoint, fetched from HF (commit `e67ac5d`):

<div class="code-head" markdown="span">
<span class="badge badge-tr">TR</span> config.json (verbatim) <a href="https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/config.json" target="_blank" rel="noopener noreferrer">on HF</a>
</div>

```json
{
  "architectures": ["Qwen3MoeForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "decoder_sparse_step": 1,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 6144,
  "max_position_embeddings": 262144,
  "max_window_layers": 48,
  "mlp_only_layers": [],
  "model_type": "qwen3_moe",
  "moe_intermediate_size": 768,
  "norm_topk_prob": true,
  "num_attention_heads": 32,
  "num_experts": 128,
  "num_experts_per_tok": 8,
  "num_hidden_layers": 48,
  "num_key_value_heads": 4,
  "output_router_logits": false,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 10000000,
  "router_aux_loss_coef": 0.001,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
```

Here's what each field controls, and which SGLang code path reads it:

| Field                                  | Value                               | Meaning                                                                                                            | Read by                                                                                                                                                                                                                                                                                                                                                                                                                                |
|----------------------------------------|-------------------------------------|--------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `architectures`                        | `["Qwen3MoeForCausalLM"]`           | Tells SGLang which Python class to instantiate.                                                                    | <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/utils.py#L193" target="_blank" rel="noopener noreferrer"><code>get_model_architecture</code></a> → `ModelRegistry.resolve_model_cls`                                                                                                                                                                                                           |
| `model_type`                           | `"qwen3_moe"`                       | Key HF's `AutoConfig` uses to map to `Qwen3MoeConfig`.                                                             | HF `AutoConfig.from_pretrained`                                                                                                                                                                                                                                                                                                                                                                                                        |
| `hidden_size`                          | <span class="num">2048</span>       | Per-token residual-stream width. H in all shape math below.                                                        | <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_moe.py#L432" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Qwen3MoeAttention</code></a>, <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_moe.py#L233" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Qwen3MoeSparseMoeBlock</code></a>, every linear              |
| `num_hidden_layers`                    | <span class="num">48</span>         | Number of transformer decoder blocks. L below.                                                                     | <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_moe.py#L910" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Qwen3MoeModel</code></a>; `LoRAMemoryPool.num_layer`                                                                                                                                                                                                             |
| `num_attention_heads`                  | <span class="num">32</span>         | Total Q heads. Must be divisible by TP size.                                                                       | <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L866" class="sym-link" target="_blank" rel="noopener noreferrer"><code>QKVParallelLinear</code></a>                                                                                                                                                                                                                                        |
| `num_key_value_heads`                  | <span class="num">4</span>          | K/V heads (GQA — 4 KV heads serve 32 Q heads, ratio 8:1).                                                          | <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L866" class="sym-link" target="_blank" rel="noopener noreferrer"><code>QKVParallelLinear</code></a>, <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool.py#L742" class="sym-link" target="_blank" rel="noopener noreferrer"><code>MHATokenToKVPool</code></a>                                |
| `head_dim`                             | <span class="num">128</span>        | Per-head dim. **Note** hidden_size/32 = 64, so head_dim is *not* hidden_size/num_heads here — it's set explicitly. | <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/radix_attention.py#L47" class="sym-link" target="_blank" rel="noopener noreferrer"><code>RadixAttention</code></a>, RoPE                                                                                                                                                                                                                             |
| `intermediate_size`                    | <span class="num">6144</span>       | Dense FFN up-projection width. **Unused** for this model because `mlp_only_layers=[]`.                             | `Qwen3MoeMLP` (dense path, not taken)                                                                                                                                                                                                                                                                                                                                                                                                  |
| `moe_intermediate_size`                | <span class="num">768</span>        | Per-expert up-projection width.                                                                                    | `FusedMoE.intermediate_size_per_partition`                                                                                                                                                                                                                                                                                                                                                                                             |
| `num_experts`                          | <span class="num">128</span>        | Total experts per MoE block.                                                                                       | <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L138" class="sym-link" target="_blank" rel="noopener noreferrer"><code>FusedMoE</code></a>, <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_moe.py#L233" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Qwen3MoeSparseMoeBlock</code></a>                    |
| `num_experts_per_tok`                  | <span class="num">8</span>          | Top-k routing (8 of 128 ≈ 6.25% active per token).                                                                 | <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/topk.py#L239" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TopK</code></a> module                                                                                                                                                                                                                                            |
| `decoder_sparse_step`                  | <span class="num">1</span>          | Every layer is sparse (every step). Combined with `mlp_only_layers=[]` this means all 48 layers are MoE.           | `Qwen3MoeDecoderLayer.is_layer_sparse`                                                                                                                                                                                                                                                                                                                                                                                                 |
| `mlp_only_layers`                      | `[]`                                | Which layers to force dense. Empty ⇒ all MoE.                                                                      | <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_moe.py#L710" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Qwen3MoeDecoderLayer</code></a>                                                                                                                                                                                                                                  |
| `norm_topk_prob`                       | `true`                              | Renormalize the 8 selected router probs to sum to 1.                                                               | <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/topk.py#L239" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TopK</code></a>                                                                                                                                                                                                                                                   |
| `vocab_size`                           | <span class="num">151 936</span>    | Embedding / lm_head rows.                                                                                          | <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/vocab_parallel_embedding.py#L161" class="sym-link" target="_blank" rel="noopener noreferrer"><code>VocabParallelEmbedding</code></a>, <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/vocab_parallel_embedding.py#L512" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ParallelLMHead</code></a> |
| `max_position_embeddings`              | <span class="num">262 144</span>    | Native context length.                                                                                             | RoPE construction, `ModelConfig.context_len`                                                                                                                                                                                                                                                                                                                                                                                           |
| `rope_theta`                           | <span class="num">10 000 000</span> | RoPE base period (much larger than the classic 10 000 — extends usable context).                                   | `get_rope`                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `sliding_window`, `use_sliding_window` | `null`, `false`                     | Full attention everywhere — no SWA for this variant.                                                               | <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/radix_attention.py#L47" class="sym-link" target="_blank" rel="noopener noreferrer"><code>RadixAttention</code></a>                                                                                                                                                                                                                                   |
| `tie_word_embeddings`                  | `false`                             | lm_head is a separate tensor from embed_tokens.                                                                    | <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_moe.py#L944" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Qwen3MoeForCausalLM.__init__</code></a>                                                                                                                                                                                                                          |
| `attention_bias`                       | `false`                             | No bias on q/k/v/o linears.                                                                                        | <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L866" class="sym-link" target="_blank" rel="noopener noreferrer"><code>QKVParallelLinear</code></a>, <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L1312" class="sym-link" target="_blank" rel="noopener noreferrer"><code>RowParallelLinear</code></a>                                      |
| `rms_norm_eps`                         | <span class="num">1e-6</span>       | ε for RMSNorm (including per-head q_norm and k_norm).                                                              | <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/dots_vlm_vit.py#L75" class="sym-link" target="_blank" rel="noopener noreferrer"><code>RMSNorm</code></a>                                                                                                                                                                                                                                             |
| `torch_dtype`                          | `"bfloat16"`                        | Serialized weight dtype. SGLang inherits this unless `--dtype` overrides.                                          | `ModelConfig.dtype`                                                                                                                                                                                                                                                                                                                                                                                                                    |

<div class="callout motiv" markdown="1">

#### Why the name "A3B"?

Each token activates `num_experts_per_tok / num_experts = 8/128 = 6.25 %` of MoE parameters. The total parameter count is ~30.5 B; the active count (embeddings + attention + router + activated experts + norms) comes out to roughly 3 B. Qwen calls this **"A3B"** — *active 3B*. See the <a href="https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507" target="_blank" rel="noopener noreferrer">model card</a>: "30.5B in total and 3.3B activated."

</div>

<div class="callout motiv" markdown="1">

#### Why head_dim = 128 instead of hidden_size / num_heads = 64?

On the HF side this is an explicit config field on `Qwen3MoeConfig`, and SGLang reads `head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)` in the decoder layer (`models/qwen3_moe.py`). A larger head_dim improves effective attention rank without widening the residual stream — this is a common Qwen2/Qwen3 choice. Result: QKV's total output dim is 32·128 + 4·128 + 4·128 = 5120 (\> hidden_size of 2048), so <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L866" class="sym-link" target="_blank" rel="noopener noreferrer"><code>QKVParallelLinear</code></a> expands on the way in and <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L1312" class="sym-link" target="_blank" rel="noopener noreferrer"><code>RowParallelLinear</code></a> (o_proj) contracts back.

</div>

### 1.2 HF layer class hierarchy
{: #model-layout }

HF's Qwen3Moe model is defined in the *modular* style — a short file describes deltas from other models, and a generator expands it into a full `modeling_qwen3_moe.py`. Here's the source of truth (the delta file):

<div class="code-head" markdown="span">
<span class="badge badge-hf">HF</span> src/transformers/models/qwen3_moe/modular_qwen3_moe.py:43-94 <a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modular_qwen3_moe.py#L43" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
class Qwen3MoeAttention(Qwen3Attention):  # This is the main diff with qwen2Moe!
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.layer_type
        self.sliding_window = getattr(config, "sliding_window", None)

class Qwen3MoeMLP(Qwen2MoeMLP):
    pass

class Qwen3MoeExperts(Qwen2MoeExperts):
    pass

class Qwen3MoeTopKRouter(Qwen2MoeTopKRouter):
    pass

class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.experts = Qwen3MoeExperts(config)
        self.gate = Qwen3MoeTopKRouter(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

class Qwen3MoeRMSNorm(LlamaRMSNorm):
    pass

class Qwen3MoeDecoderLayer(Qwen2MoeDecoderLayer):
    pass

class Qwen3MoePreTrainedModel(MixtralPreTrainedModel):
    _can_record_outputs = {
        "router_logits": OutputRecorder(Qwen3MoeTopKRouter, index=0),
        "hidden_states": Qwen3MoeDecoderLayer,
        "attentions": Qwen3MoeAttention,
    }

class Qwen3MoeModel(MixtralModel):
    pass

class Qwen3MoeForCausalLM(MixtralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3MoeModel(config)
        self.num_experts = config.num_experts
```

The critical takeaway: **Qwen3Moe inherits from three different parent families**. The attention path is *Qwen3*-style (with per-head RMSNorm on Q and K, inherited from <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3.py#L60" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Qwen3Attention</code></a>); the MoE path is *Qwen2Moe*-style (shared top-k router, shared experts module); and the outer model is *Mixtral*-style (<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/mixtral_quant.py#L310" class="sym-link" target="_blank" rel="noopener noreferrer"><code>MixtralModel</code></a> / <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/mixtral.py#L336" class="sym-link" target="_blank" rel="noopener noreferrer"><code>MixtralForCausalLM</code></a>). The code comment "`This is the main diff with qwen2Moe!`" calls out that the per-head Q/K norm is the defining architectural change from Qwen2MoE.

Here's the key attention code from <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3.py#L60" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Qwen3Attention</code></a> showing the per-head norms:

<div class="code-head" markdown="span">
<span class="badge badge-hf">HF</span> src/transformers/models/qwen3/modeling_qwen3.py:222-248 <a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py#L222" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
```

For Qwen3-30B-A3B-Instruct-2507, with `hidden_size=2048`, `num_attention_heads=32`, `num_key_value_heads=4`, `head_dim=128`, and `attention_bias=false`, HF allocates:

| Name in HF state dict                                            | Shape                         | Elements  | Bytes (bf16) |
|------------------------------------------------------------------|-------------------------------|-----------|--------------|
| `model.layers.{i}.self_attn.q_proj.weight`                       | (32·128, 2048) = (4096, 2048) | 8 388 608 | 16 MB        |
| `model.layers.{i}.self_attn.k_proj.weight`                       | (4·128, 2048) = (512, 2048)   | 1 048 576 | 2 MB         |
| `model.layers.{i}.self_attn.v_proj.weight`                       | (4·128, 2048) = (512, 2048)   | 1 048 576 | 2 MB         |
| `model.layers.{i}.self_attn.o_proj.weight`                       | (2048, 4096)                  | 8 388 608 | 16 MB        |
| `model.layers.{i}.self_attn.q_norm.weight`                       | (128,)                        | 128       | 256 B        |
| `model.layers.{i}.self_attn.k_norm.weight`                       | (128,)                        | 128       | 256 B        |
| `model.layers.{i}.input_layernorm.weight`                        | (2048,)                       | 2048      | 4 KB         |
| `model.layers.{i}.post_attention_layernorm.weight`               | (2048,)                       | 2048      | 4 KB         |
| `model.layers.{i}.mlp.gate.weight`                               | (128, 2048)                   | 262 144   | 512 KB       |
| `model.layers.{i}.mlp.experts.{j}.gate_proj.weight` (j = 0..127) | (768, 2048)                   | 1 572 864 | 3 MB         |
| `model.layers.{i}.mlp.experts.{j}.up_proj.weight`                | (768, 2048)                   | 1 572 864 | 3 MB         |
| `model.layers.{i}.mlp.experts.{j}.down_proj.weight`              | (2048, 768)                   | 1 572 864 | 3 MB         |

Total per layer (HF-side) = 16 + 2 + 2 + 16 + 256 B + 256 B + 4 KB + 4 KB + 512 KB + 128·(3+3+3) MB = **~1.19 GB per layer**. Over 48 layers: **~57 GB**. Plus embeddings: 2 × 151 936 × 2048 × 2 B = 1.24 GB. Plus `model.norm.weight`. Grand total matches the HF-reported **~61.1 GB**.

Notice the MoE structure on disk: each expert is stored as **three separate tensors per layer** (gate_proj, up_proj, down_proj), and each **expert is a separate key**. So there are `48 · 128 · 3 = 18 432` expert tensors in total, plus attention weights and norms. This matters for how SGLang loads the model: it will find these one-by-one as it iterates through the safetensors shards and has to stack them.

### 1.3 Tensor names that matter
{: #model-shapes }

When SGLang iterates the checkpoint's safetensors it sees a stream of `(name, tensor)` pairs. Here's a representative sample of what that stream looks like for this model:

```text
model.embed_tokens.weight                                        (151936, 2048)
model.layers.0.input_layernorm.weight                            (2048,)
model.layers.0.self_attn.q_proj.weight                           (4096, 2048)
model.layers.0.self_attn.k_proj.weight                           (512, 2048)
model.layers.0.self_attn.v_proj.weight                           (512, 2048)
model.layers.0.self_attn.o_proj.weight                           (2048, 4096)
model.layers.0.self_attn.q_norm.weight                           (128,)
model.layers.0.self_attn.k_norm.weight                           (128,)
model.layers.0.post_attention_layernorm.weight                   (2048,)
model.layers.0.mlp.gate.weight                                   (128, 2048)
model.layers.0.mlp.experts.0.gate_proj.weight                    (768, 2048)
model.layers.0.mlp.experts.0.up_proj.weight                      (768, 2048)
model.layers.0.mlp.experts.0.down_proj.weight                    (2048, 768)
...
model.layers.0.mlp.experts.127.gate_proj.weight                  (768, 2048)
model.layers.0.mlp.experts.127.up_proj.weight                    (768, 2048)
model.layers.0.mlp.experts.127.down_proj.weight                  (2048, 768)
model.layers.1.input_layernorm.weight                            (2048,)
... (repeats for 48 layers) ...
model.norm.weight                                                (2048,)
lm_head.weight                                                   (151936, 2048)
```

Every one of those names will be rewritten on the fly as SGLang loads it: `q_proj`/`k_proj`/`v_proj` → `qkv_proj` (one fused tensor per layer); the 128 expert tensors per layer get stacked into two tensors (`w13_weight` for gate+up, `w2_weight` for down). Sections [5.6](#runner-qkv) and [5.7](#runner-moe) show exactly how.

### 1.4 The safetensors files on disk
{: #model-safetensors }

Safetensors is the format HF models ship in. A safetensors file is a JSON header (byte-offsets of each tensor) followed by the raw tensor bytes. Crucially, it supports **mmap**, so the OS can fault pages in lazily as the loader reads each tensor — the whole 61 GB is never in RAM at once.

When a model's weights don't fit in a single file, HF splits them into shards with an index file:

```json
{
  "metadata": { "total_size": 61XXXXXXXXX },
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00016.safetensors",
    "model.layers.0.input_layernorm.weight": "model-00001-of-00016.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00016.safetensors",
    ...
    "model.layers.47.mlp.experts.127.down_proj.weight": "model-00016-of-00016.safetensors",
    "model.norm.weight": "model-00016-of-00016.safetensors",
    "lm_head.weight": "model-00016-of-00016.safetensors"
  }
}
```

SGLang's loader uses this index to (1) know which shard files to download/access and (2) deduplicate when both a consolidated and a sharded copy are present. You'll see the exact code in [§5.4](#runner-loader).

<div class="callout info" markdown="1">

#### Why mmap + iterate rather than load everything?

For a 61 GB bf16 model, loading into userspace RAM would blow up swap on most servers. With mmap, SGLang reads one tensor at a time, copies it straight onto the GPU via a weight_loader hook (so quantized models can requantize on the fly and TP ranks can narrow out just their shard), and the OS evicts the mmap'd pages. The `safetensors_weights_iterator` (§5.4) is a generator, so at any moment only one tensor is in motion.

</div>

### 1.5 MoE step by step — what the router and the experts actually do
{: #model-moe-stepbystep }

This section is a self-contained tour of how a single token flows through one Qwen3-30B-A3B MoE block. Reading §1.3 you saw two families of tensors per layer that both contained the word "gate":

| Tensor                                              | Shape         | How many per layer       |
|-----------------------------------------------------|---------------|--------------------------|
| `model.layers.{i}.mlp.gate.weight`                  | `(128, 2048)` | **one**                  |
| `model.layers.{i}.mlp.experts.{j}.gate_proj.weight` | `(768, 2048)` | **128** (one per expert) |

They are completely different things. The naming collision is unfortunate but locked in by the Qwen / HF convention. Here's the disambiguation:

- **`mlp.gate`** is the **router**. Its job is to pick *which* 8 of the 128 experts should be used for a given token, and with what weights to blend their outputs. It outputs logits over experts. There is exactly **one** of these per MoE block — sometimes called the "gating network" in papers. (Shape `(num_experts=128, hidden_size=2048)`.)
- **`mlp.experts.{j}.gate_proj`** is the **first matrix of expert j's gated-SwiGLU MLP**. This one is named "gate" because in SwiGLU, the MLP's first step is `gate_part = gate_proj(x)`, where "gate" refers to the SiLU-gated branch of the activation function — nothing to do with routing. There are **128 of these per layer**, one per expert. (Shape per expert: `(moe_intermediate_size=768, hidden_size=2048)`.)

To make it concrete, here's what happens when one token (hidden state vector `x` of shape `(2048,)`) enters a Qwen3 MoE block.

#### Step 1 — Router (the `gate`): pick top-8 experts

The router is just a single `nn.Linear(hidden_size, num_experts, bias=False)`:

```python
# one forward through the router
router_logits = x @ gate.weight.T     # (2048,) @ (2048, 128) -> (128,)
                                      # one logit per expert

# top-k = 8:  indices of the 8 highest-scoring experts (e.g. [ 17, 42, 3, 91, 5, 108, 66, 12 ])
topk_logits, topk_indices = router_logits.topk(k=8)

# turn those 8 logits into a 8-way probability distribution:
topk_probs = softmax(topk_logits)     # (8,)

# Qwen3 sets norm_topk_prob=true, so we renormalize.
# In Qwen3's case (softmax over just the top-k logits), topk_probs already sums to 1 by
# construction, so this division is a ~1e-7 floating-point cleanup — effectively a no-op.
# The flag exists to share code with other MoE variants (sigmoid routing, noisy gating,
# grouped top-k) where the pre-renorm weights genuinely don't sum to 1. See the callout below.
topk_probs = topk_probs / topk_probs.sum()
```

After this step you have:

- `topk_indices` — 8 integers in `[0, 127]` saying "these are the 8 experts this token will be routed to".
- `topk_probs` — 8 floats in `[0, 1]` summing to 1, saying "weight expert `topk_indices[k]`'s output by `topk_probs[k]` when blending."

The other 120 experts contribute **nothing** to this token. Their weights are never read from GPU memory during this forward. That's where the compute saving comes from: 128 potential experts, only 8 activated.

<div class="callout motiv" markdown="1">

#### Why softmax *after* top-k, not before?

There are two mathematically different ways to produce top-k expert weights. Call them:

- **Variant A — softmax-then-topk.** Softmax over all 128 logits, then keep the 8 highest. The 8 kept values were 8 entries from a 128-way distribution, so they do *not* sum to 1 and are individually tiny.
- **Variant B — topk-then-softmax.** Take the 8 highest logits, then softmax over just those 8. The 8 kept values sum to 1 by construction.

Qwen3 (and Mixtral, and most modern MoE models) use Variant B. It has two advantages: you only compute 8 exponentials instead of 128, and each activated expert gets a meaningful contribution (typically 5–25%) rather than a value that sums to 1/128 on average. See the <a href="https://arxiv.org/abs/2401.04088" target="_blank" rel="noopener noreferrer">Mixtral paper</a>'s Eq. 1 for the exact form.

</div>

<div class="callout info" markdown="1">

#### What is `norm_topk_prob` actually for, if Qwen3's top-k weights already sum to 1?

The `norm_topk_prob` flag toggles a final `topk_probs /= topk_probs.sum()` renormalize step. Under Variant B above, that sum is already 1.0 ± ε (where ε ≈ 1e-7 from floating-point arithmetic), so for Qwen3 the renormalize is effectively a no-op. **The flag exists because the same HF / SGLang MoE code path serves several other routing variants where it's load-bearing:**

- **Sigmoid routing** (DeepSeek-V3, some Qwen variants): each expert gets an independent sigmoid score in (0, 1) rather than a softmax-over-all-experts. The 8 kept sigmoid values sum to something arbitrary between 0 and 8. Renormalize is mandatory.
- **Noisy top-k gating** (Switch Transformer / Shazeer style): add Gaussian noise to logits before top-k; apply softmax over all experts; keep only the top-k entries. The kept entries are Variant A's outputs — they don't sum to 1.
- **Grouped top-k / group-limited routing** (DeepSeek-V3): experts are partitioned into groups; top-k within groups, then top-k of those. The two-stage selection doesn't preserve softmax's sum-to-1 property.

Qwen3 doesn't use any of those paths, so for your audit purposes the renormalize is a defensive no-op. It stays in the code for (a) cross-checkpoint compatibility, (b) bit-exact reproducibility of outputs trained with the flag set, and (c) robustness against future code changes that might transform `topk_probs` between softmax and use. Finally, the probs matter because the very next step multiplies each expert's output by the corresponding prob and sums: `final_y = sum(topk_probs[k] * expert_outputs[k] for k in range(8))`. If the probs don't sum to 1, the magnitude of `final_y` drifts and the downstream residual + LayerNorm receives an out-of-distribution input.

</div>

#### Step 2 — Expert MLP: SwiGLU, one per expert

Each selected expert `j ∈ topk_indices` runs the token through its own three-matrix SwiGLU MLP. The three matrices per expert are:

| Checkpoint name                | Shape         | Role                                                                                  |
|--------------------------------|---------------|---------------------------------------------------------------------------------------|
| `experts.{j}.gate_proj.weight` | `(768, 2048)` | Project `x` into a 768-d "gate" activation (the one that gets SiLU'd).                |
| `experts.{j}.up_proj.weight`   | `(768, 2048)` | Project `x` into a parallel 768-d "up" activation (the one multiplied into the gate). |
| `experts.{j}.down_proj.weight` | `(2048, 768)` | Project the 768-d hidden activation back down to 2048.                                |

```python
# Running through expert j:
gate_part = x @ experts[j].gate_proj.weight.T    # (2048,) @ (2048, 768) -> (768,)
up_part   = x @ experts[j].up_proj.weight.T      # (2048,) @ (2048, 768) -> (768,)

# SwiGLU: silu = x * sigmoid(x)
hidden = silu(gate_part) * up_part               # (768,)  element-wise product

# Down-project back to hidden size:
out_j  = hidden @ experts[j].down_proj.weight.T  # (768,) @ (768, 2048) -> (2048,)
```

This happens for each of the 8 activated experts, in parallel on the GPU — in practice fused into a single kernel launch via <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L138" class="sym-link" target="_blank" rel="noopener noreferrer"><code>FusedMoE</code></a>'s grouped GEMM (§5.7), not 8 separate calls. But mathematically the above is exactly what each expert does.

<div class="callout info" markdown="1">

#### Why three matrices, not one big one?

A "classic" MLP is `down_proj(activation(up_proj(x)))` — two matrices. SwiGLU adds a third matrix to *gate* the activation: `down_proj( silu(gate_proj(x)) * up_proj(x) )`. This lets the network learn element-wise "which features of up_proj(x) matter" via a data-dependent mask. <a href="https://arxiv.org/abs/2002.05202" target="_blank" rel="noopener noreferrer">Shazeer's 2020 paper</a> showed this gives a meaningful quality improvement over vanilla GeLU MLPs at fixed parameter count, and basically every LLM since Llama-2 uses it. Qwen3 inherits the convention.

</div>

#### Step 3 — Combine the 8 expert outputs into one

```python
# topk_indices = [17, 42, 3, 91, 5, 108, 66, 12]
# topk_probs   = [0.21, 0.18, 0.15, 0.13, 0.12, 0.10, 0.07, 0.04]
final_y = sum(topk_probs[k] * out_{topk_indices[k]} for k in range(8))  # (2048,)
```

`final_y` is the output of the MoE block for this token. It then goes into the residual connection and off to the next decoder layer. That's the whole MoE block.

#### Putting it together — one MoE block, one token

End to end for a single token with `x.shape = (2048,)`:

```python
# === ROUTER (1 tensor: gate.weight of shape (128, 2048)) ===
router_logits = x @ gate.weight.T                            # (128,)
topk_logits, topk_indices = router_logits.topk(k=8)          # (8,), (8,)
topk_probs = softmax(topk_logits)
topk_probs = topk_probs / topk_probs.sum()                   # norm_topk_prob=True

# === EXPERTS (3 * 128 tensors: gate_proj, up_proj, down_proj for each of 128 experts) ===
expert_outputs = []
for j in topk_indices:                    # 8 chosen experts out of 128
    g = x @ experts[j].gate_proj.weight.T                    # (768,)
    u = x @ experts[j].up_proj.weight.T                      # (768,)
    h = silu(g) * u                                          # (768,)
    o = h @ experts[j].down_proj.weight.T                    # (2048,)
    expert_outputs.append(o)

# === COMBINE ===
final_y = sum(topk_probs[k] * expert_outputs[k] for k in range(8))  # (2048,)
```

#### Per-layer FLOP and parameter accounting

Let's count exactly what this costs per token for Qwen3-30B-A3B-Instruct-2507.

| Operation                          | FLOPs (per token)                 | Params touched             | Params defined                |
|------------------------------------|-----------------------------------|----------------------------|-------------------------------|
| Router: `x @ gate.weight.T`        | 2 · 2048 · 128 = 524 288          | 262 144                    | 262 144                       |
| 8 experts × gate_proj              | 8 · (2 · 2048 · 768) = 25 165 824 | 8 · 1 572 864 = 12 582 912 | 128 · 1 572 864 = 201 326 592 |
| 8 experts × up_proj                | 25 165 824                        | 12 582 912                 | 201 326 592                   |
| 8 experts × SiLU + mult            | 8 · 2 · 768 = 12 288              | 0                          | 0                             |
| 8 experts × down_proj              | 8 · (2 · 768 · 2048) = 25 165 824 | 12 582 912                 | 201 326 592                   |
| Combine (topk-weighted sum)        | 8 · 2048 = 16 384                 | 0                          | 0                             |
| **Total per MoE block, per token** | **~76 M FLOPs**                   | **~38 M params**           | **~604 M params**             |

Multiply by **48 layers** and add attention + embeddings:

- **Active params (touched per token):** ~48 · 38 M (MoE) + 48 · 5 M (attention) + 0.3 B (embeddings) ≈ **~3.3 B**. That's the **"A3B"** in the model name — "active 3B".
- **Total params (defined):** ~48 · 604 M (MoE) + 48 · 27 M (attention) + 0.6 B (embed + lm_head) ≈ **~30.5 B**. That's the "30B" in the name.

The ratio 3.3 / 30.5 ≈ 10.8% sits slightly above the naïve `k/E = 8/128 = 6.25 %` because attention and embeddings are dense (every token touches every attention weight), and the router itself is also dense. But the dominant parameters — the 128 experts — are sparsely activated.

#### Scaling to a real batch

Everything above was for one token. In a real batch with *T* tokens (across all requests), the per-layer picture changes like this:

1. **Router is dense in tokens**: you compute `T × 128` logits, one top-k + softmax per token. Cost: `O(T · H · E)`.
2. **Routing histogram**: once you know each token's 8 chosen experts, you can count "how many tokens picked expert j" for every j. Call this `tokens_per_expert[j]`; total is `8T`.
3. **Experts execute in parallel**: expert j runs one gate/up/down triple on exactly `tokens_per_expert[j]` tokens. Some experts may see zero tokens in a given batch — their weights aren't read from GPU memory at all this step (the base-MoE kernel skips empty groups).
4. **Per-expert GEMMs are grouped**: rather than 128 separate kernel launches of variable size, SGLang's <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L138" class="sym-link" target="_blank" rel="noopener noreferrer"><code>FusedMoE</code></a> (§5.7) runs a single Triton *grouped GEMM* kernel that processes all 128 expert-shards in one launch.

This is why the MoE block is faster than a dense MLP of equivalent capacity at inference: given a batch of *T* tokens, only `8T / 128 = 6.25 %` of the expert weights participate in the FLOPs (ignoring the small dense router cost).

<div class="callout info" markdown="1">

#### Why we need *both* `gate.weight` (router) and `experts.{j}.gate_proj.weight`, in one sentence

`gate.weight` says *which* experts get this token, and `experts.{j}.gate_proj.weight` is *inside* expert j's own SwiGLU MLP — the network that actually processes the token once it's been routed there. The name "gate" is reused because both are "gating" things, but at entirely different levels of the architecture.

</div>

<div class="callout motiv" markdown="1">

#### How routing is *trained* (and why it stays sensible at inference)

During training, an auxiliary "load balancing" loss encourages the router to distribute tokens evenly across experts. Without it the router collapses: all tokens go to the best expert early, the others never learn, and you end up with a very expensive 1-expert model. Qwen3 uses the coefficient `router_aux_loss_coef=0.001` (in `config.json`). At inference the aux loss is no longer computed (`output_router_logits=false`), but its effect is baked into the trained router weights. If inference-time routing becomes too imbalanced — one expert getting many more tokens than others — SGLang's EPLB (Expert Parallel Load Balancer) can detect it and rebalance. See the <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/eplb/eplb_manager.py#L16" class="sym-link" target="_blank" rel="noopener noreferrer"><code>EPLBManager</code></a> call in §5.1.

</div>

---

<p class="bridge" markdown="span">*With the model's shape and weights understood, the next question is how SGLang actually loads it and wires up the serving infrastructure around it — which means following the launch command end-to-end.*</p>

## 2 · Launching `sglang.launch_server`
{: #launch }

We run `python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 --tp 4 ...`. Python imports `sglang/launch_server.py` as a module. Here's the *entire* file — it's tiny.

### 2.1 The entrypoint: `launch_server.py`
{: #launch-entry }

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> python/sglang/launch_server.py (71 lines total) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/launch_server.py" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
"""Launch the inference server."""

import asyncio
import os
import sys
import warnings

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()

def run_server(server_args):
    """Run the server based on server_args.grpc_mode and server_args.encoder_only."""
    if server_args.encoder_only:
        # For encoder disaggregation
        if server_args.grpc_mode:
            from sglang.srt.disaggregation.encode_grpc_server import serve_grpc_encoder
            asyncio.run(serve_grpc_encoder(server_args))
        else:
            from sglang.srt.disaggregation.encode_server import launch_server
            launch_server(server_args)
    elif server_args.grpc_mode:
        # TODO: Once the native Rust gRPC server starts alongside HTTP in the
        # default path below (controlled by SGLANG_ENABLE_GRPC / SGLANG_GRPC_PORT),
        # remove this legacy SMG path and the grpc_mode flag.
        from sglang.srt.entrypoints.grpc_server import serve_grpc
        asyncio.run(serve_grpc(server_args))
    elif server_args.use_ray:
        try:
            from sglang.srt.ray.http_server import launch_server
        except ImportError:
            raise ImportError(
                "Ray is required for --use-ray mode. "
                "Install it with: pip install 'sglang[ray]'"
            )
        launch_server(server_args)
    else:
        # Default mode: HTTP mode.
        from sglang.srt.entrypoints.http_server import launch_server
        launch_server(server_args)

if __name__ == "__main__":
    warnings.warn(
        "'python -m sglang.launch_server' is still supported, but "
        "'sglang serve' is the recommended entrypoint.\n"
        "  Example: sglang serve --model-path <model> [options]",
        UserWarning,
        stacklevel=1,
    )

    from sglang.srt.plugins import load_plugins
    load_plugins()

    server_args = prepare_server_args(sys.argv[1:])

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
```

The control flow is:

1. `suppress_noisy_warnings()` silences noisy third-party warnings (tokenizer parallelism, transformer-engine FP8 messages, etc.).
2. A deprecation warning prints. Since SGLang v0.5-ish, the blessed CLI is `sglang serve`, which lives in the `sglang` console script. <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/launch_server.py" class="sym-link" target="_blank" rel="noopener noreferrer"><code>launch_server</code></a> is kept for backward compatibility.
3. `load_plugins()` imports any plugin packages the user registered via `SGLANG_PLUGIN_PACKAGES` env. This gives third-party shims a chance to register new models, new attention backends, etc., **before** any argument is parsed.
4. `prepare_server_args(sys.argv[1:])` → returns a <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L285" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ServerArgs</code></a> dataclass (see §2.2).
5. `run_server(server_args)` dispatches based on four flags.
6. On exit, `kill_process_tree` makes sure the scheduler and detokenizer subprocesses die with us — otherwise a Ctrl-C would orphan them.

For a standard run (what we're tracing), only the last branch matters: `from sglang.srt.entrypoints.http_server import launch_server; launch_server(server_args)`.

### 2.2 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L6943" class="sym-link" target="_blank" rel="noopener noreferrer"><code>prepare_server_args</code></a> and <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L748" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ServerArgs.__post_init__</code></a>
{: #launch-args }

<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L285" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ServerArgs</code></a> is a large dataclass (~7183 lines of server_args.py). <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L6943" class="sym-link" target="_blank" rel="noopener noreferrer"><code>prepare_server_args</code></a> builds an argparse parser from its fields, parses CLI, applies a YAML `--config` file if present, and calls the dataclass constructor. That constructor's `__post_init__` does extensive cross-field validation and back-fills defaults.

`__post_init__` runs a sequence of ~40 private `_handle_*` methods. Each deals with one slice of the config — device backends, attention backend compatibility, MoE kernel choice, pipeline parallelism, etc. Here's the outline:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> python/sglang/srt/server_args.py:748-870 (truncated) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L748" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def __post_init__(self):
    """Orchestrates the handling of various server arguments..."""

    self._maybe_download_model_for_runai()
    self._handle_load_balance_method()
    self._handle_multimodal()
    self._handle_ssl_validation()

    if self.model_path.lower() in ["none", "dummy"]:
        return  # dummy models skip the rest

    self._handle_deprecated_args()
    self._handle_prefill_delayer_env_compat()

    # quantization resolution
    if self.quantization == "unquant":
        self.quantization = None
        self._quantization_explicitly_unset = True
    else:
        self._quantization_explicitly_unset = False

    self._handle_missing_default_values()

    # Device-specific backends
    self._handle_hpu_backends();    self._handle_cpu_backends()
    self._handle_npu_backends();    self._handle_mps_backends();  self._handle_xpu_backends()
    current_platform.apply_server_args_defaults(self)

    self._handle_piecewise_cuda_graph()
    self._handle_multi_item_scoring()

    gpu_mem = get_device_memory_capacity(self.device)
    self._handle_gpu_memory_settings(gpu_mem)

    self._handle_model_specific_adjustments()

    # Kernel backend pickers
    self._handle_sampling_backend()
    self._handle_attention_backend_compatibility()
    self._handle_mamba_backend(); self._handle_linear_attn_backend()
    self._handle_kv4_compatibility(); self._handle_page_size()
    self._handle_amd_specifics(); self._handle_nccl_pre_warm()
    self._handle_grammar_backend()

    # Caches
    self._handle_hicache()

    # Parallelism
    self._handle_data_parallelism(); self._handle_context_parallelism()
    self._handle_moe_kernel_config(); self._handle_a2a_moe()
    self._handle_eplb_and_dispatch(); self._handle_expert_distribution_metrics()
    self._handle_elastic_ep(); self._handle_pipeline_parallelism()

    # Exotic modes
    self._handle_speculative_decoding()
    self._handle_load_format()
    self._handle_pd_disaggregation(); self._handle_encoder_disaggregation()

    self._handle_tokenizer_batching()
    self._handle_environment_variables()
    self._handle_cache_compatibility()
    self._handle_deterministic_inference()
    self._handle_dllm_inference()
    self._handle_debug_utils()
    self._handle_other_validations()
```

**Important:** there is no `_handle_lora_settings()` in `__post_init__`. LoRA validation is deferred to `check_server_args()`/`check_lora_server_args()`, which run later in <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/engine.py#L633" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Engine._launch_subprocesses</code></a>. We cover those in §2.3.

Two examples of what a `_handle_*` method actually does:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> Example: _handle_attention_backend_compatibility (partial, server_args.py:2406+) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L2406" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
# From the comment block at the top of the method:
#     1.1 We will turn on FA3 on hopper unless user use spec decode with topk > 1 or page_size > 1.
#     1.2 Use trtllm_mha for SM100/SM103 (Blackwell B200/GB200/B300) excluding spec with topk > 1.
#         Note: trtllm_mha does not support SM120, which will fall back to flashinfer.
#     1.3 In other cases, we will use flashinfer if available, otherwise use triton.
#     2. Models with MLA Architecture and using FA3
#         2.1 We will use FA3 backend on hopper.
#         2.2 We will use Flashinfer backend on blackwell.
#         2.3 Otherwise, we will use triton backend.
#     ...
    f"Attention backend not specified. Use {self.attention_backend} backend by default."
```

For an H100/H200 run of Qwen3-30B (non-MLA, bf16), this picker will set `attention_backend="fa3"`. The backend choice then drives page_size, KV layout, kernel warmup, and cuda graph shape — which is why this method is early and important.

### 2.3 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L6467" class="sym-link" target="_blank" rel="noopener noreferrer"><code>check_server_args</code></a> and <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L6659" class="sym-link" target="_blank" rel="noopener noreferrer"><code>check_lora_server_args</code></a>
{: #launch-check }

After `__post_init__` has normalized everything, a second validator runs *once the engine is about to launch subprocesses*. This is where LoRA settings land.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> server_args.py:6467 — check_server_args <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L6467" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def check_server_args(self):
    # Check parallel size constraints
    assert (self.tp_size * self.pp_size) % self.nnodes == 0, \
        "tp_size must be divisible by number of nodes"

    if self.pp_size > 1:
        assert (self.disable_overlap_schedule and self.speculative_algorithm is None), \
            "Pipeline parallelism is not compatible with overlap schedule, speculative decoding"

    assert not (self.dp_size > 1 and self.nnodes != 1 and not self.enable_dp_attention), \
        "multi-node data parallel is not supported unless dp attention!"

    assert self.base_gpu_id >= 0, "base_gpu_id must be non-negative"
    assert self.gpu_id_step >= 1, "gpu_id_step must be positive"

    assert self.moe_dense_tp_size in {1, None}, \
        "moe_dense_tp_size only support 1 and None currently"

    # served_model_name cannot contain ':' (reserved for base:adapter LoRA syntax)
    if not is_runai_obj_uri(self.served_model_name):
        assert ":" not in self.served_model_name, (
            "served_model_name cannot contain a colon (':') character. ..."
        )

    # Check LoRA
    self.check_lora_server_args()

    # Check speculative decoding
    if self.speculative_algorithm is not None:
        assert not self.enable_mixed_chunk, \
            "enable_mixed_chunk is required for speculative decoding"

    # Check chunked prefill
    if self.chunked_prefill_size > 0 and self.disaggregation_mode != "decode":
        assert self.chunked_prefill_size % self.page_size == 0, \
            "chunked_prefill_size must be divisible by page_size"
    # ... pdmux check omitted ...
```

And the actual LoRA validator it calls:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> server_args.py:6659 — check_lora_server_args (key parts) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L6659" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def check_lora_server_args(self):
    assert self.max_loras_per_batch > 0, "max_loras_per_batch must be positive"

    # Enable LoRA if any LoRA paths are provided for backward compatibility.
    if self.lora_paths:
        if self.enable_lora is None:
            self.enable_lora = True
            logger.warning("--enable-lora is set to True because --lora-paths is provided.")
        elif self.enable_lora is False:
            logger.warning("--enable-lora is set to False, any provided lora_paths will be ignored.")

    if self.enable_lora:
        if self.enable_lora_overlap_loading is None:
            self.enable_lora_overlap_loading = False

        if self.enable_lora_overlap_loading:
            max_loaded_loras_limit = self.max_loras_per_batch * 2
            assert self.max_loaded_loras is not None and self.max_loaded_loras <= max_loaded_loras_limit, \
                "Enabling LoRA overlap loading requires pinning LoRA adapter weights in CPU memory, ..."

        # Validate compatibility with speculative decoding
        if self.speculative_algorithm not in ["NGRAM", None]:
            raise ValueError("Currently LoRA is only compatible with NGRAM speculative decoding.")

        # Parse lora_paths into List[LoRARef]
        if isinstance(self.lora_paths, list):
            lora_paths = self.lora_paths
            self.lora_paths = []
            for lora_path in lora_paths:
                if isinstance(lora_path, str):
                    if "=" in lora_path:
                        name, path = lora_path.split("=", 1)
                        lora_ref = LoRARef(lora_name=name, lora_path=path, pinned=False)
                    else:
                        lora_ref = LoRARef(lora_name=lora_path, lora_path=lora_path, pinned=False)
                elif isinstance(lora_path, dict):
                    lora_ref = LoRARef(
                        lora_name=lora_path["lora_name"],
                        lora_path=lora_path["lora_path"],
                        pinned=lora_path.get("pinned", False),
                    )
                self.lora_paths.append(lora_ref)
        elif isinstance(self.lora_paths, dict):
            self.lora_paths = [LoRARef(lora_name=k, lora_path=v, pinned=False)
                               for k, v in self.lora_paths.items()]
        elif self.lora_paths is None:
            self.lora_paths = []

        # Normalize target modules
        if self.lora_target_modules:
            self.lora_target_modules = set(self.lora_target_modules)
            if "all" in self.lora_target_modules:
                assert len(self.lora_target_modules) == 1

        # Must have either lora_paths or both max_lora_rank + lora_target_modules
        assert self.lora_paths or (self.max_lora_rank and self.lora_target_modules), \
            "When no initial --lora-paths is provided, you need to specify both " \
            "--max-lora-rank and --lora-target-modules for LoRA initialization."

        # Validate max_loaded_loras
        if self.max_loaded_loras is not None:
            assert self.max_loaded_loras >= self.max_loras_per_batch
            assert len(self.lora_paths) <= self.max_loaded_loras

        # max_lora_chunk_size power-of-2 between 16..128
        if self.max_lora_chunk_size is not None:
            assert 16 <= self.max_lora_chunk_size <= 128 \
                and (self.max_lora_chunk_size & (self.max_lora_chunk_size - 1)) == 0, \
                "--max-lora-chunk-size must be a power of 2 between 16 and 128."
```

<div class="callout err" markdown="1">

#### Earlier drafts of this doc had a bug here

An earlier version of this doc claimed <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L6659" class="sym-link" target="_blank" rel="noopener noreferrer"><code>check_lora_server_args</code></a> forced `disable_radix_cache=True` when LoRA was enabled. **That is wrong.** Grep-verified on commit `1ebe1c57`: the only `self.disable_radix_cache = True` assignments in `server_args.py` are at lines 2273, 2325, 2450, 2594, 3561, 3832, and none of them are LoRA-triggered. Historically that *was* required — see <a href="https://github.com/sgl-project/sglang/discussions/2141" target="_blank" rel="noopener noreferrer">Discussion #2141</a> for the old assert — but <a href="https://github.com/sgl-project/sglang/pull/7216" target="_blank" rel="noopener noreferrer">PR #7216</a> (merged August 2025, by <a href="https://github.com/Fridge003" target="_blank" rel="noopener noreferrer">@Fridge003</a>) made radix cache compatible with LoRA by keying the tree on `(token_ids, lora_id)`. See §6.10.

</div>

### 2.4 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/common.py#L2492" class="sym-link" target="_blank" rel="noopener noreferrer"><code>run_server</code></a> branching
{: #launch-run-server }

With validated args, <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/common.py#L2492" class="sym-link" target="_blank" rel="noopener noreferrer"><code>run_server</code></a> picks one of four paths:

| Branch              | When                         | Entry point                                            |
|---------------------|------------------------------|--------------------------------------------------------|
| Encoder-only + gRPC | `--encoder-only --grpc-mode` | `disaggregation.encode_grpc_server.serve_grpc_encoder` |
| Encoder-only + HTTP | `--encoder-only`             | `disaggregation.encode_server.launch_server`           |
| Legacy gRPC mode    | `--grpc-mode`                | `entrypoints.grpc_server.serve_grpc`                   |
| Ray-coordinated     | `--use-ray`                  | `ray.http_server.launch_server`                        |
| **Default HTTP**    | (no flags)                   | **`entrypoints.http_server.launch_server`**            |

We'll follow the default HTTP path for the rest of this document.

<div class="callout info" markdown="1">

#### What's actually different about Ray-coordinated mode?

`ray/http_server.py` (69 lines) is a drop-in replacement for the default launcher — same FastAPI, same TokenizerManager, same scheduler code, same ZMQ wiring. The only difference is *how scheduler processes get spawned*. Default mode calls `mp.Process(target=run_scheduler_process, ...).start()` from <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/engine.py#L633" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Engine._launch_subprocesses</code></a>. Ray mode uses `RayEngine._launch_subprocesses` (`ray/engine.py:92`), which creates a Ray placement group, finds the bundle co-located with the Engine (`_find_engine_bundle` — rank 0 scheduler **must** live on the same node as the Engine so ZMQ IPC sockets work), and instantiates one <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/ray/scheduler_actor.py#L31" class="sym-link" target="_blank" rel="noopener noreferrer"><code>SchedulerActor</code></a> per rank as a Ray actor via `SchedulerActor.options(placement_group=pg, bundle_idx=...).remote(...)`. The actor's constructor (`ray/scheduler_actor.py:38`) then calls into the same <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L317" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Scheduler</code></a> constructor and `run_event_loop()` as default mode.

**Why use it?** (1) Multi-node orchestration without torchrun or SSH — Ray's placement group handles cross-node process placement. (2) Unified GPU resource management when running in a shared Ray cluster (e.g. KubeRay). (3) Integration with larger Ray pipelines (Ray Serve, RLHF rollout pipelines). For single-node runs you get nothing from it — default mode is simpler and has fewer dependencies.

One subtle implementation note: Ray actors can't be wrapped with `numactl` (the default NUMA-bind path uses subprocess wrapping), so <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/ray/scheduler_actor.py#L38" class="sym-link" target="_blank" rel="noopener noreferrer"><code>SchedulerActor.__init__</code></a> at `ray/scheduler_actor.py:95` explicitly binds to the right NUMA node in-process via libnuma. And the actor reads its actual GPU ID from `ray.get_runtime_context().get_accelerator_ids()` rather than trusting the passed `gpu_id`, since Ray may remap it for placement reasons.

</div>

### 2.5 `http_server.launch_server`
{: #launch-http }

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> python/sglang/srt/entrypoints/http_server.py:2313 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py#L2313" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def launch_server(
    server_args: ServerArgs,
    init_tokenizer_manager_func: Callable = init_tokenizer_manager,
    run_scheduler_process_func: Callable = run_scheduler_process,
    run_detokenizer_process_func: Callable = run_detokenizer_process,
    execute_warmup_func: Callable = _execute_server_warmup,
    launch_callback: Optional[Callable[[], None]] = None,
):
    """
    Launch SRT (SGLang Runtime) Server.

    The SRT server consists of an HTTP server and an SRT engine.

    - HTTP server: A FastAPI server that routes requests to the engine.
    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches,
           forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the
           result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager all run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
    """
    # Launch subprocesses
    (
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result,
        subprocess_watchdog,
    ) = Engine._launch_subprocesses(
        server_args=server_args,
        init_tokenizer_manager_func=init_tokenizer_manager_func,
        run_scheduler_process_func=run_scheduler_process_func,
        run_detokenizer_process_func=run_detokenizer_process_func,
    )

    _setup_and_run_http_server(
        server_args,
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result.scheduler_infos,
        subprocess_watchdog,
        execute_warmup_func=execute_warmup_func,
        launch_callback=launch_callback,
    )
```

Four callables are passed in — this is a dependency-injection pattern that lets tests replace any of the three process types with fakes. In normal operation, the defaults (`init_tokenizer_manager`, <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L3738" class="sym-link" target="_blank" rel="noopener noreferrer"><code>run_scheduler_process</code></a>, `run_detokenizer_process`) are used.

The returned 5-tuple is the shape of the rest of the engine:

- `tokenizer_manager` — singleton instance of <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py#L215" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TokenizerManager</code></a>, lives in this process.
- `template_manager` — manages the chat template used to turn `messages=[...]` into a prompt string.
- `port_args` — a <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L6985" class="sym-link" target="_blank" rel="noopener noreferrer"><code>PortArgs</code></a> object describing every ZMQ IPC path that will be used.
- `scheduler_init_result` — a <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/engine.py#L114" class="sym-link" target="_blank" rel="noopener noreferrer"><code>SchedulerInitResult</code></a> struct carrying the scheduler subprocess's reported memory budget, context length, etc.
- `subprocess_watchdog` — background thread that checks scheduler and detokenizer health and kills the server if they die.

### 2.6 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/engine.py#L633" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Engine._launch_subprocesses</code></a>
{: #launch-engine }

This classmethod is the heart of startup. It's called both by `http_server.launch_server` and directly by <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/engine.py#L165" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Engine.__init__</code></a>.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> python/sglang/srt/entrypoints/engine.py:633-720 (excerpt) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/engine.py#L633" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
@classmethod
def _launch_subprocesses(
    cls,
    server_args: ServerArgs,
    init_tokenizer_manager_func: Callable,
    run_scheduler_process_func: Callable,
    run_detokenizer_process_func: Callable,
    port_args: Optional[PortArgs] = None,
) -> Tuple[
    TokenizerManager,
    TemplateManager,
    PortArgs,
    SchedulerInitResult,
    Optional[SubprocessWatchdog],
]:
    """Launch the TokenizerManager in the main process, the Scheduler in a subprocess,
       and the DetokenizerManager in another subprocess."""
    # Configure global environment
    configure_logger(server_args)
    _set_envs_and_config(server_args)

    # Defensive: ensure plugins loaded (may already be loaded by Engine.__init__ or CLI entry).
    load_plugins()

    server_args.check_server_args()
    _set_gc(server_args)

    # Allocate ports for inter-process communications
    if port_args is None:
        port_args = PortArgs.init_new(server_args)
    logger.info(f"{server_args=}")

    # Start the engine info bootstrap server if per-rank info is needed.
    # ... (elastic EP backup manager code omitted) ...

    # Launch scheduler processes
    scheduler_init_result, scheduler_procs = cls._launch_scheduler_processes(
        server_args, port_args, run_scheduler_process_func
    )
    scheduler_init_result.engine_info_bootstrap_server = engine_info_bootstrap_server
    # ... (multi-node rank-0 gating) ...
```

Three things are worth noting:

1. **`server_args.check_server_args()`** is called here, *not* from `__post_init__`. This is where <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L6659" class="sym-link" target="_blank" rel="noopener noreferrer"><code>check_lora_server_args</code></a> fires (§2.3) — which means the LoRA-related mutations (setting `enable_lora=True`, parsing `lora_paths` into <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_registry.py#L27" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRARef</code></a> objects) land just before subprocesses fork.
2. **<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L7006" class="sym-link" target="_blank" rel="noopener noreferrer"><code>PortArgs.init_new</code></a>** allocates a bundle of IPC paths — one for scheduler input, one for tokenizer's reply channel, one for the detokenizer. These are ZMQ IPC paths (typically `ipc:///tmp/sglang-XXX`), not TCP sockets. See §2.7.
3. **`_launch_scheduler_processes`** spawns one scheduler subprocess per TP×PP×DP combination. For `--tp 4 --dp 1 --pp 1` that's exactly 4 subprocesses, each bound to one GPU.

### 2.7 Process topology & ZMQ IPC
{: #launch-ipc }

<div class="code-head" markdown="span">
Process layout for `--tp 4`
</div>

```text
┌───────────────────────────────────────────────────────────────┐
│  Main process                                                 │
│   ├── FastAPI / uvicorn (HTTP)                                │
│   ├── TokenizerManager  (tokenization, multimodal preproc)    │
│   │     ↓  ZMQ PUSH → scheduler_input_ipc_name                │
│   │     ↑  ZMQ PULL ← tokenizer_ipc_name                      │
│   ├── TemplateManager   (chat template)                       │
│   └── SubprocessWatchdog                                      │
└───────────────────────────────────────────────────────────────┘
          ↓ (fork)                             ↓ (fork)
┌─────────────────────────────┐  ┌─────────────────────────────┐
│  Scheduler rank 0 (GPU 0)   │  │  DetokenizerManager         │
│   Scheduler(ScheduleBatch)  │  │   ZMQ PULL ← scheduler_out  │
│    ├── TpModelWorker        │  │   ZMQ PUSH → tokenizer_in   │
│    │    └── ModelRunner     │  └─────────────────────────────┘
│    │         ├── Qwen3Moe   │
│    │         ├── KV pool    │
│    │         ├── LoRA mgr   │
│    │         └── CUDA graph │
│    └── RadixCache (tree)    │
│   NCCL group: (rank 0..3)   │
└─────────────────────────────┘
┌─────────────────────────────┐
│  Scheduler rank 1 (GPU 1)   │   ... rank 2 (GPU 2) ... rank 3 (GPU 3)
└─────────────────────────────┘
```

ZMQ IPC channels use UNIX domain sockets, not TCP. That matters because it sidesteps kernel TCP buffering and gives round-trip latency measured in microseconds. The ZMQ patterns used are simple `PUSH/PULL` queues (one-way pipelines) rather than `REQ/REP` or `PUB/SUB` — TokenizerManager pushes <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/io_struct.py#L694" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TokenizedGenerateReqInput</code></a> messages to the scheduler and pulls `BatchTokenIDOut` messages from the detokenizer.

<div class="callout motiv" markdown="1">

#### Why three processes?

Putting the scheduler in a **dedicated subprocess** is part of SGLang's zero-overhead batching story (<a href="https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/" target="_blank" rel="noopener noreferrer">LMSYS v0.4 blog post</a>): while the GPU runs one batch, the scheduler CPU-side is already preparing the next batch's metadata (positions, seg_indptr, prefix matches, etc.) and launches it on the next stream. The tokenizer-and-detokenizer split then keeps CPU-heavy JSON parsing / BPE decoding off the scheduler's critical path. For rollout frameworks, this process-per-role structure also makes it easy to attach a checkpoint engine, profile each stage independently, or replace the detokenizer with a streaming HTTP worker pool.

</div>

### 2.8 Multi-node wiring: where ZMQ stops and NCCL/Gloo take over
{: #launch-ipc-multi }

The ASCII diagram above is for single-node. When you extend to `--nnodes 2 --tp 16`, a natural question is: **ZMQ `ipc://` sockets are Unix-domain sockets tied to the local filesystem, so how do ranks on node 1 receive requests from the TokenizerManager on node 0?** The answer is that they don't — at least not via ZMQ.

The key invariant is: **only one scheduler rank ever reads from ZMQ — the rank that happens to live on the same node as the TokenizerManager.** Every other rank (on the same or a different node) learns about new requests via a *broadcast* on a CPU-side distributed group.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> managers/scheduler.py:498-540 — init_ipc_channels (only rank-0 binds) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L498" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def init_ipc_channels(self, port_args: PortArgs):
    context = zmq.Context(2)
    self.idle_sleeper = None

    if self.pp_rank == 0 and self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
        self.recv_from_tokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.scheduler_input_ipc_name, False
        )
        self.send_to_tokenizer = SenderWrapper(
            get_zmq_socket(context, zmq.PUSH, port_args.tokenizer_ipc_name, False)
        )
        ...
    else:
        self.recv_from_tokenizer = None
        self.send_to_tokenizer = SenderWrapper(None)
```

All ranks whose `(pp_rank, attn_tp_rank, attn_cp_rank)` is not `(0, 0, 0)` skip the socket setup entirely — their `recv_from_tokenizer` is literally `None`. They'll receive request lists via <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/common.py#L1195" class="sym-link" target="_blank" rel="noopener noreferrer"><code>broadcast_pyobj</code></a> from rank 0, through the TP **CPU process group**:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> managers/scheduler.py:1506-1530 — recv_requests (broadcast to other ranks) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L1506" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def recv_requests(self) -> List[Any]:
    """Receive requests from the tokenizer manager over ZMQ."""
    if self.attn_tp_rank == 0 and self.pp_rank == 0:
        recv_reqs = []
        while True:
            try:
                recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            recv_reqs.append(recv_req)
    else:
        recv_reqs = None
    # Then tp-broadcast to other ranks:
    if self.tp_size > 1:
        work = broadcast_pyobj(recv_reqs, src=attn_tp_rank_0, group=self.tp_cpu_group)
        recv_reqs = work
    return recv_reqs
```

#### What is a "CPU process group"? It's Gloo, not NCCL.

NCCL is GPU-only — it requires tensors allocated in CUDA memory. For Python-object broadcasts we need something that works on CPU. SGLang constructs **two parallel process groups** per logical rank set, one for each backend:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> distributed/parallel_state.py:295-308 — GroupCoordinator builds two groups <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/parallel_state.py#L295" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
pg_options = get_torch_distributed_pg_options(group_name)
device_group = torch.distributed.new_group(
    ranks, backend=torch_distributed_backend, pg_options=pg_options
)
# a group with `gloo` backend, to allow direct coordination
# between processes through the CPU.
cpu_group = torch.distributed.new_group(
    ranks, backend="gloo", timeout=gloo_timeout
)
...
self.device_group = device_group   # NCCL — GPU tensor collectives
self.cpu_group = cpu_group         # Gloo — CPU tensor / Python object broadcasts
```

**Gloo** is one of PyTorch's built-in distributed backends, authored by Meta, that runs over TCP sockets between processes (shared memory on a single node). Every <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/parallel_state.py#L193" class="sym-link" target="_blank" rel="noopener noreferrer"><code>GroupCoordinator</code></a> — the TP group, PP group, MoE-EP group, etc. — holds both a NCCL-backed `device_group` (for the actual model-forward collectives counted in §8.2) and a Gloo-backed `cpu_group` (for Python-object and control-plane messages). The Gloo group bootstraps over the same `dist_init_addr` as NCCL, and because it's TCP, **it works transparently across nodes**.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> utils/common.py:1195-1240 — broadcast_pyobj (serialize → Gloo broadcast) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/common.py#L1195" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def broadcast_pyobj(
    data: List[Any],
    rank: int,
    dist_group: Optional[torch.distributed.ProcessGroup] = None,
    src: int = 0,
    force_cpu_device: bool = True,
):
    """Broadcast inputs from src rank to all other ranks with torch.dist backend."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not force_cpu_device
        else "cpu"
    )

    if rank == src:
        if len(data) == 0:
            tensor_size = torch.tensor([0], dtype=torch.long, device=device)
            dist.broadcast(tensor_size, src=src, group=dist_group)
        else:
            serialized_data = pickle.dumps(data)
            size = len(serialized_data)

            tensor_data = torch.ByteTensor(
                np.frombuffer(serialized_data, dtype=np.uint8)
            ).to(device)
            tensor_size = torch.tensor([size], dtype=torch.long, device=device)

            dist.broadcast(tensor_size, src=src, group=dist_group)
            dist.broadcast(tensor_data, src=src, group=dist_group)
        return data
    else:
        tensor_size = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(tensor_size, src=src, group=dist_group)
        size = tensor_size.item()
        if size == 0:
            return []
        tensor_data = torch.empty(size, dtype=torch.uint8, device=device)
        dist.broadcast(tensor_data, src=src, group=dist_group)
        serialized_data = bytes(tensor_data.cpu().numpy())
        data = pickle.loads(serialized_data)
        return data
```

The function pickles the list of request objects to bytes, wraps them in a **CPU** `ByteTensor`, and broadcasts over the CPU group. PyTorch dispatches this to Gloo because the tensor is CPU-resident.

#### Why three reasons to have a separate CPU group

1. **NCCL requires GPU-resident tensors.** To broadcast a list of <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/io_struct.py#L694" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TokenizedGenerateReqInput</code></a> objects via NCCL you'd have to serialize, pin, copy-to-GPU, broadcast, copy-back. Gloo keeps the whole round-trip in CPU memory.
2. **NCCL operations serialize on CUDA streams.** Using NCCL for control-plane messages would stall GPU compute, because the broadcast kernel would enqueue behind model forwards. Gloo runs on CPU threads, entirely decoupled.
3. **Gloo handles variable-size messages gracefully.** NCCL collectives assume same-size tensors on all ranks; request lists vary in size. The <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/common.py#L1195" class="sym-link" target="_blank" rel="noopener noreferrer"><code>broadcast_pyobj</code></a> pattern (broadcast size first, then payload) sidesteps this.

#### When does ZMQ itself switch to TCP?

Even after the per-rank invariant above, there's still a case where ZMQ needs to cross nodes: `enable_dp_attention=True`. With DP-attention enabled, each DP group has its own rank-0 scheduler, and those rank-0 schedulers can live on different nodes — they can't share a filesystem `ipc://` socket. The code at `server_args.py:7010` switches transports accordingly:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> server_args.py:7010-7080 — PortArgs.init_new (IPC vs TCP selection) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L7010" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
if not server_args.enable_dp_attention:
    # Normal case, use IPC within a single node
    return PortArgs(
        tokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
        scheduler_input_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
        detokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
        ...
    )
else:
    # DP attention. Use TCP + port to handle both single-node and multi-node.
    ...
    return PortArgs(
        tokenizer_ipc_name=NetworkAddress(dist_init_host, port_base).to_tcp(),
        scheduler_input_ipc_name=NetworkAddress(dist_init_host, scheduler_input_port).to_tcp(),
        detokenizer_ipc_name=NetworkAddress(dist_init_host, detokenizer_port).to_tcp(),
        ...
    )
```

So the transport decision matrix is:

| Configuration                         | ZMQ transport                          | Cross-node broadcasts via  |
|---------------------------------------|----------------------------------------|----------------------------|
| Single node, no DP attention          | `ipc://` (Unix domain sockets in /tmp) | N/A — everything local.    |
| `--nnodes 2 --tp 16`, no DP attention | `ipc://` (only used on node 0)         | Gloo CPU group (over TCP). |
| Any `--enable-dp-attention`           | `tcp://dist_init_host:port`            | Gloo CPU group + ZMQ TCP.  |

<div class="callout motiv" markdown="1">

#### Why not just always use TCP ZMQ?

`ipc://` uses Unix domain sockets, which skip TCP's buffer/ACK/handshake machinery and give round-trip latencies of a few microseconds on the same host. TCP on localhost adds ~30 μs per round trip even with loopback optimizations. For high-throughput decoding where the scheduler dispatches batches every ~10 ms, that extra per-batch control-plane latency is measurable. The code defaults to `ipc://` whenever it can and falls back to TCP only when forced by the DP-attention topology.

</div>

<div class="callout warn" markdown="1">

#### Multi-node gotcha: `dist_init_addr` must be reachable from every node

`dist_init_addr` is used both for NCCL's initial handshake and (when DP-attention is on) for ZMQ TCP endpoints. It must resolve to an IP that every node can actually reach. On Kubernetes that's typically the headless Service DNS name; on bare metal it's an IB-side interface IP; on Tailscale-meshed clusters it's the Tailscale hostname of node 0. If `--dist-init-addr` is unreachable from `--node-rank 1`, NCCL init silently hangs with a very unhelpful timeout message.

</div>

---

<p class="bridge" markdown="span">*Now that the processes are spawned and connected, let's look at what each one actually does. The main process hosts the TokenizerManager — the front door of the server.*</p>

## 3 · TokenizerManager init
{: #tokmgr }

<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py#L215" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TokenizerManager</code></a> is built in the main process, before any subprocess fork. Its constructor does nothing GPU-related — it sets up the tokenizer, IPC sockets, request-routing state, and the LoRA *registry* (distinct from the LoRA *manager*, which lives in the scheduler).

### 3.1 Constructor shape
{: #tokmgr-init }

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> python/sglang/srt/managers/tokenizer_manager.py:215-260 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py#L215" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
class TokenizerManager(TokenizerControlMixin, TokenizerManagerScoreMixin):
    """TokenizerManager is a process that tokenizes the text."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # Parse args
        self.server_args = server_args
        self.enable_metrics = server_args.enable_metrics
        self.preferred_sampling_params = server_args.preferred_sampling_params
        self.crash_dump_folder = server_args.crash_dump_folder
        set_global_server_args_for_tokenizer(server_args)

        # Init model config
        self.init_model_config()

        # Initialize tokenizer and multimodalprocessor
        self.init_tokenizer_and_processor()

        # Init inter-process communication
        self.init_ipc_channels(port_args)

        # Init running status
        self.init_running_status()

        # Init logging and dumping
        self.init_request_logging_and_dumping()

        # Init weight update
        self.init_weight_update()

        # Init LoRA status
        self.init_lora()

        # Init PD disaggregation and encoder disaggregation
        self.init_disaggregation()

        # Init metric collector and watchdog
        self.init_metric_collector_watchdog()

        # Init request dispatcher
        self.init_request_dispatcher()
```

Two sub-steps are especially relevant to our walkthrough — IPC channel setup and LoRA registry init. The rest (running-status tracking, logging, weight-update manager, metrics, watchdog, disaggregation) is bookkeeping that doesn't affect the hot path.

### 3.2 IPC channels (ZMQ pull/push)
{: #tokmgr-ipc }

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> tokenizer_manager.py:344-363 — init_ipc_channels <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py#L344" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def init_ipc_channels(self, port_args: PortArgs):
    context = zmq.asyncio.Context(2)
    self.recv_from_detokenizer = get_zmq_socket(
        context, zmq.PULL, port_args.tokenizer_ipc_name, True
    )
    if self.server_args.tokenizer_worker_num == 1:
        self.send_to_scheduler = get_zmq_socket(
            context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
        )
    else:
        from sglang.srt.managers.multi_tokenizer_mixin import SenderWrapper
        send_to_scheduler = get_zmq_socket(
            context, zmq.PUSH, port_args.tokenizer_worker_ipc_name, False
        )
        self.send_to_scheduler = SenderWrapper(port_args, send_to_scheduler)
```

Exactly two sockets: a **PULL** for results from the detokenizer and a **PUSH** to the scheduler. In multi-worker mode (`tokenizer_worker_num > 1`) each worker pushes to a shared `tokenizer_worker_ipc_name` and the <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/multi_tokenizer_mixin.py#L509" class="sym-link" target="_blank" rel="noopener noreferrer"><code>SenderWrapper</code></a> attaches the worker's identity so the scheduler can route the response back to the right worker.

### 3.3 LoRA registry (not the manager)
{: #tokmgr-lora }

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> tokenizer_manager.py:420-428 — init_lora <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py#L420" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def init_lora(self):
    # LoRA
    # Initialize the `LoRARegistry` with initial LoRA adapter paths provided in `server_args`.
    # The registry dynamically updates as adapters are loaded / unloaded during runtime. It
    # serves as the source of truth for available adapters and maps user-friendly LoRA names
    # to internally used unique LoRA IDs.
    self.lora_registry = LoRARegistry(self.server_args.lora_paths)
    # Lock to serialize LoRA update operations.
    # Please note that, unlike `model_update_lock`, this does not block inference, allowing
    # LoRA updates and inference to overlap.
    self.lora_update_lock = asyncio.Lock()
```

**This is the naming registry, not the weight manager.** Two distinct components, both called "LoRA":

- <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_registry.py#L54" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRARegistry</code></a> lives in <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py#L215" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TokenizerManager</code></a> (main process). Maps user-facing `lora_name` strings to internal UUIDs. When a request says `"model": "Qwen:adapter0"` the registry decides that "adapter0" → some UUID, which then gets attached to the request's `lora_ids` field before it's sent to the scheduler.
- <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L53" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAManager</code></a> lives in the **scheduler subprocess**, owned by <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L292" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelRunner</code></a>. It owns the GPU memory pool, the layer wrappers, and the kernel backend. See §6.

The registry can be updated at runtime via the `/load_lora_adapter` and `/unload_lora_adapter` HTTP endpoints; it serializes writes with `lora_update_lock` but does not block reads/inference.

---

<p class="bridge" markdown="span">*TokenizerManager hands off tokenized requests to the scheduler via ZMQ. The scheduler subprocess is where actual batching, KV allocation, and GPU dispatch happens.*</p>

## 4 · Scheduler subprocess
{: #sched }

Four scheduler processes fork off (one per TP rank). Each will: set up its distributed group, build a <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tp_worker.py#L217" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TpModelWorker</code></a> (which owns a <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L292" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelRunner</code></a>), pick a tree-cache flavor, and run an event loop forever.

### 4.1 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L3738" class="sym-link" target="_blank" rel="noopener noreferrer"><code>run_scheduler_process</code></a>
{: #sched-entry }

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> python/sglang/srt/managers/scheduler.py:3738 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L3738" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    # Load plugins so hooks can override Scheduler and its dependencies.
    load_plugins()
    dp_rank = configure_scheduler_process(
        server_args, gpu_id, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank, pp_rank, dp_rank,
    )
    parent_process = psutil.Process().parent()

    # Set up tracing ...

    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(
            server_args, port_args, gpu_id,
            tp_rank, moe_ep_rank, pp_rank, attn_cp_rank, moe_dp_rank, dp_rank,
        )
        pipe_writer.send(scheduler.get_init_info())
        scheduler.run_event_loop()   # blocks forever
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
```

The signature carries **six different ranks**, not three. This is because SGLang supports overlapping parallelism dimensions:

- `tp_rank` — tensor-parallel rank, splits attention heads and MLP dims.
- `attn_cp_rank` — context-parallel rank for attention (sequence-dim sharding).
- `moe_dp_rank`, `moe_ep_rank` — data / expert parallelism inside MoE.
- `pp_rank` — pipeline-parallel rank (layer-dim sharding).
- `dp_rank` — outer data-parallel rank (for dp-attention or fully replicated engines).

For our `--tp 4` run, only `tp_rank ∈ {0..3}` is non-zero; every other rank is 0.

<div class="callout info" markdown="1">

#### Invariant: one <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tp_worker.py#L217" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TpModelWorker</code></a> per scheduler process, one scheduler per GPU

Despite the seven rank-like integers above, the actual process count doesn't multiply out. **Every GPU in the system gets exactly one scheduler process, and every scheduler holds exactly one <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tp_worker.py#L217" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TpModelWorker</code></a>** (which owns one <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L292" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelRunner</code></a>, which owns one `torch.nn.Module`). The seven integers describe where that process sits within the various logical partitionings, not how many processes exist.

The arithmetic is: total scheduler processes = `dp_size × pp_size × tp_size`, where `tp_size` is the flat "world size per PP stage" and internally decomposes two different ways depending on which group you're looking at:

```text
Attention view:  tp_size = attn_tp_size × attn_dp_size × attn_cp_size
MoE view:        tp_size = moe_tp_size × moe_ep_size × moe_dp_size
```

Both decompositions must equal the same `tp_size`. Each rank simultaneously plays one role in the attention partitioning and one in the MoE partitioning — same GPU, two different logical group memberships.

One exception: if speculative decoding is enabled, the scheduler *also* holds a `draft_worker` for the draft model (see `scheduler.py:687`). So a rank can technically have two workers in that mode, but the target-model worker is still just one.

</div>

### 4.2 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L332" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Scheduler.__init__</code></a>
{: #sched-init }

This is an enormous constructor (~700 lines). The important milestones are: building the TpModelWorker, querying it for the memory budget, then creating the tree cache. Here's the worker-creation part:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> scheduler.py:~633 (TP worker construction, excerpt) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L633" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
# Inside Scheduler.__init__:
if self.server_args.mlx_mode:
    self.tp_worker = MlxTpModelWorker(**worker_kwargs)
else:
    self.tp_worker = TpModelWorker(**worker_kwargs)
# ... if speculative decoding is enabled: wrap with spec_info ...

(
    self.max_total_num_tokens,
    self.max_req_input_len,
    self.max_running_requests,
    self.max_queued_requests,
    self.model_config,
    self.worker_init_log,
    tokenizer_object,
) = self.tp_worker.get_worker_info()
# ... more: pad_input_ids_func, sliding_window_size, etc. ...
(
    self.req_to_token_pool,
    self.token_to_kv_pool_allocator,
) = self.tp_worker.get_memory_pool()
```

<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tp_worker.py#L217" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TpModelWorker</code></a> constructs a <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L292" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelRunner</code></a>, which is where the model actually loads onto the GPU — that's all of §5. After the ModelRunner finishes, the worker reports back the total token budget (`max_total_num_tokens`), and the scheduler now knows how big a tree cache it can afford.

### 4.3 Tree-cache selector — 9+ flavors
{: #sched-tree }

Right after memory-pool handshake, the scheduler picks a tree-cache implementation. This is *not* a binary "radix vs chunk" — there are ~9 classes, and the right one depends on attention type (full / SWA / MLA / Mamba / hybrid) and host-device-copy needs.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> scheduler.py:820-896 (tree_cache selector, excerpt) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L820" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
# (hicache / swa / mamba / mla / lmc branches, simplified):
if self.server_args.disable_radix_cache:
    if self.sliding_window_size is None:
        from sglang.srt.mem_cache.chunk_cache import ChunkCache
        self.tree_cache = ChunkCache(params)
    else:
        from sglang.srt.mem_cache.chunk_cache import SWAChunkCache
        self.tree_cache = SWAChunkCache(params)
elif self.server_args.enable_radix_cache_cpp:
    from sglang.srt.mem_cache.radix_cache_cpp import RadixCacheCpp
    self.tree_cache = RadixCacheCpp(params=params, server_args=server_args)
elif self.server_args.enable_hierarchical_cache:
    if self.is_hybrid_mamba:
        from ... import HiMambaRadixCache
        self.tree_cache = HiMambaRadixCache(...)
    else:
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
        self.tree_cache = HiRadixCache(...)
    self.tp_worker.register_hicache_layer_transfer_counter(
        self.tree_cache.cache_controller.layer_done_counter
    )
elif ... UnifiedRadixCache ...:
    self.tree_cache = UnifiedRadixCache(params)
elif self.sliding_window_size is not None:
    from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache
    self.tree_cache = SWARadixCache(params=params)
elif self.is_hybrid_mamba:
    from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
    self.tree_cache = MambaRadixCache(params)
elif self.server_args.enable_lmcache:
    from ... import LMCRadixCache
    self.tree_cache = LMCRadixCache(...)
else:
    self.tree_cache = RadixCache(params)

if (... needs streaming ...) and not self.tree_cache.supports_streaming_session():
    self.tree_cache = StreamingSession(self.tree_cache)
```

For Qwen3-30B-A3B-Instruct-2507 with default flags (full attention, no MLA, no Mamba, no hicache, no LMCache), we land on **<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py#L285" class="sym-link" target="_blank" rel="noopener noreferrer"><code>RadixCache</code></a>** — SGLang's classic radix-tree prefix-cache.

<div class="callout motiv" markdown="1">

#### What RadixCache does

Described in the original <a href="https://www.lmsys.org/blog/2024-01-17-sglang/" target="_blank" rel="noopener noreferrer">RadixAttention blog</a> and paper. Instead of hashing whole blocks (vLLM's approach), SGLang maintains a token-level radix tree over every active sequence. Two requests with a shared prefix share the same tree path and reuse each other's KV — even dynamic branching (tree-of-thought, few-shot sharing) works out of the box. When memory pressure grows, an LRU eviction removes leaf-most nodes. Up to 5× throughput improvement on prompts with shared prefixes.

</div>

### 4.4 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L1373" class="sym-link" target="_blank" rel="noopener noreferrer"><code>run_event_loop</code></a> / <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L3652" class="sym-link" target="_blank" rel="noopener noreferrer"><code>dispatch_event_loop</code></a>
{: #sched-loop }

After the init finishes, the scheduler enters `run_event_loop()`. This is a thin wrapper that sets up a CUDA stream and delegates to `dispatch_event_loop(self)`, which picks the right loop for the current config.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> scheduler.py:1373-1384 — run_event_loop <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L1373" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def run_event_loop(self) -> None:
    """Run the scheduler's event loop.

    Sets up the schedule stream and dispatches to the appropriate event loop.
    The event loop blocks until shutdown.
    """
    self.schedule_stream = self.device_module.Stream(priority=0)
    if self.device == "cpu":
        self.schedule_stream.synchronize = lambda: None  # No-op for CPU
    with self.device_module.StreamContext(self.schedule_stream):
        dispatch_event_loop(self)
```

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> scheduler.py:3652-3678 — dispatch_event_loop <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L3652" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def dispatch_event_loop(scheduler: Scheduler):
    # Dispatch to the appropriate event loop based on the disaggregation mode
    server_args = scheduler.server_args
    disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
    if disaggregation_mode == DisaggregationMode.NULL:
        if scheduler.enable_pdmux:
            scheduler.event_loop_pdmux()
        elif server_args.pp_size > 1:
            scheduler.event_loop_pp()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
    elif disaggregation_mode == DisaggregationMode.PREFILL:
        if server_args.pp_size > 1:
            scheduler.event_loop_pp_disagg_prefill()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_prefill()
        else:
            scheduler.event_loop_normal_disagg_prefill()
    elif disaggregation_mode == DisaggregationMode.DECODE:
        if server_args.pp_size > 1:
            scheduler.event_loop_pp_disagg_decode()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_decode()
        else:
            scheduler.event_loop_normal_disagg_decode()
```

That's **eight distinct loops**. For our default run (`pp_size=1`, no PD disagg, overlap enabled), we take the <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L1414" class="sym-link" target="_blank" rel="noopener noreferrer"><code>event_loop_overlap</code></a> branch:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> scheduler.py:1386-1411 — event_loop_normal (for reference) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L1386" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def event_loop_normal(self):
    """A normal scheduler loop."""
    while True:
        # Receive requests
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)
        if self._engine_paused:
            self.cancel_bubble_timer()
            continue

        # Get the next batch to run
        batch = self.get_next_batch_to_run()
        self.cur_batch = batch

        # Launch the current batch
        if batch:
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
        else:
            # When the server is idle, do self-check and re-init some states.
            self.on_idle()

        # Update last_batch
        self.last_batch = batch
        if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
            self.self_check_during_busy()
```

The <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L1414" class="sym-link" target="_blank" rel="noopener noreferrer"><code>event_loop_overlap</code></a> variant maintains a `deque` of `(batch, result)` pairs and peels work off *one batch ahead* of the GPU — CPU-side it's already preparing the next batch's metadata while the GPU runs the current one. This is "zero-overhead batching" from the <a href="https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/" target="_blank" rel="noopener noreferrer">v0.4 blog post</a>.

### 4.5 The overlap scheduler in detail — what the CPU does while the GPU works
{: #sched-overlap }

<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L1414" class="sym-link" target="_blank" rel="noopener noreferrer"><code>event_loop_overlap</code></a> is the default loop (enabled unless you pass `--disable-overlap-schedule`). This section is an audit of every state update that happens per iteration. It's worth being careful here because the design makes a deliberate correctness-vs-throughput trade that isn't obvious from the outside.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> managers/scheduler.py:1414-1465 — event_loop_overlap <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L1414" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def event_loop_overlap(self):
    """A scheduler loop that overlaps the CPU processing and GPU computation."""
    self.result_queue: Deque[
        Tuple[ScheduleBatch, Union[GenerationBatchResult, EmbeddingBatchResult]]
    ] = deque()

    def pop_and_process():
        # Process the results of the last batch
        tmp_batch, tmp_result = self.result_queue.popleft()
        self.process_batch_result(tmp_batch, tmp_result)

    while True:
        # [Point A] Receive requests and form the next batch
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)
        if self._engine_paused:
            continue

        batch = self.get_next_batch_to_run()
        self.cur_batch = batch
        disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)

        if disable_overlap_for_batch:
            pop_and_process()

        # [Point B] Launch current batch on GPU (async)
        if batch:
            batch_result = self.run_batch(batch)
            self.result_queue.append((batch.copy(), batch_result))
        else:
            batch_result = None
            self.cancel_bubble_timer()

        # [Point C] Process the PREVIOUS iteration's result
        if self.last_batch:
            if not disable_overlap_for_batch:
                pop_and_process()
        elif batch is None:
            self.on_idle()

        # [Point D] Run sample for the current batch (Path B only)
        # It depends on the result of the last batch (e.g., grammar),
        # so we run it after the last batch is processed.
        if self.is_generation:
            self.launch_batch_sample_if_needed(batch_result)

        # [Point E] Rotate last_batch pointer
        self.last_batch = batch
```

<div class="callout warn" markdown="1">

#### What `pop_and_process` actually does

It's tempting to think of `pop_and_process` as mostly data movement, but it does substantial state updates — including determining whether each request has finished — and those updates are what make the overlap scheduler work. The full list below is extracted from <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler_output_processor_mixin.py#L387" class="sym-link" target="_blank" rel="noopener noreferrer"><code>process_batch_result_decode</code></a>.

</div>

#### Every operation in `pop_and_process` for a decode batch

Taken directly from <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler_output_processor_mixin.py#L387" target="_blank" rel="noopener noreferrer"><code>scheduler_output_processor_mixin.py:387-535</code></a> — <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler_output_processor_mixin.py#L387" class="sym-link" target="_blank" rel="noopener noreferrer"><code>process_batch_result_decode</code></a>:

1. **`result.copy_done.synchronize()`** — wait for the async D2H copy of the sampled token IDs to complete. In overlap mode, this synchronize runs in parallel with the GPU crunching the next iteration's forward, so it's effectively free wall-clock.
2. **`next_token_ids.tolist()`** — convert the (now-CPU) sampled-IDs tensor to a Python list. Also converts logprob tensors if `return_logprob` is set.
3. **`self.num_generated_tokens += len(batch.reqs)`** — global throughput metric.
4. **`self.token_to_kv_pool_allocator.free_group_begin()`** — open a transactional region for batched KV frees that may occur below.
5. For each request in `batch.reqs`:
    1.  **Skip-if-finished guard.** If `self.enable_overlap and (req.finished() or req.is_retracted)`, `continue`. This is the critical overlap-specific path — more on it below.
    2.  **`req.output_ids.append(next_token_id)`** — extend the Python output list.
    3.  **`self._maybe_update_reasoning_tokens(req, next_token_id)`** — if the request uses a reasoning parser (`<think>`-style), update the in-think vs out-of-think state.
    4.  **`self._mamba_prefix_cache_update(...)`** — Mamba SSM bookkeeping if the model has Mamba layers.
    5.  **`req.time_stats.set_last_decode_finish_time()`** — per-request latency metric.
    6.  **`req.check_finished(new_accepted_len)`** — **this is where finish detection happens.** It sets `req.finished_reason` if the newly-appended token triggers any of: EOS token, `max_new_tokens` reached, grammar FSM terminated, stop-token match, stop-string match, stop-regex match, or NaN detection.
    7.  **`self._handle_finished_req(req, i, logits_output)`** — if finished: release KV cache slots back to the pool, write the request's full token sequence into the radix tree (for future prefix reuse), release multimodal features, and send the final response packet to the detokenizer.
    8.  Accumulate logprobs / hidden states if the request requested them.
    9.  **`req.grammar.accept_token(next_token_id)`** — if this request is under grammar constraint, advance its FSM state.
6. **`self.stream_output(batch.reqs, batch.return_logprob)`** — ship all the new tokens + metadata to the detokenizer process over ZMQ.
7. **`self.token_to_kv_pool_allocator.free_group_end()`** — commit any KV frees accumulated in step 4.
8. Metrics: `forward_ct_decode`, `report_decode_stats`.

<div class="callout info" markdown="1">

#### Why this is consistent: <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L2302" class="sym-link" target="_blank" rel="noopener noreferrer"><code>get_next_batch_to_run</code></a> reads a one-iteration-stale `req.finished()`

At Point A of iteration N+1, the scheduler calls <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L2302" class="sym-link" target="_blank" rel="noopener noreferrer"><code>get_next_batch_to_run</code></a> → <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L2667" class="sym-link" target="_blank" rel="noopener noreferrer"><code>update_running_batch</code></a> → <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py#L2261" class="sym-link" target="_blank" rel="noopener noreferrer"><code>filter_batch</code></a>, which reads `req.finished()` to decide which requests stay in the running batch. But iteration N's token has just been sampled on GPU — it hasn't been D2H-copied, appended to `output_ids`, or passed through <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py#L1169" class="sym-link" target="_blank" rel="noopener noreferrer"><code>check_finished</code></a>. Those happen later at Point C.

So **the `req.finished()` reading at Point A of iteration N+1 reflects state as of iteration N-1's <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py#L1169" class="sym-link" target="_blank" rel="noopener noreferrer"><code>check_finished</code></a>**, not iteration N's. It's one iteration stale. This is deliberate: making iteration N+1's batch formation wait for iteration N's results would serialize CPU post-processing and GPU next-forward, defeating the entire purpose of overlap.

</div>

#### The "over-allocated tokens" trade-off

A request whose iteration-N sample was EOS is *still present* in the running batch at iteration N+1's Point A, because the filter at that point didn't know about iteration N's EOS yet. So iteration N+1's GPU forward pass runs for it anyway — producing another token, allocating another KV row, sampling logits. All of that work is wasted: when iteration N+1's `pop_and_process` runs (at the *following* iteration's Point C), the per-request guard discards it:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> scheduler_output_processor_mixin.py:440-444 — the overlap skip guard <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler_output_processor_mixin.py#L440" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
if self.enable_overlap and (req.finished() or req.is_retracted):
    # NOTE: This (req.finished() or req.is_retracted) should only happen when overlap scheduling is enabled.
    # And all the over-allocated tokens will be freed in `release_kv_cache`.
    continue
```

So in overlap mode, **every request wastes exactly one forward pass at the end of its generation** — the one that ran between iteration N's EOS-sample and iteration N+1's EOS-detection. The KV rows allocated for that wasted step get reclaimed when `release_kv_cache` runs inside <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler_output_processor_mixin.py#L543" class="sym-link" target="_blank" rel="noopener noreferrer"><code>_handle_finished_req</code></a>.

This is a throughput cost, not a correctness issue. The extra token is never delivered to the user — it's discarded at step 5a above.

#### Why grammar is different: a correctness issue, not a throughput one

For EOS / max_tokens / stop-sequence, the worst case of stale state is one wasted forward pass. For grammar, stale state is an actual correctness bug.

Grammar FSMs produce a **vocab mask** applied to logits before sampling: "of these 151 936 tokens, only these are legal." When a token is sampled, the FSM advances and the mask for the *next* sample is different. If iteration N+1 sampled with iteration N's pre-advance FSM state, the sampler could legally pick a token that violates the new grammar state.

Concretely, suppose a request is generating a JSON object under schema `{"name": string, "age": int}`:

{% raw %}
```text
iter N:      FSM state: "expect '{'"
             sample produces '{' — correct.
             ... pop_and_process(batch_{N-1}) runs between iterations ...
             iter N's sample is ON GPU but not yet accepted into the FSM.

iter N+1:    FSM state: still "expect '{'" if no correction is made.
             Without correction, iter N+1 samples under the mask for "expect '{'",
             could produce '{' again — invalid JSON "{{".

             Path B fixes this:
             - run_batch(iter N+1): forward computes logits but DOES NOT sample;
               closure stashed in batch_result.delay_sample_func
             - pop_and_process(batch_N): appends '{' to output_ids,
               calls req.grammar.accept_token('{') — FSM advances to "expect '\"'"
             - launch_batch_sample_if_needed(iter N+1): closure fires NOW,
               samples under the correct mask for "expect '\"'" — produces '\"' correctly.
```
{% endraw %}

The check that triggers Path B lives in <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tp_worker.py#L217" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TpModelWorker</code></a>:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> managers/tp_worker.py:483-497 — sample inline vs delayed <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tp_worker.py#L483" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
if (
    self.enable_overlap
    and not self.enable_spec
    and model_worker_batch.sampling_info.grammars is not None
):
    def sample_batch_func():
        batch_result.next_token_ids = self.model_runner.sample(
            logits_output, forward_batch
        )
        return batch_result

    batch_result.delay_sample_func = sample_batch_func
    return batch_result

if not model_worker_batch.is_prefill_only:
    # For normal requests, sample the next token ids.
    batch_result.next_token_ids = self.model_runner.sample(
        logits_output, forward_batch
    )
```

Path A (`grammars is None`) samples inline and lives with the one-iteration-stale `req.finished()`. Path B (`grammars is not None`) defers sampling so the FSM has the newest state when the sample happens.

<div class="callout motiv" markdown="1">

#### Why not just make Path B the default, and avoid over-allocated tokens entirely?

Because Path B serializes the critical path: iteration N+1's GPU sample cannot launch until `pop_and_process(batch_N)` has completed on CPU. That reintroduces exactly the CPU-stall that overlap was designed to eliminate. The one-iteration over-allocation in Path A is strictly cheaper than the CPU serialization in Path B, for any request that doesn't need grammar-accurate sampling.

</div>

#### How `input_ids` for iteration N+1 gets iteration N's sampled token

If `req.output_ids` on CPU is stale at Point A of iteration N+1, how does iteration N+1's GPU forward pass know what input token to feed each request? Via the **future-index** mechanism, which bypasses CPU `output_ids` entirely.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> managers/scheduler.py:2796-2815 — run_batch overlap branch (future indices) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L2796" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
bs = len(model_worker_batch.seq_lens)
future_indices = self.future_map.alloc_future_indices(bs)

with self.forward_stream_ctx, self.record_bubble_metrics(batch):
    self.forward_stream.wait_stream(self.schedule_stream)
    self.future_map.resolve_future(model_worker_batch)
    with self.record_forward_metrics(batch):
        batch_result = self.model_worker.forward_batch_generation(
            model_worker_batch
        )
    batch_result.copy_done = self.device_module.Event()
    if batch_result.delay_sample_func is None:
        self.future_map.store_to_map(future_indices, batch_result)
        batch_result.copy_to_cpu(return_logprob=batch.return_logprob)
    else:
        batch_result.future_indices = future_indices

future_indices_or_next_token_ids = -future_indices.indices
...
batch.output_ids = future_indices_or_next_token_ids
```

Three things happen per iteration:

1. **Allocate future indices.** `future_map.alloc_future_indices(bs)` advances a circular buffer pointer and returns a slice of indices. These are the "slots" in a GPU-resident buffer where iteration N's sampled tokens will eventually land.
2. **Resolve earlier futures before forward.** `future_map.resolve_future(model_worker_batch)` patches `input_ids` on GPU: wherever it finds a negative value (a forward reference), it looks the actual token up in `token_ids_buf` and substitutes it in place. This happens on the forward stream, serialized after any previous sample kernel, so it always reads the latest tokens.
3. **Assign placeholders for downstream consumers.** `batch.output_ids = -future_indices.indices` — negative indices that iteration N+2 will resolve via the same <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/overlap_utils.py#L130" class="sym-link" target="_blank" rel="noopener noreferrer"><code>resolve_future</code></a> call, unless iteration N's tokens have arrived on CPU by then (they typically haven't).

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> managers/overlap_utils.py:21-27 — _resolve_future_token_ids <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/overlap_utils.py#L21" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def _resolve_future_token_ids_native(input_ids, future_token_ids_map):
    input_ids[:] = torch.where(
        input_ids < 0,
        future_token_ids_map[torch.clamp(-input_ids, min=0)],
        input_ids,
    )
```

The mechanism is a GPU-side late-binding: negative values in `input_ids` are indices into `future_token_ids_map`; the kernel rewrites them just before forward runs. Because this happens on the forward stream and the sample kernel from the previous iteration writes to the same map (also on the forward stream), ordering is guaranteed without any explicit synchronization.

<div class="callout info" markdown="1">

#### Three pieces of state that the scheduler handles differently

| State                          | Where it lives                 | Update timing                                                                                                                                                                                                                           | Staleness handling                                                                                       |
|--------------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `input_ids` for next forward   | GPU tensor                     | Written by sample kernel (iter N), read by resolve_future (iter N+1)                                                                                                                                                                    | Always correct — resolve runs on the forward stream after sample.                                        |
| `req.output_ids` (Python list) | CPU list                       | Appended inside `pop_and_process`                                                                                                                                                                                                       | Not needed at Point A — the scheduler uses `seq_lens` and future indices instead.                        |
| `req.finished_reason`          | CPU Python attr                | Set inside <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py#L1169" class="sym-link" target="_blank" rel="noopener noreferrer"><code>check_finished</code></a> in `pop_and_process` | Stale by one iteration at Point A. Wastes one forward pass per completion; safe for EOS/max_tokens/stop. |
| `req.grammar` FSM state        | CPU (xgrammar/outlines object) | Advanced inside `accept_token` in `pop_and_process`                                                                                                                                                                                     | Staleness is a correctness bug → Path B delays sampling.                                                 |

</div>

#### Quick summary of the life of one request across consecutive batches

Suppose user A's request prefills at iteration 5 and generates 100 tokens before EOS:

- **Iterations 5-104:** request is in `running_batch`. Each iteration produces one new token.
- **Iteration 104:** sample produces EOS. On GPU: stored in `token_ids_buf` slot. On CPU: not yet known.
- **Iteration 105:** Point A's filter_batch reads `req.finished() == False` (one iteration stale), keeps request in batch. Point B launches forward. <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/overlap_utils.py#L130" class="sym-link" target="_blank" rel="noopener noreferrer"><code>resolve_future</code></a> fetches the EOS token as iteration 105's input — meaning the model does a forward pass conditioned on EOS as if it were generating more. At Point C (pop_and_process for iteration 104), `output_ids.append(EOS)` + <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py#L1169" class="sym-link" target="_blank" rel="noopener noreferrer"><code>check_finished</code></a> sets `finished_reason = FINISH_MATCHED_TOKEN`. Nothing iteration 105-specific has been post-processed yet.
- **Iteration 106:** Point A's filter_batch sees `req.finished() == True` now, removes request from running_batch. Point C's pop_and_process for iteration 105 hits the skip guard (req.finished is True), discards the over-allocated token + releases KV.

Net: request uses 100 actual decode forwards (iter 5-104) + 1 wasted decode forward (iter 105). The 1 wasted pass is the overlap tax.

#### The `is_disable_overlap_for_batch` escape hatch

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> managers/scheduler.py:1468-1490 — is_disable_overlap_for_batch
</div>

```python
def is_disable_overlap_for_batch(self, batch: ScheduleBatch) -> bool:
    # For two consecutive prefill batches, we disable overlap to improve the TTFT
    # of the first batch. This might slightly hurt the throughput, so we use an
    # environment variable to control it.
    ...
```

For back-to-back prefill batches, overlap is automatically disabled so the first request's TTFT (time-to-first-token) doesn't eat the full overlap latency. This is a tiny latency-over-throughput trade.

#### Answer summary to "when does a new request join a batch?"

- A request arriving at the ZMQ socket **after** iteration N's `recv_requests()` sits in the socket buffer until iteration N+1's `recv_requests()` drains it.
- It lands in `self.waiting_queue` via <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L1693" class="sym-link" target="_blank" rel="noopener noreferrer"><code>process_input_requests</code></a>.
- At iteration N+1's <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L2302" class="sym-link" target="_blank" rel="noopener noreferrer"><code>get_next_batch_to_run</code></a>, it becomes eligible for batch formation — as either a new prefill batch (preempting decode) or as an addition to a mixed prefill-decode batch.
- Prefill can span multiple iterations if the prompt exceeds `chunked_prefill_size`.
- Once prefilled, the request joins `self.running_batch` and participates in every decode iteration until its <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py#L1169" class="sym-link" target="_blank" rel="noopener noreferrer"><code>check_finished</code></a> fires — at which point the over-allocated one-more-forward pattern runs one last time before the KV is released.

Latency from "request arrives at scheduler" to "first forward pass includes this request" is bounded by one iteration — typically 20-30 ms for decode, but up to several hundred ms if a prefill is currently running.

---

<p class="bridge" markdown="span">*The scheduler decides **what** to run; `ModelRunner` owns **how** — model weights, KV pool, attention backend, CUDA graphs. This is the largest part of the doc because it's the largest part of the system.*</p>

## 5 · ModelRunner: weights, KV, graphs
{: #runner }

<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L292" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelRunner</code></a> is where the model actually lands on the GPU. It owns the `nn.Module`, the KV pool, the attention backend, the CUDA graphs, and the LoRA manager. All four scheduler subprocesses run one each in parallel, sharing an NCCL process group for TP collectives.

### 5.1 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L526" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelRunner.initialize</code></a> — the real order of operations
{: #runner-init }

The constructor itself is mostly attribute-setting. The real work happens in `initialize(pre_model_load_memory)`, which runs *after* NCCL is up and the TP group is joined. Here's the actual order on `main`:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> python/sglang/srt/model_executor/model_runner.py:526 — initialize() <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L526" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def initialize(self, pre_model_load_memory: float):
    server_args = self.server_args

    self.memory_saver_adapter = TorchMemorySaverAdapter.create(
        enable=self.server_args.enable_memory_saver
    )
    # ... remote-weight-loader setup ...

    if not self.is_draft_worker:
        set_global_expert_location_metadata(
            compute_initial_expert_location_metadata(
                server_args=server_args, model_config=self.model_config,
                moe_ep_rank=self.moe_ep_rank,
            )
        )
        set_global_expert_distribution_recorder(
            ExpertDistributionRecorder.init_new(server_args, get_global_expert_location_metadata(), rank=self.tp_rank)
        )

    # Expert parallelism
    self.eplb_manager = EPLBManager(self) if self.server_args.enable_eplb and (not self.is_draft_worker) else None
    self.expert_location_updater = ExpertLocationUpdater()
    ElasticEPStateManager.init(self.server_args) if self.server_args.elastic_ep_backend else None

    # Load the model
    self.sampler = create_sampler()
    self.load_model()

    # ... expert_backup_client, remote_instance weight registration, MTP-layer math ...

    # Apply torchao quantization
    torchao_applied = getattr(self.model, "torchao_applied", False)
    if not torchao_applied:
        apply_torchao_config_to_model(self.model, get_global_server_args().torchao_config)

    # Apply torch TP if the model supports it
    supports_torch_tp = getattr(self.model, "supports_torch_tp", False)
    if self.tp_size > 1 and supports_torch_tp:
        self.apply_torch_tp()

    # Init lora
    if server_args.enable_lora:
        self.init_lora_manager()
        if not server_args.disable_cuda_graph:
            # Phase 1 of LoRA CUDA graph init: pre-allocate large MoE
            # intermediate buffers before init_memory_pool() so memory
            # profiling accounts for them.  Phase 2 (dense LoRA batch
            # metadata) is handled in CudaGraphRunner.__init__() via
            # lora_manager.init_cuda_graph_batch_info().
            self._init_lora_cuda_graph_moe_buffers()

    # Deduce KV cache dtype
    self.configure_kv_cache_dtype()

    # Init memory pool and attention backends
    self.init_memory_pool(pre_model_load_memory)

    # ngram embedding, hisparse, routed expert capturer, aux hidden state ...
    self.maybe_init_ngram_embedding()
    self.init_routed_experts_capturer()
    self.init_aux_hidden_state_capture()

    if self.device == "cuda" or self.device == "musa":
        self.init_cublas()
        self.init_attention_backend()
        self.kernel_warmup()
        self._pre_initialize_flashinfer_allreduce_workspace()
        self.init_device_graphs()
    elif self.device in ["npu", "cpu"]:
        self.init_attention_backend()
        self.init_device_graphs()
    # ... out-of-tree platforms ...

    if server_args.forward_hooks:
        register_forward_hooks(self.model, server_args.forward_hooks)

    # Initialize piecewise CUDA graph
    self.init_piecewise_cuda_graphs()
    self.prealloc_symmetric_memory_pool()
```

The parts that concern us for a Qwen3-30B-A3B-Instruct-2507 + LoRA run:

1. `self.sampler = create_sampler()` — builds a CUDA-side sampler (top-k/top-p, grammar support, etc.).
2. `self.load_model()` — section §5.4–§5.8. Calls <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py#L675" class="sym-link" target="_blank" rel="noopener noreferrer"><code>DefaultModelLoader.load_model</code></a>, which calls <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_moe.py#L1099" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Qwen3MoeForCausalLM.load_weights</code></a>, which routes every safetensors tensor through `weight_loader` hooks.
3. `self.init_lora_manager()` — §6.2. Creates the <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L53" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAManager</code></a>, which wraps every target `nn.Linear` in a <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L30" class="sym-link" target="_blank" rel="noopener noreferrer"><code>BaseLayerWithLoRA</code></a> subclass and allocates the <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/mem_pool.py#L49" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAMemoryPool</code></a>.
4. `self._init_lora_cuda_graph_moe_buffers()` — **Phase 1** of LoRA CUDA graph init. Pre-allocates large MoE intermediate buffers *before* the KV memory pool is sized, so the profiler sees them as committed and doesn't over-allocate KV.
5. `self.init_memory_pool(pre_model_load_memory)` — allocates KV pool and `req_to_token_pool`. This is where the "KV Cache is allocated. #tokens=X" log line comes from.
6. `init_attention_backend()` — picks the attention kernel family (FA3 on H100/H200, FlashInfer on Blackwell, Triton as fallback) — §5.10.
7. `init_device_graphs()` — captures CUDA graphs at `cuda_graph_max_bs` and its reductions — §5.11. This is **Phase 2** of LoRA CUDA graph init: <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L110" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAManager.init_cuda_graph_batch_info</code></a> runs inside <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/cuda_graph_runner.py#L515" class="sym-link" target="_blank" rel="noopener noreferrer"><code>CudaGraphRunner.__init__</code></a>.

<div class="callout warn" markdown="1">

#### Subtle: init order for LoRA matters

**LoRA manager is built *before* the KV pool.** Why? Because the MoE LoRA intermediate buffers (e.g. a big `[cuda_graph_max_bs × moe_intermediate × num_experts]` activation scratch space) need to be committed before memory profiling decides how much room is left for KV. If you inverted the order, KV would grab everything and LoRA would OOM on first captured batch. You can see this in the comment on `_init_lora_cuda_graph_moe_buffers` at <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L1853" target="_blank" rel="noopener noreferrer">model_runner.py:1853</a>.

</div>

### 5.2 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/model_config.py#L96" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelConfig</code></a> and HF's `AutoConfig`
{: #runner-config }

<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/model_config.py#L96" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelConfig</code></a> is SGLang's wrapper around the HF config. It does two things: call HF's `AutoConfig.from_pretrained` to load the on-disk `config.json`, then compute derived fields (dtype, head_dim, context length, sliding window, attention chunk size, etc.).

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> python/sglang/srt/configs/model_config.py:96 — ModelConfig.__init__ (excerpt) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/model_config.py#L96" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
class ModelConfig:
    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = True,
        # ... many kwargs ...
    ) -> None:
        self.model_path = model_path
        # ...

        # Get hf config
        self._maybe_pull_model_for_runai(self.model_path)
        self._maybe_pull_model_tokenizer_from_remote()
        self.model_override_args = json.loads(model_override_args)
        kwargs = {}
        if override_config_file and override_config_file.strip():
            kwargs["_configuration_file"] = override_config_file.strip()
        self.hf_config = get_config(
            self.model_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            model_override_args=self.model_override_args,
            **kwargs,
        )
        self.hf_text_config = get_hf_text_config(self.hf_config)
        # ...

        self.num_attention_heads = self.hf_text_config.num_attention_heads
        self.num_key_value_heads = getattr(
            self.hf_text_config, "num_key_value_heads", None
        )
        # ...
        self.num_hidden_layers = self.hf_text_config.num_hidden_layers
```

The bridge to HF is `get_config`, which calls `AutoConfig.from_pretrained`:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> python/sglang/srt/utils/hf_transformers/config.py:52 — get_config <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/hf_transformers/config.py#L52" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def get_config(
    model: str, trust_remote_code: bool,
    revision: Optional[str] = None, model_override_args: Optional[dict] = None, **kwargs,
):
    is_gguf = check_gguf_file(model)
    if is_gguf:
        _ensure_gguf_version()
        kwargs["gguf_file"] = model
        model = Path(model).parent
    # ... runai / remote URL branches ...

    if is_mistral_model(model):
        config = load_mistral_config(model, trust_remote_code=trust_remote_code, revision=revision)
    else:
        try:
            config = AutoConfig.from_pretrained(
                model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
            )
        except (ValueError, KeyError) as e:
            # ... Phi4MM, DeepSeek-V3.2, Longcat special cases ...
```

`AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")` reads the `config.json` from disk (or downloads it), looks at `model_type = "qwen3_moe"`, and instantiates the right config class, which is `Qwen3MoeConfig` — the one we saw in §1.1.

Here's `Qwen3MoeConfig` again, highlighting the base_model_tp_plan that ships with the config:

<div class="code-head" markdown="span">
<span class="badge badge-hf">HF</span> src/transformers/models/qwen3_moe/configuration_qwen3_moe.py:53-70 <a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/configuration_qwen3_moe.py#L53" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
# Default tensor parallel plan for base model `Qwen3Moe`
base_model_tp_plan = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
    "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
    "layers.*.mlp.experts.down_proj": "rowwise",
    "layers.*.mlp.experts": "moe_tp_experts",
    "layers.*.mlp.gate_proj": "colwise",
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
}
```

This dict is HF's TP annotation. SGLang does **not** use this directly — it has its own layer classes (<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L866" class="sym-link" target="_blank" rel="noopener noreferrer"><code>QKVParallelLinear</code></a>, <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L1312" class="sym-link" target="_blank" rel="noopener noreferrer"><code>RowParallelLinear</code></a>, <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L138" class="sym-link" target="_blank" rel="noopener noreferrer"><code>FusedMoE</code></a>) that know how to shard. But the HF plan tells you the intent: attention projections split along output, o_proj splits along input, MoE experts gate_up are packed column-parallel, down is row-parallel. SGLang's sharding will match this, just with different layer classes.

### 5.3 Model registry and `EntryClass` discovery
{: #runner-registry }

After <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/model_config.py#L96" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelConfig</code></a> has loaded `config.json` and extracted `architectures = ["Qwen3MoeForCausalLM"]`, the model loader needs to turn that string into a Python class. SGLang maintains a process-global `ModelRegistry` keyed by architecture name.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> python/sglang/srt/models/registry.py:78-132 (excerpt) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/registry.py#L78" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def resolve_model_cls(
    self,
    architectures: Union[str, List[str]],
) -> Tuple[Type[nn.Module], str]:
    architectures = self._normalize_archs(architectures)
    for arch in architectures:
        model_cls = self._try_load_model_cls(arch)
        if model_cls is not None:
            return (model_cls, arch)
    return self._raise_for_unsupported(architectures)

# ...

@lru_cache()
def import_model_classes(package_name: str, strict: bool = False):
    model_arch_name_to_cls = {}
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            if name.split(".")[-1] in envs.SGLANG_DISABLED_MODEL_ARCHS.get():
                continue
            try:
                module = importlib.import_module(name)
            except Exception as e:
                if strict: raise
                logger.warning(f"Ignore import error when loading {name}: {e}")
                continue
            if hasattr(module, "EntryClass"):
                entry = module.EntryClass
                if isinstance(entry, list):
                    for tmp in entry:
                        model_arch_name_to_cls[tmp.__name__] = tmp
                else:
                    model_arch_name_to_cls[entry.__name__] = entry
    return model_arch_name_to_cls

ModelRegistry = _ModelRegistry()
ModelRegistry.register("sglang.srt.models")
```

And the registration convention inside each model file is a single line at the bottom:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> python/sglang/srt/models/qwen3_moe.py:1223 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_moe.py#L1223" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
EntryClass = Qwen3MoeForCausalLM
```

So the end-to-end resolution is:

1. HF's `AutoConfig` reads `config.json` → `architectures = ["Qwen3MoeForCausalLM"]`.
2. SGLang's `ModelRegistry` scans `sglang.srt.models`, finds `qwen3_moe.py`, sees `EntryClass = Qwen3MoeForCausalLM`, registers `"Qwen3MoeForCausalLM" → Qwen3MoeForCausalLM`.
3. `get_model_architecture(model_config)` in the loader returns this class.
4. `_initialize_model(...)` instantiates it: `Qwen3MoeForCausalLM(config=hf_config, quant_config=None)`.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> model_loader/loader.py:261-281 — _initialize_model <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py#L261" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def _initialize_model(
    model_config: ModelConfig,
    load_config: LoadConfig,
    quant_config: Optional[QuantizationConfig] = None,
) -> nn.Module:
    """Initialize a model with the given configurations."""
    model_class, _ = get_model_architecture(model_config)
    kwargs = {
        "config": model_config.hf_config,
        "quant_config": quant_config,
    }

    if envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set():
        kwargs["sparse_head"] = envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.get()
        kwargs["model_path"] = model_config.model_path

    if load_config.draft_model_idx is not None:
        kwargs["draft_model_idx"] = load_config.draft_model_idx

    return model_class(**kwargs)
```

After this call, `model` is a <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_moe.py#L933" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Qwen3MoeForCausalLM</code></a> *with all submodules allocated on GPU but not yet filled with weights*. The `with target_device:` context-manager in the caller causes PyTorch to allocate new parameters straight on the target GPU, so peak memory is already committed.

### 5.4 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py#L306" class="sym-link" target="_blank" rel="noopener noreferrer"><code>DefaultModelLoader</code></a> — safetensors → RAM → GPU
{: #runner-loader }

Walking through `load_model` from the outside in. First, <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L1167" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelRunner.load_model</code></a>:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> model_runner.py:1167-1270 (excerpt) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L1167" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def load_model(self):
    tic_total = time.perf_counter()
    before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
    logger.info(f"Load weight begin. avail mem={before_avail_memory:.2f} GB")

    if self.device != "cpu":
        torch.set_num_threads(1)
    if self.device == "cuda":
        if torch.cuda.get_device_capability()[0] < 8:
            logger.info("Compute capability below sm80. Use float16 due to lack of bfloat16 support.")
            self.server_args.dtype = "float16"
            self.model_config.dtype = torch.float16
            # ...
    set_cuda_arch()

    # Prepare the model config, modelopt config, load_config ...
    self.load_config = LoadConfig(
        load_format=self.server_args.load_format,
        download_dir=self.server_args.download_dir,
        model_loader_extra_config=self.server_args.model_loader_extra_config,
        tp_rank=self.tp_rank,
        # ... many modelexpress fields ...
    )

    # Load the model
    monkey_patch_vllm_parallel_state()

    with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_WEIGHTS, enable_cpu_backup=...):
        self.loader = get_model_loader(
            load_config=self.load_config,
            model_config=self.model_config,
        )
        self.model = self.loader.load_model(
            model_config=self.model_config,
            device_config=DeviceConfig(self.device, self.gpu_id),
        )
    monkey_patch_vllm_parallel_state(reverse=True)
```

`get_model_loader` picks a loader flavor (<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py#L1269" class="sym-link" target="_blank" rel="noopener noreferrer"><code>DummyModelLoader</code></a>, <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py#L2637" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelOptModelLoader</code></a>, <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py#L1506" class="sym-link" target="_blank" rel="noopener noreferrer"><code>BitsAndBytesModelLoader</code></a>, <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py#L1984" class="sym-link" target="_blank" rel="noopener noreferrer"><code>GGUFModelLoader</code></a>, <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py#L722" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LayeredModelLoader</code></a>, etc.). For a standard bf16 load, we get <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py#L306" class="sym-link" target="_blank" rel="noopener noreferrer"><code>DefaultModelLoader</code></a>.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> model_loader/loader.py:675-704 — DefaultModelLoader.load_model <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py#L675" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def load_model(
    self, *, model_config: ModelConfig, device_config: DeviceConfig,
) -> nn.Module:
    # ... modelopt fast path ...
    target_device = torch.device(device_config.device)
    quant_config = _get_quantization_config(model_config, self.load_config)
    with set_default_torch_dtype(model_config.dtype):
        with target_device:
            model = _initialize_model(model_config, self.load_config, quant_config)
        self.load_weights_and_postprocess(
            model, self._get_all_weights(model_config, model), target_device
        )
    self.counter_after_loading_weights = time.perf_counter()
    return model.eval()

@staticmethod
def load_weights_and_postprocess(model, weights, target_device):
    model.load_weights(weights)
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)
```

`_get_all_weights(model_config, model)` builds the generator of `(name, tensor)` pairs that `model.load_weights` will consume. Its internals are in `_get_weights_iterator`:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> model_loader/loader.py:480-554 — _get_weights_iterator (excerpt) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py#L480" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def _get_weights_iterator(
    self, source: "Source",
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Get an iterator for the model weights based on the load format."""
    extra_config = self.load_config.model_loader_extra_config
    use_multithread = extra_config.get("enable_multithread_load", True)
    hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
        source.model_or_path, source.revision, source.fall_back_to_pt
    )

    if use_safetensors and source.model_config is not None:
        hf_weights_files = maybe_add_mtp_safetensors(
            hf_weights_files, hf_folder,
            "model.safetensors.index.json", source.model_config.hf_config,
        )

    if self.load_config.load_format == LoadFormat.NPCACHE:
        # ... skip ...
    elif use_safetensors:
        server_args = get_global_server_args()
        weight_loader_disable_mmap = server_args.weight_loader_disable_mmap
        weight_loader_prefetch = server_args.weight_loader_prefetch_checkpoints
        prefetch_num_threads = server_args.weight_loader_prefetch_num_threads

        if self.load_config.load_format == LoadFormat.FASTSAFETENSORS:
            weights_iterator = fastsafetensors_weights_iterator(hf_weights_files)
        elif use_multithread:
            weights_iterator = buffered_multi_thread_safetensors_weights_iterator(
                hf_weights_files,
                max_workers=extra_config.get("num_threads", self.DEFAULT_NUM_THREADS),
                disable_mmap=weight_loader_disable_mmap,
                prefetch=weight_loader_prefetch,
                prefetch_num_threads=prefetch_num_threads,
            )
        else:
            weights_iterator = safetensors_weights_iterator(
                hf_weights_files, disable_mmap=weight_loader_disable_mmap,
                prefetch=weight_loader_prefetch, prefetch_num_threads=prefetch_num_threads,
            )
    # ... pt iterator fallback ...

    # Apply the prefix.
    return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)
```

Step into the single-threaded safetensors iterator to see the core bit:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> model_loader/weight_utils.py:819-850 — safetensors_weights_iterator <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/weight_utils.py#L819" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def safetensors_weights_iterator(
    hf_weights_files: List[str],
    disable_mmap: bool = False,
    prefetch: bool = False,
    prefetch_num_threads: int = 4,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files."""
    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )
    sorted_files = sorted(hf_weights_files)
    if prefetch and not disable_mmap:
        _prefetch_all_checkpoints(sorted_files, num_threads=prefetch_num_threads)
    for st_file in tqdm(sorted_files, desc="Loading safetensors checkpoint shards", ...):
        if disable_mmap:
            with open(st_file, "rb") as f:
                result = safetensors.torch.load(f.read())
                for name in sorted(result.keys()):
                    yield name, result[name]
        else:
            with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)
```

This is a simple `for shard in shards: for tensor in shard: yield (name, tensor)` generator. The `safetensors.safe_open(..., device="cpu")` call mmaps the file, so the actual bytes aren't read until `f.get_tensor(name)` pulls them. The returned tensor is a CPU tensor pointing into the mmap — until it's copied by `weight_loader(...)` onto the GPU.

The `_prepare_weights` method (called at the top of `_get_weights_iterator`) is what finds the shard files and filters out the wrong ones:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> model_loader/loader.py:385-479 — _prepare_weights (excerpt) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py#L385" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def _prepare_weights(
    self, model_name_or_path: str, revision: Optional[str], fall_back_to_pt: bool
) -> Tuple[str, List[str], bool]:
    """Prepare weights for the model. If the model is not local, it will be downloaded."""
    model_name_or_path = self._maybe_download_from_modelscope(model_name_or_path, revision)

    is_local = os.path.isdir(model_name_or_path)
    load_format = self.load_config.load_format
    use_safetensors = False
    index_file = SAFE_WEIGHTS_INDEX_NAME
    if load_format == LoadFormat.AUTO:
        allow_patterns = ["*.safetensors", "*.bin"]
    elif load_format == LoadFormat.SAFETENSORS or load_format == LoadFormat.FASTSAFETENSORS:
        use_safetensors = True
        allow_patterns = ["*.safetensors"]
    # ... other formats ...

    if fall_back_to_pt:
        allow_patterns += ["*.pt"]

    if not is_local:
        hf_folder = download_weights_from_hf(model_name_or_path, self.load_config.download_dir,
                                              allow_patterns, revision, ignore_patterns=...)
    else:
        hf_folder = model_name_or_path

    hf_weights_files: List[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
        if len(hf_weights_files) > 0:
            if pattern == "*.safetensors":
                use_safetensors = True
            break

    if use_safetensors:
        # For models with both sharded + consolidated, deduplicate using the index.
        if not is_local:
            download_safetensors_index_file_from_hf(model_name_or_path, index_file, ...)
        hf_weights_files = filter_duplicate_safetensors_files(
            hf_weights_files, hf_folder, index_file
        )
    else:
        hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)
    # ...
    return hf_folder, hf_weights_files, use_safetensors
```

For our run: the folder has 16 safetensors shards, `allow_patterns = ["*.safetensors", "*.bin"]`, glob finds the 16 shards, `filter_duplicate_safetensors_files` cross-checks with the index (there's no consolidated copy, so nothing is filtered), and the loop completes.

---

### 5.5 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_moe.py#L1099" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Qwen3MoeForCausalLM.load_weights</code></a> — the name-remap dance
{: #runner-load-weights }

Now the critical question: HF emits tensor names like `model.layers.0.self_attn.q_proj.weight` and `model.layers.0.mlp.experts.0.gate_proj.weight`, but SGLang's layers are named `model.layers.0.self_attn.qkv_proj.weight` (one tensor for q+k+v) and `model.layers.0.mlp.experts.w13_weight` (3D stack of all 128 experts' gate_up). Who does the remapping, and how?

The answer: <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_moe.py#L1099" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Qwen3MoeForCausalLM.load_weights</code></a> itself. Every model in `sglang.srt.models` implements a <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_moe.py#L1099" class="sym-link" target="_blank" rel="noopener noreferrer"><code>load_weights</code></a> method that declares the HF-side → SGLang-side mapping and routes each incoming tensor to the right parameter's `weight_loader` hook.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> python/sglang/srt/models/qwen3_moe.py:1099 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_moe.py#L1099" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    expert_params_mapping = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=self.config.num_experts,
    )

    # Pre-define `params_dict` to avoid repeated expensive traversal of model parameters.
    params_dict = dict(self.named_parameters())

    for name, loaded_weight in weights:
        layer_id = get_layer_id(name)
        if (
            layer_id is not None
            and hasattr(self.model, "start_layer")
            and (layer_id < self.model.start_layer or layer_id >= self.model.end_layer)
        ):
            continue  # Layer not on this pipeline-parallel rank

        if "rotary_emb.inv_freq" in name:
            continue  # Skip freq buffers; SGLang computes these fresh

        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            if "mlp.experts" in name:
                continue  # Experts handled by the expert_params_mapping loop below.
            name = name.replace(weight_name, param_name)
            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            is_expert_weight = False
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                is_expert_weight = True
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue  # Expert not on this EP rank; skip
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(
                    param, loaded_weight, name,
                    shard_id=shard_id, expert_id=expert_id,
                )
                break
            else:
                if is_expert_weight:
                    continue  # Expert weight not mapped here, skip rest
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")
    # ... routed_experts_weights_of_layer memoization at end ...
```

<div class="callout info" markdown="1">

#### What is `self.named_parameters()` here?

`named_parameters()` is a standard PyTorch method on `nn.Module`. It walks the entire submodule tree and yields `(name: str, param: torch.nn.Parameter)` tuples, where `name` is the dotted path reflecting how submodules are nested (`model.layers.0.self_attn.qkv_proj.weight`, etc.). `dict(self.named_parameters())` materializes this iterator into a dict for O(1) lookups — the comment in the source calls out that doing it once up front avoids N tree walks inside the per-weight loop.

The critical thing to notice is that the keys of this dict correspond to **SGLang's parameter layout after TP sharding and MoE fusion**, not the HuggingFace checkpoint layout. That's the reason <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_moe.py#L1099" class="sym-link" target="_blank" rel="noopener noreferrer"><code>load_weights</code></a> has to do a name-remap dance at all. Compare:

| SGLang key (in `params_dict`)                 | Corresponding HF checkpoint keys                                                                                  |
|-----------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `model.layers.0.self_attn.qkv_proj.weight`    | `q_proj.weight` + `k_proj.weight` + `v_proj.weight` (three tensors fused into one)                                |
| `model.layers.0.mlp.experts.w13_weight`       | 128 × `experts.{j}.gate_proj.weight` + 128 × `experts.{j}.up_proj.weight` (256 tensors fused into one 3-D tensor) |
| `model.layers.0.mlp.experts.w2_weight`        | 128 × `experts.{j}.down_proj.weight` (128 tensors fused into one 3-D tensor)                                      |
| `model.layers.0.mlp.gate.weight`              | `mlp.gate.weight` (router; not fused; replicated, not sharded)                                                    |
| `model.layers.0.self_attn.q_norm.weight`      | same (RMSNorm; no remap needed)                                                                                   |
| `model.embed_tokens.weight`, `lm_head.weight` | same                                                                                                              |

For the full Qwen3-30B-A3B-Instruct-2507, `params_dict` has roughly 1,000 entries: 2 embedding/lm-head + 48 layers × (6 attention-side + 3 MoE-side + 2 norm) + 1 final norm, plus a few scalar bias tensors from quantized paths that aren't relevant here. All of them are **already TP-sharded** for this rank — `qkv_proj.weight` on rank 2 of 4 has shape `(q_shard + k_shard + v_shard, 2048)` where each shard is 1/4 of the global size.

</div>

<div class="callout info" markdown="1">

#### "Each worker only has params on its rank" — yes, exactly

You're right. The `nn.Module` on this GPU is constructed with TP-sharded shapes from the start — see <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L310" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ColumnParallelLinear.__init__</code></a> (§5.6) where `output_size_per_partition = output_size / tp_size`. So `params_dict` only holds the 1/`tp_size` fraction of each column-parallel weight this rank is responsible for.

The skipping happens at two levels during <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_moe.py#L1099" class="sym-link" target="_blank" rel="noopener noreferrer"><code>load_weights</code></a>:

1. **For TP-sharded weights** (`qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`, etc.): the full HF tensor reaches the worker, but the per-parameter `weight_loader` calls `loaded_weight.narrow(output_dim, tp_rank*shard_size, shard_size)` to copy only this rank's slice into the destination (`linear.py:564-566` for column-parallel, `linear.py:1115-1117` for row-parallel). The other ranks' slices never land on GPU.
2. **For EP-sharded experts**: if the expert_id doesn't belong to this EP rank, <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L573" class="sym-link" target="_blank" rel="noopener noreferrer"><code>_map_global_expert_id_to_local_expert_id</code></a> returns `-1` (`fused_moe_triton/layer.py:581`), and the weight_loader short-circuits on `if expert_id == -1: return` (line 612). That expert's tensor is dropped entirely — never copied to GPU at all.

The `# Expert not on this EP rank; skip` comment you saw is a docstring of the second case. For your `--tp 4 --ep-size 1` run there's no EP, so every rank holds all 128 experts (each TP-sharded internally). If you added `--ep-size 4`, each rank would hold only 32 experts, and the EP-skip branch would fire for 3/4 of the incoming expert tensors on each worker.

Same pattern for PP: the layer range check at the top of the loop (`if layer_id < self.model.start_layer or layer_id >= self.model.end_layer: continue`) skips entire layers that aren't on this pipeline-parallel rank, so their weights never reach the `weight_loader` at all.

</div>

Reading the logic top to bottom:

1. **`stacked_params_mapping`** is a list of `(param_name, shard_name, shard_id)`. Each line says: "if a checkpoint tensor's name contains `shard_name`, then it's actually a shard of a fused parameter `param_name`, identified by `shard_id`." For Qwen3MoE: `q_proj → qkv_proj[shard="q"]`, `k_proj → qkv_proj[shard="k"]`, `v_proj → qkv_proj[shard="v"]`, `gate_proj → gate_up_proj[shard=0]`, `up_proj → gate_up_proj[shard=1]`.
2. **`expert_params_mapping`** is built by <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L1050" class="sym-link" target="_blank" rel="noopener noreferrer"><code>FusedMoE.make_expert_params_mapping</code></a> (see below). It has one tuple per `(expert_id, ckpt_shard)` — 3·128 = 384 tuples.
3. **Main loop**: for each incoming `(name, tensor)`:
    - If the layer is outside this PP rank's range, skip.
    - Skip `rotary_emb.inv_freq` (a HF buffer; SGLang rebuilds RoPE freqs itself).
    - Try the qkv / gate_up `stacked_params_mapping` first. If a match is found and it's *not* an expert tensor, rewrite the name (`q_proj→qkv_proj`), look up the param, call `param.weight_loader(param, loaded_weight, shard_id)`. The weight_loader knows how to narrow the loaded_weight into the right offset/size of the fused param.
    - Otherwise, try the `expert_params_mapping`. Match rewrites e.g. `experts.0.gate_proj → experts.w13_`, and the weight_loader gets called with `(param, tensor, name, shard_id="w1", expert_id=0)`.
    - Otherwise, fall through to <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/weight_utils.py#L1137" class="sym-link" target="_blank" rel="noopener noreferrer"><code>default_weight_loader</code></a> — a plain size-check + copy. This handles all norms, embeddings, the gate/router, and any other 1:1 parameter.

Putting concrete names on this flow, here's what happens as the iterator yields weights for layer 0:

| HF tensor name                                   | Rewritten to                               | Path taken                                           |
|--------------------------------------------------|--------------------------------------------|------------------------------------------------------|
| `model.embed_tokens.weight`                      | (same)                                     | default_weight_loader                                |
| `model.layers.0.input_layernorm.weight`          | (same)                                     | default_weight_loader                                |
| `model.layers.0.self_attn.q_proj.weight`         | `model.layers.0.self_attn.qkv_proj.weight` | QKVParallelLinear.weight_loader(shard_id="q")        |
| `model.layers.0.self_attn.k_proj.weight`         | `model.layers.0.self_attn.qkv_proj.weight` | QKVParallelLinear.weight_loader(shard_id="k")        |
| `model.layers.0.self_attn.v_proj.weight`         | `model.layers.0.self_attn.qkv_proj.weight` | QKVParallelLinear.weight_loader(shard_id="v")        |
| `model.layers.0.self_attn.o_proj.weight`         | (same)                                     | RowParallelLinear.weight_loader                      |
| `model.layers.0.self_attn.q_norm.weight`         | (same)                                     | default_weight_loader                                |
| `model.layers.0.self_attn.k_norm.weight`         | (same)                                     | default_weight_loader                                |
| `model.layers.0.post_attention_layernorm.weight` | (same)                                     | default_weight_loader                                |
| `model.layers.0.mlp.gate.weight`                 | (same)                                     | ReplicatedLinear (gate=router) default_weight_loader |
| `model.layers.0.mlp.experts.0.gate_proj.weight`  | `...experts.w13_weight`                    | FusedMoE.weight_loader(expert_id=0, shard_id="w1")   |
| `model.layers.0.mlp.experts.0.up_proj.weight`    | `...experts.w13_weight`                    | FusedMoE.weight_loader(expert_id=0, shard_id="w3")   |
| `model.layers.0.mlp.experts.0.down_proj.weight`  | `...experts.w2_weight`                     | FusedMoE.weight_loader(expert_id=0, shard_id="w2")   |
| ... (127 more experts × 3 tensors each) ...      |                                            |                                                      |

<div class="callout info" markdown="1">

#### Why this pattern?

The *stacked_params_mapping* approach is adapted from vLLM — see the "Adapted from vllm" comment at the top of <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/registry.py" target="_blank" rel="noopener noreferrer">models/registry.py</a>. It lets every model file declare its own fusion pattern in a tiny list, and reuse the generic <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L1095" class="sym-link" target="_blank" rel="noopener noreferrer"><code>QKVParallelLinear.weight_loader</code></a> / <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L583" class="sym-link" target="_blank" rel="noopener noreferrer"><code>FusedMoE.weight_loader</code></a> machinery without having to write a custom loader per model. Llama-family models use the exact same pattern: q_proj/k_proj/v_proj/gate_proj/up_proj, different `num_experts` or absent in dense models.

</div>

### 5.6 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L866" class="sym-link" target="_blank" rel="noopener noreferrer"><code>QKVParallelLinear</code></a> — why q/k/v become one tensor
{: #runner-qkv }

SGLang doesn't allocate three separate `nn.Linear` layers for Q, K, V. It allocates one <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L866" class="sym-link" target="_blank" rel="noopener noreferrer"><code>QKVParallelLinear</code></a> whose single weight matrix is `[q_size + k_size + v_size, hidden]`, concatenated along the output dim. Three reasons:

1. **One GEMM instead of three.** A larger GEMM typically hits higher arithmetic intensity on modern GPUs than three back-to-back smaller GEMMs.
2. **One allreduce instead of three.** In TP, each rank produces `q + k + v` partition locally; we can do one allgather at the end (if needed) rather than three.
3. **One weight_loader.** The loader knows the "q, k, v" offsets once, at construction, and the loop above can just hand it each HF shard with a letter tag.

Here's the constructor, trimmed:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> python/sglang/srt/layers/linear.py:866-955 — QKVParallelLinear.__init__ <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L866" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
class QKVParallelLinear(ColumnParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = True,
        # ... quant / presharded options ...
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads

        if tp_rank is None:  tp_rank = get_tensor_model_parallel_rank()
        if tp_size is None:  tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        self.q_proj_shard_size = self.num_heads * self.head_size
        self.kv_proj_shard_size = self.num_kv_heads * self.head_size
        self.v_proj_shard_size = self.num_kv_heads * self.v_head_size

        input_size = self.hidden_size
        output_size = (
            self.num_heads * self.head_size
            + self.num_kv_heads * self.head_size
            + self.num_kv_heads * self.v_head_size
        ) * tp_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,        # q_proj (global)
            self.num_kv_heads * self.head_size * tp_size,     # k_proj (global)
            self.num_kv_heads * self.v_head_size * tp_size,   # v_proj (global)
        ]
        # super().__init__ allocates .weight of shape (output_size / tp_size, input_size)
```

For Qwen3-30B-A3B at TP=4:

| Quantity                              | Computation                          | Value              |
|---------------------------------------|--------------------------------------|--------------------|
| `total_num_heads`                     | config.num_attention_heads           | 32                 |
| `total_num_kv_heads`                  | config.num_key_value_heads           | 4                  |
| `head_size`                           | config.head_dim                      | 128                |
| `tp_size ≥ total_num_kv_heads`?       | 4 ≥ 4                                | yes                |
| `num_kv_heads` (per rank)             | 1                                    | 1                  |
| `num_kv_head_replicas`                | tp_size / total_num_kv_heads = 4 / 4 | 1                  |
| `num_heads` (per rank)                | 32 / 4                               | 8                  |
| `q_proj_shard_size`                   | 8 × 128                              | 1024               |
| `kv_proj_shard_size`                  | 1 × 128                              | 128                |
| `output_sizes` (global, not per rank) | \[32·128, 4·128, 4·128\]             | \[4096, 512, 512\] |
| `weight.shape` per rank               | \[(1024 + 128 + 128), 2048\]         | \[1280, 2048\]     |

And here's the weight_loader code:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> linear.py:538-713 — QKVParallelLinear.weight_loader (excerpt) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L538" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def weight_loader(
    self,
    param: Parameter,
    loaded_weight: torch.Tensor,
    loaded_shard_id: tuple[int, ...] | int | None = None,
):
    if isinstance(loaded_shard_id, tuple):
        # weight_loader_v2 for parameter-objects with structured metadata
        return self.weight_loader_v2(param, loaded_weight, loaded_shard_id)

    # ... GGUF special cases ...

    param_data = param.data
    output_dim = getattr(param, "output_dim", None)

    if loaded_shard_id is None:
        # Checkpoint already has a fused qkv tensor (rare, e.g. Phi-3)
        # ... fuse-and-recurse path ...
        return

    assert loaded_shard_id < len(self.output_sizes)
    if output_dim is not None:
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        # ... quant adjustments ...

        param_data = param_data.narrow(output_dim, shard_offset, shard_size)
        start_idx = self.tp_rank * shard_size

        # ... cpu-padding path ...

        if not use_bitsandbytes_4bit and not self.use_presharded_weights:
            # Narrow the loaded_weight to this rank's slice
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
    # ... scalar broadcast cases ...

    assert param_data.shape == loaded_weight.shape
    param_data.copy_(loaded_weight)
```

**Important detail:** the `output_sizes` list is in *global* dims (not per-rank). So `shard_offset = sum(output_sizes[:id]) // tp_size` converts to per-rank offset. For TP=4 with our numbers:

| shard_id | shard_offset (per-rank) | shard_size (per-rank) | global slice    |
|----------|-------------------------|-----------------------|-----------------|
| "q" = 0  | 0                       | 1024                  | rows 0..1023    |
| "k" = 1  | 1024                    | 128                   | rows 1024..1151 |
| "v" = 2  | 1152                    | 128                   | rows 1152..1279 |

Then `start_idx = tp_rank * shard_size` picks which *rank's* slice of the HF tensor to use. For tp_rank=2, the Q slice is `loaded_weight[2048:3072, :]` (heads 16..23 of the full 32), copied into `qkv_proj.weight[0:1024, :]`. Three `.copy_` calls later, this rank's `qkv_proj.weight` is fully populated with its slice of Q + K + V concatenated along dim 0.

<div class="callout motiv" markdown="1">

#### Per-rank KV head replication (GQA + small #KV)

Qwen3 has 4 KV heads but we're running TP=4. The code path `if tp_size >= self.total_num_kv_heads: self.num_kv_heads = 1; self.num_kv_head_replicas = tp_size / total_num_kv_heads` means each rank gets exactly 1 KV head (perfect split). If we bumped to `--tp 8`, `tp_size > total_num_kv_heads` — then `num_kv_heads_per_rank = 1` but `num_kv_head_replicas = 2`: 8 ranks share the 4 KV heads, with each head replicated on 2 ranks. This "replicate KV but shard Q" is the classic GQA parallelism described in the S-LoRA paper (Sec. 4.3 of <a href="https://arxiv.org/abs/2311.03285" target="_blank" rel="noopener noreferrer">arXiv:2311.03285</a>).

</div>

### 5.7 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L138" class="sym-link" target="_blank" rel="noopener noreferrer"><code>FusedMoE</code></a> — how 128 experts become one tensor
{: #runner-moe }

The story here is more dramatic. On disk, each of the 128 experts per layer is stored as three separate tensors (`experts.i.gate_proj.weight`, `experts.i.up_proj.weight`, `experts.i.down_proj.weight`). On GPU, SGLang fuses these into exactly two parameters per layer: `w13_weight` (stacked gate+up for all experts) and `w2_weight` (stacked down for all experts), each 3D.

The mapping from checkpoint names to (param, expert_id, shard_id) is:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> layers/moe/fused_moe_triton/layer.py:1050-1075 — make_expert_params_mapping <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L1050" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def make_expert_params_mapping(
    cls,
    ckpt_gate_proj_name: str,
    ckpt_down_proj_name: str,
    ckpt_up_proj_name: str,
    num_experts: int,
) -> List[Tuple[str, str, int, str]]:
    return [
        # (param_name, weight_name, expert_id, shard_id)
        (
            ("experts.w13_"
             if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
             else "experts.w2_"),
            f"experts.{expert_id}.{weight_name}.",
            expert_id,
            shard_id,
        )
        for expert_id in range(num_experts)
        for shard_id, weight_name in [
            ("w1", ckpt_gate_proj_name),
            ("w2", ckpt_down_proj_name),
            ("w3", ckpt_up_proj_name),
        ]
    ]
```

For Qwen3 (gate/down/up = "gate_proj"/"down_proj"/"up_proj", num_experts=128), this returns 3·128 = 384 tuples. Example tuples:

```python
("experts.w13_", "experts.0.gate_proj.", 0, "w1"),
("experts.w2_",  "experts.0.down_proj.", 0, "w2"),
("experts.w13_", "experts.0.up_proj.",   0, "w3"),
("experts.w13_", "experts.1.gate_proj.", 1, "w1"),
...
("experts.w13_", "experts.127.up_proj.", 127, "w3"),
```

Note that `"w1"` and `"w3"` both map to the same `w13_` parameter — they're two halves of the gated MLP, stacked in dim 1 of the 3D tensor. `"w2"` is the down projection (separate tensor).

#### FusedMoE weight allocation (UnquantizedFusedMoEMethod)

The actual parameter shapes are allocated by the quant method's `create_weights`:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> layers/quantization/unquant.py:163-235 — UnquantizedFusedMoEMethod.create_weights <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/unquant.py#L163" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, MultiPlatformOp):
    """MoE method without quantization."""
    # ...

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        self.with_bias = with_bias

        # Fused gate_up_proj (column parallel)
        w13_up_dim = (
            2 * intermediate_size_per_partition
            if layer.moe_runner_config.is_gated
            else intermediate_size_per_partition
        )
        w13_weight_n, w13_weight_k = (w13_up_dim, hidden_size)
        if self.use_triton_kernels:
            w13_weight_n, w13_weight_k = w13_weight_k, w13_weight_n
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, w13_weight_n, w13_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        # ... optional with_bias branch ...

        # down_proj (row parallel)
        w2_weight_n, w2_weight_k = (
            hidden_size,
            intermediate_size_per_partition,
        )
        if self.use_triton_kernels:
            w2_weight_n, w2_weight_k = w2_weight_k, w2_weight_n
        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, w2_weight_n, w2_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
```

For Qwen3-30B-A3B at TP=4 (no EP, so `num_local_experts = 128`):

| Parameter            | Shape                                       | Elements    | Bytes (bf16) |
|----------------------|---------------------------------------------|-------------|--------------|
| `experts.w13_weight` | (128, 2×768/4=384, 2048) = (128, 384, 2048) | 100 663 296 | 192 MB       |
| `experts.w2_weight`  | (128, 2048, 768/4=192) = (128, 2048, 192)   | 50 331 648  | 96 MB        |

So per layer, MoE takes up **~288 MB per rank**. Over 48 layers: **~13.8 GB per rank**.

#### FusedMoE weight_loader: `_load_w13` and `_load_w2`

When the model loader hands an expert tensor to <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L583" class="sym-link" target="_blank" rel="noopener noreferrer"><code>FusedMoE.weight_loader</code></a>, it dispatches by shard_id:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> layers/moe/fused_moe_triton/layer.py:415-477 — _load_w13 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L415" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def _load_w13(
    self,
    expert_data: torch.Tensor,
    shard_dim: int,
    shard_id: str,
    loaded_weight: torch.Tensor,
    tp_rank: int,
    is_bias: bool = False,
):
    # Index the loaded weight for tp sharding.
    # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
    assert shard_id in {"w1", "w3", "w13"}

    if is_bias:
        shard_dim = -1

    if shard_id in {"w1", "w3"} and self.moe_runner_config.is_gated:
        # non-fused version: w1 and w3 each take half of the fused dim
        shard_size = expert_data.shape[shard_dim] // 2
    elif shard_id in {"w13"} or (shard_id in {"w1", "w3"} and not self.moe_runner_config.is_gated):
        # fused version
        shard_size = expert_data.shape[shard_dim]
    else:
        raise NotImplementedError

    # w1, gate_proj: Load into first logical weight of w13.
    # w3, up_proj: Load into second logical weight of w13.
    switch_w13 = getattr(self.quant_method, "load_up_proj_weight_first", False)
    if ((switch_w13 and shard_id == "w1") or (not switch_w13 and shard_id == "w3")) \
            and self.moe_runner_config.is_gated:
        start = shard_size
    else:
        start = 0

    if self.use_padded_loading:
        expert_data, loaded_weight = narrow_padded_param_and_loaded_weight(...)
    else:
        if not self.use_presharded_weights:
            if not is_bias and self.use_triton_kernels:
                loaded_weight = loaded_weight.transpose(-2, -1)
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size
            )
        expert_data = expert_data.narrow(shard_dim, start, shard_size)
    expert_data.copy_(loaded_weight)
```

What `_load_w13` does, step by step for Qwen3 at TP=4:

1. `expert_data` is a 2D slice: `w13_weight[expert_id]`, shape `(384, 2048)`.
2. `loaded_weight` is the incoming HF tensor: `experts.{i}.gate_proj.weight` or `up_proj.weight`, shape `(768, 2048)`.
3. `shard_dim = 1` (the fused dim-0 of the 3D param is dim-1 inside this 2D view).
4. `shard_size = expert_data.shape[1] // 2 = 192` (half of the fused 384).
5. `start = 0` for w1 (gate), `start = 192` for w3 (up) — so gate goes into the first half, up into the second half.
6. `loaded_weight.narrow(1, tp_rank * 192, 192)` — take my rank's slice of the expert's gate/up column. For tp_rank=2, rows 384..575 of the original (768, 2048).
7. `expert_data.narrow(1, start, 192).copy_(loaded_weight)` — copy into the right half of the fused param.

So for expert 42 in layer 17, after both w1 and w3 have loaded, the 2D slice `w13_weight[42, :, :]` is laid out as `[ gate_partition(192) | up_partition(192) ] × [hidden(2048)]`.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> layers/moe/fused_moe_triton/layer.py:477-540 — _load_w2 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L477" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def _load_w2(
    self,
    expert_data: torch.Tensor,
    shard_dim: int,
    shard_id: str,
    loaded_weight: torch.Tensor,
    tp_rank: int,
    is_bias: bool = False,
):
    # Index the loaded weight for tp sharding.
    # down_proj: "RowParallel" so tp sharding on input_dim
    if shard_id != "w2":
        raise ValueError(f"shard_id must be 'w2', got {shard_id}")

    if is_bias:
        shard_size = expert_data.shape[-1]
    else:
        shard_size = expert_data.shape[shard_dim]

    if self.use_padded_loading:
        # ... padded path ...
    else:
        if not is_bias and not self.use_presharded_weights:
            if self.use_triton_kernels:
                loaded_weight = loaded_weight.transpose(-2, -1)
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size
            )
    expert_data.copy_(loaded_weight)
```

For `w2`: `expert_data = w2_weight[expert_id]` shape `(2048, 192)`. `loaded_weight` = HF `experts.{i}.down_proj.weight` shape `(2048, 768)`. The narrow takes `loaded_weight[:, tp_rank*192:(tp_rank+1)*192]`, then copies. Each rank thus gets the quarter of the MLP intermediate dim that its gate/up produced.

<div class="callout motiv" markdown="1">

#### Why "w1/w2/w3" and not "gate/up/down"?

Naming artifact inherited from vLLM and Megatron — the original SwiGLU paper (<a href="https://arxiv.org/abs/2002.05202" target="_blank" rel="noopener noreferrer">arXiv:2002.05202</a>) labeled its three matrices W₁, W₂, W₃ where W₁ and W₃ are the two halves of the gated projection and W₂ is the down projection. SGLang keeps this convention for compatibility with checkpoints already packed in this layout.

</div>

#### How TP and EP compose inside FusedMoE

`FusedMoE` supports two orthogonal partitioning axes, and they compose. It's easy to assume "each rank has a subset of full experts" (pure EP) or "each rank has all experts but each is sharded" (pure TP), but in general **neither is the full picture** — SGLang applies both simultaneously if both sizes are \> 1. The arithmetic is two independent divisions:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> fused_moe_triton/layer.py:197-219 — two independent world sizes [GitHub](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L197)
</div>

```python
self.moe_ep_size = get_moe_expert_parallel_world_size()
self.moe_tp_size = get_moe_tensor_parallel_world_size()
...
self._num_local_routed = self._num_global_routed // self.moe_ep_size
self.num_local_experts = self._num_local_routed + num_fused_shared_experts
...
assert intermediate_size % self.moe_tp_size == 0
self.intermediate_size_per_partition = intermediate_size // self.moe_tp_size
```

- `num_local_experts = num_experts / moe_ep_size` — how many full experts this rank owns.
- `intermediate_size_per_partition = intermediate_size / moe_tp_size` — how wide each of those experts' MLP is on this rank.

The constraint is `moe_tp_size × moe_ep_size = tp_size` (the total TP world size per PP stage). Different choices trade comms patterns for memory footprint, here's the table for Qwen3-30B-A3B-Instruct-2507 at TP=4 with different EP splits:

| Flags                           | `moe_tp_size` | `moe_ep_size` | `num_local_experts` | `intermediate_size_per_partition` | Per-rank `w13_weight` shape | Per-rank `w2_weight` shape |
|---------------------------------|---------------|---------------|---------------------|-----------------------------------|-----------------------------|----------------------------|
| `--tp 4 --ep 1` (default)       | 4             | 1             | 128                 | 192                               | `(128, 384, 2048)`          | `(128, 2048, 192)`         |
| `--tp 4 --ep 4`                 | 1             | 4             | 32                  | 768                               | `(32, 1536, 2048)`          | `(32, 2048, 768)`          |
| `--tp 8 --ep 2` (8 GPUs, mixed) | 4             | 2             | 64                  | 192                               | `(64, 384, 2048)`           | `(64, 2048, 192)`          |

The comment `# Expert not on this EP rank; skip` in the weight loader only fires when `moe_ep_size > 1`. For the default `--tp 4` Qwen3 run, that branch never triggers — every rank loads its shard of every expert.

#### How W13 and W2 get sharded under TP (`moe_tp_size > 1`)

The two fused tensors follow the standard column-parallel → row-parallel pattern, just applied once per expert inside the first tensor dimension:

- **`w13_weight` is sharded along dim 1** (the `2 * intermediate_size` axis): each rank gets `2 * intermediate_size_per_partition` rows out of the full `2 * intermediate_size`. This is the MergedColumnParallelLinear pattern — split output dim, all ranks see the full input.
- **`w2_weight` is sharded along dim 2** (the `intermediate_size` axis): each rank gets `intermediate_size_per_partition` columns out of the full `intermediate_size`. This is the RowParallelLinear pattern — split input dim, reduce output.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> fused_moe_triton/layer.py:415-475 — _load_w13 (the narrow) [GitHub](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L415)
</div>

```python
def _load_w13(
    self,
    expert_data: torch.Tensor,
    shard_dim: int,
    shard_id: str,
    loaded_weight: torch.Tensor,
    tp_rank: int,
    is_bias: bool = False,
):
    # Index the loaded weight for tp sharding.
    # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
    ...
    # Narrow parameter and load.
    # w1, gate_proj: Load into first logical weight of w13.
    # w3, up_proj:   Load into second logical weight of w13.
    ...
    loaded_weight = loaded_weight.narrow(
        shard_dim, shard_size * tp_rank, shard_size
    )
    expert_data = expert_data.narrow(shard_dim, start, shard_size)
    expert_data.copy_(loaded_weight)
```

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> fused_moe_triton/layer.py:477-540 — _load_w2 (the narrow) [GitHub](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L477)
</div>

```python
def _load_w2(self, expert_data, shard_dim, shard_id, loaded_weight, tp_rank, is_bias=False):
    # down_proj: "RowParallel" so tp sharding on input_dim
    # Narrow parameter and load.
    ...
    # this parameter is a weight matrix
    # for w2 in TP, it shards the input_features, i.e., shard_dim=2
    shard_size = expert_data.shape[shard_dim]
    ...
    loaded_weight = loaded_weight.narrow(
        shard_dim, shard_size * tp_rank, shard_size
    )
```

For the default `--tp 4 --ep 1` case, when rank 2 loads expert 37's `gate_proj` from the HF checkpoint, it narrows the 768-row tensor to rows `[384, 576)` — its 192-row slice — and writes into `w13_weight[37, 0:192, :]`. When it later loads expert 37's `up_proj`, it narrows the same 768-row range into rows `[384, 576)` and writes into `w13_weight[37, 192:384, :]`. The top half of `w13_weight[37]` is this rank's slice of w1; the bottom half is this rank's slice of w3.

No all-reduce is needed between the w13 GEMM and the w2 GEMM. Each rank computes its chunk of the intermediate activations (`2 * 192 = 384` dim), applies `silu_and_mul` locally (producing 192 dim per rank), then multiplies by its own 192-column slice of w2. Partial outputs are all-reduced at the end of the expert — the same pattern as a regular column-parallel → row-parallel MLP, just over MoE routing.

#### What the expert actually computes — SwiGLU, spelled out

A single expert's forward pass in Qwen3 is a SwiGLU MLP:

```text
expert(x) = w2 @ ( SiLU(w1 @ x) ⊙ (w3 @ x) )

where ⊙ is elementwise multiplication (Hadamard product),
and SiLU(z) = z · σ(z) = z / (1 + exp(-z))
```

Concretely for Qwen3 with TP=4, each rank does:

1. **One fused GEMM** producing the stacked `[gate; up]` output:

    ```text
    a  = w13 @ x                    # shape (2 * 192,) = (384,)
    a_gate = a[0:192]                # top half = w1 @ x for this rank's slice
    a_up   = a[192:384]              # bottom half = w3 @ x for this rank's slice
    ```

    Reading `x` from HBM once produces both halves of the intermediate activation.

2. **One fused elementwise kernel** — `silu_and_mul`, implemented in `sgl_kernel`:

    ```text
    c = SiLU(a_gate) ⊙ a_up         # shape (192,)
    ```

    One kernel pass over the 384-element input that splits, applies SiLU to the first half, multiplies by the second half, and writes 192 outputs. No intermediate HBM round-trip.

3. **One more GEMM**:

    ```text
    y_partial = w2 @ c              # shape (hidden_size,) = (2048,)
    ```

    Each rank's `y_partial` is the contribution of its intermediate slice.

4. **All-reduce across `moe_tp_group`**: `y = sum(y_partial across tp ranks)`.

You can see step 2 in `fused_marlin_moe.py:176` (the pattern is the same for every MoE backend variant):

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> fused_moe_triton/fused_marlin_moe.py:176 — silu_and_mul after w13 GEMM [GitHub](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py#L176)
</div>

```python
silu_and_mul(intermediate_cache1.view(-1, 2 * N), intermediate_cache2)
```

Here `intermediate_cache1` is the `(num_tokens_routed_to_this_expert, 2 * N)` output of the w13 GEMM, and `intermediate_cache2` is the `(num_tokens_routed_to_this_expert, N)` pre-allocated destination for the gated activation.

#### Why fuse w1 and w3 into w13 at all?

Two fundamental wins, both specific to GPU execution:

1. **One kernel launch instead of two.** Each CUDA kernel launch costs ~5-10 μs on H100/H200 regardless of problem size. For decoding, where GEMMs are skinny (batch-size rows × hidden-size cols), kernel launch is already a measurable fraction of total latency. With 48 layers and 8 routed experts per token, doing w1 and w3 as separate GEMMs would add thousands of launches per decode step. One fused w13 GEMM halves that.
2. **Input activation `x` is read from HBM once instead of twice.** At MoE decode-time batch sizes, the computation is HBM-bandwidth-bound: the arithmetic intensity of a skinny GEMM is low, so the GPU waits on memory. A fused w13 multiplies the same `x` against twice as many weight rows, emitting two halves of the intermediate activation in one pass. This is the dominant benefit in practice — for very small decode batches, the w1+w3 fusion is nearly a 2× speedup of the first GEMM alone.

A third benefit is structural rather than performance: **w1 and w3 are both column-parallel under TP and shard along the same axis** (output intermediate dim). Concatenating them along that axis preserves the partition pattern — each rank's `w13` is simply `[w1_shard; w3_shard]` stacked, same split semantics as if they were separate `MergedColumnParallelLinear` tensors.

The cost is the small `silu_and_mul` post-kernel that splits the output and applies the gated activation. But that kernel is already elementwise and was going to run anyway — it just takes its input in the fused `(2*N,)` layout and writes to the `(N,)` layout instead of reading two separate inputs.

### 5.8 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/weight_utils.py#L1137" class="sym-link" target="_blank" rel="noopener noreferrer"><code>default_weight_loader</code></a> — the boring remaining tensors
{: #runner-default-wl }

For everything that isn't qkv/gate_up/moe_experts — norms, embeddings, lm_head, the gate (router) tensor — <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/weight_utils.py#L1137" class="sym-link" target="_blank" rel="noopener noreferrer"><code>default_weight_loader</code></a> handles it:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> model_loader/weight_utils.py:1137-1158 — default_weight_loader <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/weight_utils.py#L1137" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    try:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # Sometimes scalar values aren't considered tensors with shapes
            # so if both param and loaded_weight are a scalar,
            # "broadcast" instead of copy
            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) "
                f"into parameter ({param.size()})"
            )
            param.data.copy_(loaded_weight)
    except Exception:
        raise
```

Simple: assert shapes match and memcpy. But how do per-layer per-rank shapes end up matching for norms and embeddings? Because the layer itself was built on-GPU with the right shape: `RMSNorm(hidden_size=2048)` allocates `(2048,)`, and the HF tensor is also `(2048,)`. <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/vocab_parallel_embedding.py#L161" class="sym-link" target="_blank" rel="noopener noreferrer"><code>VocabParallelEmbedding</code></a> does the vocab-dim sharding in its own weight_loader (not <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/weight_utils.py#L1137" class="sym-link" target="_blank" rel="noopener noreferrer"><code>default_weight_loader</code></a>).

Notice: this function has **no knowledge of tensor parallelism**. It just asserts and copies. The per-rank narrowing was done by the specialized weight_loaders above. For the small/unparallel tensors (layer norms, scalars, router gate), the whole tensor fits on every rank identically and there's nothing to shard.

---

### 5.9 KV pool — <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool.py#L742" class="sym-link" target="_blank" rel="noopener noreferrer"><code>MHATokenToKVPool</code></a>
{: #runner-kv }

With weights loaded and LoRA buffers committed (§6.2), `configure_kv_cache_dtype()` and `init_memory_pool()` allocate the token → KV-entry pool. Every attention forward reads and writes here. First, dtype selection:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> model_runner.py:2026 — configure_kv_cache_dtype <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L2026" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def configure_kv_cache_dtype(self):
    if self.server_args.kv_cache_dtype == "auto":
        quant_config = getattr(self.model, "quant_config", None)
        kv_cache_quant_algo = getattr(quant_config, "kv_cache_quant_algo", None)
        if isinstance(kv_cache_quant_algo, str) and kv_cache_quant_algo.upper() == "FP8":
            self.kv_cache_dtype = torch.float8_e4m3fn   # or fp8_dtype on AMD
        else:
            self.kv_cache_dtype = self.dtype
    elif self.server_args.kv_cache_dtype == "fp8_e5m2":
        self.kv_cache_dtype = torch.float8_e5m2
    elif self.server_args.kv_cache_dtype == "fp8_e4m3":
        self.kv_cache_dtype = torch.float8_e4m3fn
    elif self.server_args.kv_cache_dtype in ("bf16", "bfloat16"):
        self.kv_cache_dtype = torch.bfloat16
    elif self.server_args.kv_cache_dtype == "fp4_e2m1":
        if hasattr(torch, "float4_e2m1fn_x2"):
            self.kv_cache_dtype = torch.float4_e2m1fn_x2
        else:
            self.kv_cache_dtype = self.dtype
    else:
        raise ValueError(...)
    log_info_on_rank0(logger, f"Using KV cache dtype: {self.kv_cache_dtype}")
```

For a default Qwen3-30B-A3B-Instruct-2507 run (`--kv-cache-dtype auto`, bf16 weights), **`kv_cache_dtype = torch.bfloat16`**.

Then `init_memory_pool` profiles remaining GPU memory, divides by per-token KV size, and allocates the buffer:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> model_runner_kv_cache_mixin.py:754 — init_memory_pool <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py#L754" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def init_memory_pool(self: ModelRunner, pre_model_load_memory: int):
    if not self.spec_algorithm.is_none() and self.is_draft_worker:
        assert self.memory_pool_config is not None, "Draft worker requires memory_pool_config"
    else:
        self.memory_pool_config = self._resolve_memory_pool_config(pre_model_load_memory)

    self._apply_memory_pool_config(self.memory_pool_config)

    logger.info(
        f"Memory pool end. "
        f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
    )
```

The interesting part is inside `_apply_memory_pool_config` → <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool.py#L744" class="sym-link" target="_blank" rel="noopener noreferrer"><code>MHATokenToKVPool.__init__</code></a>:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> mem_cache/memory_pool.py:742 — MHATokenToKVPool.__init__ and _create_buffers <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool.py#L742" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
class MHATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,              # max_total_num_tokens (post-profile)
        page_size: int,
        dtype: torch.dtype,     # kv_cache_dtype
        head_num: int,          # per-rank KV heads
        head_dim: int,          # config.head_dim
        layer_num: int,         # per-rank layers (=48 for no PP)
        device: str,
        enable_memory_saver: bool,
        ...
    ):
        super().__init__(size, page_size, dtype, layer_num, device, enable_memory_saver, ...)
        self.head_num = swa_head_num if swa_head_num is not None else head_num
        self.head_dim = swa_head_dim if swa_head_dim is not None else head_dim
        self.v_head_dim = (
            swa_v_head_dim if swa_v_head_dim is not None
            else v_head_dim if v_head_dim is not None
            else head_dim
        )
        self._create_buffers()
        # ...

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # [size, head_num, head_dim] for each layer
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            self.k_buffer = [
                torch.zeros(
                    (self.size + self.page_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype, device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.v_buffer = [
                torch.zeros(
                    (self.size + self.page_size, self.head_num, self.v_head_dim),
                    dtype=self.store_dtype, device=self.device,
                )
                for _ in range(self.layer_num)
            ]
        self.k_data_ptrs = torch.tensor([x.data_ptr() for x in self.k_buffer], dtype=torch.uint64, device=self.device)
        self.v_data_ptrs = torch.tensor([x.data_ptr() for x in self.v_buffer], dtype=torch.uint64, device=self.device)
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [np.prod(x.shape[1:]) * x.dtype.itemsize for x in self.k_buffer + self.v_buffer],
            device=self.device,
        )
```

For Qwen3-30B-A3B-Instruct-2507 at TP=4, bf16 KV cache:

| Constructor arg        | Value for our run                                                  |
|------------------------|--------------------------------------------------------------------|
| `head_num` (per rank)  | num_key_value_heads / tp_size = 4 / 4 = <span class="num">1</span> |
| `head_dim`             | <span class="num">128</span>                                       |
| `layer_num` (per rank) | <span class="num">48</span> (no PP)                                |
| `dtype`                | `torch.bfloat16`                                                   |
| `page_size`            | <span class="num">1</span> (default for FA3 on H100/H200)          |

<div class="callout info" markdown="1">

#### What `page_size = 1` actually means

`page_size` is the number of tokens' worth of KV state that share one allocation unit — the same concept as the "block size" parameter in vLLM's PagedAttention paper. It's the quantum in which the allocator hands out memory. When `page_size = 1`, SGLang **skips PagedAttention entirely** and uses a flat per-token allocator: every token is its own unit.

The allocator selection happens at `model_runner_kv_cache_mixin.py:638`:

```python
elif self.page_size == 1:
    self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(...)   # flat, no blocks
else:
    self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
        self.max_total_num_tokens,
        page_size=self.page_size,
        ...
    )
```

<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/allocator.py#L117" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TokenToKVPoolAllocator</code></a> hardcodes its page size to 1 in its call to the base class (`allocator.py:128`), and its `alloc(need_size)` just returns `free_pages[:need_size]` — N requested tokens get N arbitrary free row indices, no block alignment. The attention backend (FA3 on H200) reads/writes `k_buffer[layer][idx]` for each of those indices directly, with no concept of block.

The concrete payoff:

- **No block-padding waste.** A request using 37 tokens holds exactly 37 KV rows, not 48 (with padding up to a block of 16) or 64 (up to a block of 64).
- **No fragmentation model needed.** Requests of any length get exactly their size from the free pool.
- The cost is that the attention kernel has to gather KV at per-token granularity — but FA3 on Hopper uses async TMA loads that handle this efficiently, so the cost is negligible. This is exactly why the <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L2675" class="sym-link" target="_blank" rel="noopener noreferrer"><code>_handle_page_size</code></a> default is 1 on CUDA/ROCm and 64 on MUSA (MooreThreads) — MUSA's attention kernel benefits from block-aligned loads but Hopper's does not.

The `self.size + self.page_size` term in the allocation — visible in the buffer shape `(max_total_num_tokens + page_size, head_num, head_dim)` — is the "extra row reservation" for the **padded slot 0**, which is where batched attention writes dummy outputs for padded tokens. With `page_size = 1` this is just one extra row; with `page_size = 64` it would be a whole extra page of 64 rows.

</div>

Per-token cost (both K and V, across all local layers, one rank):

```text
per_token_bytes = 2 (K+V) · layer_num · head_num · head_dim · dtype_bytes
                = 2 · 48 · 1 · 128 · 2 bytes
                = 24 576 bytes
                = 24 KB per token per rank
```

Given `mem_fraction_static` (default 0.88 on H200), and the H200's 141 GB HBM3e per GPU: after subtracting ~14.5 GB for model weights, ~0.3 GB for LoRA buffers (§6.3), ~1 GB for CUDA graphs (§5.11), ~0.3 GB for CUBLAS/NCCL/FA workspaces, there's roughly **~105 GB left per rank** for KV.

```text
max_total_num_tokens ≈ 105 GB / 24 KB per token ≈ 4.4 million tokens per rank

# So across all 4 ranks the scheduler sees the min of these (each rank computes its own
# budget and takes the MIN in tp_worker.get_worker_info), which means ~4.4M tokens total
# available — way more than the 32k context we asked for, so the server will happily
# schedule thousands of concurrent 32k-token requests.
```

The allocated buffer itself: 48 × 2 × (max_total_num_tokens + page_size) × 1 × 128 × 2 B. Startup log line: `"KV Cache is allocated. #tokens: {max_total_num_tokens}, K size: {k_size_bytes/GB} GB, V size: {v_size_bytes/GB} GB"` (this log is observed, not derived; it prints from `_finalize_allocation_log`).

<div class="callout motiv" markdown="1">

#### Why `k_buffer` is a **list** of tensors, one per layer?

Storing KV as `List[Tensor]` (one (size, head_num, head_dim) tensor per layer) rather than one big 4D `(layer, size, head, dim)` tensor has two advantages: (1) layers can have different head counts (critical for MLA, SWA, hybrid Mamba); (2) you can hand layer `i`'s K/V pointer directly to the attention kernel without slicing, which would otherwise force a stride computation inside the kernel. The `data_ptrs` + `data_strides` tensors exist for Triton kernels that need to walk all layers in one kernel launch (e.g. KV-copy during speculative decoding).

</div>

### 5.10 Attention-backend selection
{: #runner-attn }

SGLang ships 15+ attention backends (FA3, FlashInfer, Triton, TRTLLM, Ascend, etc.). `server_args.get_attention_backends()` returns `(prefill_backend, decode_backend)` and `init_attention_backend` dispatches.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> model_runner.py:2083 — init_attention_backend <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L2083" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def init_attention_backend(self):
    """Init attention kernel backend."""
    if self.server_args.enable_pdmux:
        self.attn_backend = self._get_attention_backend(init_new_workspace=True)
        self.decode_attn_backend_group = [self._get_attention_backend() for _ in range(self.server_args.sm_group_num)]
        self.decode_attn_backend = self.decode_attn_backend_group[0]
    elif self.server_args.enable_two_batch_overlap and not self.is_draft_worker:
        self.attn_backend = TboAttnBackend.init_new(self._get_attention_backend)
    else:
        self.attn_backend = self._get_attention_backend()

def _get_attention_backend(self, init_new_workspace: bool = False):
    # ... draft worker override ...
    self.prefill_attention_backend_str, self.decode_attention_backend_str = \
        self.server_args.get_attention_backends()
    if self.decode_attention_backend_str != self.prefill_attention_backend_str:
        attn_backend = HybridAttnBackend(self,
            decode_backend=self._get_attention_backend_from_str(self.decode_attention_backend_str, ...),
            prefill_backend=self._get_attention_backend_from_str(self.prefill_attention_backend_str, ...))
    else:
        attn_backend = self._get_attention_backend_from_str(self.server_args.attention_backend, ...)
    return attn_backend
```

The decision of which backend gets auto-picked lives in `server_args._handle_attention_backend_compatibility`, whose comment block spells out the policy:

> "**1.1** We will turn on FA3 on hopper unless user use spec decode with topk \> 1 or page_size \> 1. **1.2** Use trtllm_mha for SM100/SM103 (Blackwell B200/GB200/B300) excluding spec with topk \> 1. Note: trtllm_mha does not support SM120, which will fall back to flashinfer. **1.3** In other cases, we will use flashinfer if available, otherwise use triton."

On an H200 (Hopper SM_90), no spec decode, page_size=1, Qwen3 is non-MLA → the picker sets `attention_backend = "fa3"`. That routes to <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashattention_backend.py#L87" class="sym-link" target="_blank" rel="noopener noreferrer"><code>FlashAttentionBackend</code></a>, which wraps `flash-attn 3`. Prefill uses varlen FA3, decode uses the optimized single-token FA3 kernel.

<div class="callout info" markdown="1">

#### The `page_size > 1` constraint in policy 1.1, explained

FA3 on Hopper is designed around per-token attention gathers — its async TMA loads handle each token's K/V row independently. It does not support operating on block-aligned KV pages the way Flashinfer's block-sparse or TRT-LLM's block-MHA kernels do. So the policy is: **if the user forces `page_size > 1` (e.g., for a backend that requires paged layout, or on a GPU like MUSA where the default is 64), FA3 is structurally incompatible** and the picker skips past rule 1.1 to pick a paged-aware backend. The same is true for speculative decoding with `topk > 1`: the tree-attention pattern needed there doesn't fit FA3's kernel shape either. See §5.9 for what `page_size = 1` means in the allocator.

</div>

On Blackwell (SM_100, B200/GB200), it switches to `trtllm_mha` (TensorRT-LLM's MHA kernels, via TensorRT-LLM's Python bindings). On consumer cards (SM_120 RTX 50) it falls back to FlashInfer.

<div class="callout motiv" markdown="1">

#### Why three families of kernels?

FA3 (<a href="https://arxiv.org/abs/2407.08608" target="_blank" rel="noopener noreferrer">Dao et al. 2024</a>) uses asynchronous warp-specialization optimized for Hopper; it's the fastest on H100/H200 for bf16/fp16 attention, especially at long contexts. FlashInfer exposes finer-grained kernels per attention pattern (prefix-shared, tensor-core, etc.) and is the only family with mature Blackwell SM_100 support today. TRTLLM-MHA uses NVIDIA's production kernels including FP8 MHA. Triton is the catch-all fallback so you can always run on any platform — it's slower but portable.

</div>

### 5.11 CUDA-graph capture
{: #runner-graphs }

CUDA graphs eliminate the CPU-launch overhead of running 48 × ~30 kernels per forward pass. `init_device_graphs()` builds a <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/cuda_graph_runner.py#L512" class="sym-link" target="_blank" rel="noopener noreferrer"><code>CudaGraphRunner</code></a> and captures graphs for each batch size in `capture_bs`.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> cuda_graph_runner.py:512-620 — CudaGraphRunner.__init__ (excerpt) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/cuda_graph_runner.py#L512" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
class CudaGraphRunner:
    """A CudaGraphRunner runs the forward pass of a model with cuda graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        self.model_runner = model_runner
        # ...
        self.capture_forward_mode = ForwardMode.DECODE
        self.capture_hidden_mode = CaptureHiddenMode.NULL
        self.num_tokens_per_bs = 1
        if model_runner.spec_algorithm.is_speculative():
            self.capture_forward_mode = ForwardMode.TARGET_VERIFY
            self.num_tokens_per_bs = self.model_runner.server_args.speculative_num_draft_tokens
        elif self.is_dllm:
            self.capture_forward_mode = ForwardMode.DLLM_EXTEND
            self.num_tokens_per_bs = self.dllm_config.block_size

        # Batch sizes to capture
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(
            model_runner, self.num_tokens_per_bs
        )
        log_info_on_rank0(logger, f"Capture cuda graph bs {self.capture_bs}")

        # Attention backend pre-allocates metadata for max bs
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs
        self.model_runner.attn_backend.init_cuda_graph_state(self.max_bs, self.max_num_token)

        # ...

        if self.model_runner.server_args.enable_lora:
            # Phase 2 of LoRA CUDA graph init: dense LoRA batch metadata.
            # Phase 1 (MoE buffers) was handled earlier in ModelRunner via
            # lora_manager.init_cuda_graph_moe_buffers().
            self.model_runner.lora_manager.init_cuda_graph_batch_info(
                max_bs_in_cuda_graph=self.max_bs,
                num_tokens_per_bs=self.num_tokens_per_bs,
            )
```

For default run, `capture_bs` is a list like `[1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, ...]` up to `cuda_graph_max_bs`. The actual list comes from <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/cuda_graph_runner.py#L137" target="_blank" rel="noopener noreferrer">get_batch_sizes_to_capture</a> which bands small bs densely and larger bs coarsely.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> cuda_graph_runner.py:761 — capture loop (excerpt) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/cuda_graph_runner.py#L761" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def capture(self) -> None:
    # ...
    def _capture_one_stream(stream_idx: Optional[int] = None):
        # Reverse the order to enable better memory sharing across cuda graphs.
        capture_range = (tqdm.tqdm(list(reversed(self.capture_bs)))
                         if get_tensor_model_parallel_rank() == 0
                         else reversed(self.capture_bs))
        for i, bs in enumerate(capture_range):
            # ...
            with patch_model(self.model_runner.model, bs in self.compile_bs,
                             num_tokens=bs * self.num_tokens_per_bs,
                             tp_group=self.model_runner.tp_group) as forward:
                graph, output_buffers = self.capture_one_batch_size(bs, forward, stream_idx)
                key = bs if stream_idx is None else f"{stream_idx}_{bs}"
                self.graphs[key] = graph
                self.output_buffers[key] = output_buffers

    with freeze_gc(self.model_runner.server_args.enable_cudagraph_gc):
        with graph_capture() as graph_capture_context, profile_context as prof:
            self.stream = graph_capture_context.stream
            _capture_one_stream()
```

**Capture largest → smallest.** Each `CUDAGraph` allocates its own memory pool for intermediate activations, and PyTorch's graph capture context lets smaller graphs *reuse* the memory from a larger graph's pool (since any activation needed at bs=32 fits in the buffer already committed at bs=128). This one detail saves several GB of GPU memory in practice.

At serving time, when a decode batch of size B comes in, the runner picks the smallest captured bs ≥ B, pads the batch out to that size, copies inputs into the static input buffers the graph was captured against, and `graph.replay()` — a single CUDA launch that runs all 48 layers × dozens of kernels without any Python or driver overhead.

<div class="callout info" markdown="1">

#### What happens if my batch is bigger than `cuda_graph_max_bs`?

`CudaGraphRunner.can_run(forward_batch)` returns `False`, and the forward falls back to eager execution. You'll see no error — just a latency spike. Raise `--cuda-graph-max-bs` to cover your peak. For Qwen3-30B-A3B-Instruct-2507 at TP=4, the default max is 256; each captured graph costs single-digit MBs of VRAM.

</div>

---

<p class="bridge" markdown="span">*With the base model fully set up, LoRA adds a parallel infrastructure layered on top: adapter weight buffers, layer wrappers that splice LoRA deltas into each forward pass, and per-batch routing metadata. Skip this Part if you're not running with `--enable-lora`; otherwise read it top to bottom because the pieces interlock tightly.*</p>

## 6 · LoRA subsystem
{: #lora }

SGLang's LoRA implementation adapts two pieces of prior research: <a href="https://arxiv.org/abs/2311.03285" target="_blank" rel="noopener noreferrer">S-LoRA</a> (unified paging for thousands of concurrent adapters) and <a href="https://arxiv.org/abs/2310.18547" target="_blank" rel="noopener noreferrer">Punica</a> (grouped GEMMs via segmented sgmv). The top-of-file comment on `lora/lora_manager.py` credits both explicitly: `# Integrates "S-LoRA: Serving Thousands of Concurrent LoRA Adapters" and "Punica: Multi-Tenant LoRA Serving"`. The design goals are:

- **Many adapters, one batch.** At runtime a single batch can contain requests using different LoRA adapters (including no adapter); one forward pass handles them all through per-segment GEMMs.
- **Pay only for what's active.** Memory cost is `max_loras_per_batch × max_rank`, not `total_loaded_adapters × max_rank`. Loaded-but-unused adapters live in CPU memory and stream in on demand.
- **No change to the base model's cost.** The base model runs exactly as before; LoRA is a delta added after each targeted linear via two small GEMMs (A then B).

The moving parts, roughly in the order they're encountered:

| Component                                                                                                                                                                                                    | Process                        | Owns                                                                                                                                                                                                                                                                |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_registry.py#L54" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRARegistry</code></a>               | TokenizerManager (main)        | Name → <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_registry.py#L27" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRARef</code></a> mapping, concurrent-request counters, load/unload synchronization. |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L53" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAManager</code></a>                 | Scheduler (per-TP-rank)        | Wrapped layer modules, memory pool, backend, per-batch metadata orchestration.                                                                                                                                                                                      |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/mem_pool.py#L49" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAMemoryPool</code></a>                  | (inside LoRAManager)           | GPU tensors: `A_buffer`/`B_buffer` keyed by target module, plus embedding buffers.                                                                                                                                                                                  |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L30" class="sym-link" target="_blank" rel="noopener noreferrer"><code>BaseLayerWithLoRA</code></a> subclasses      | (wrap each target `nn.Module`) | Override `forward` to add the LoRA delta; call out to the backend.                                                                                                                                                                                                  |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/backend/triton_backend.py#L22" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TritonLoRABackend</code></a> | (owned by LoRAManager)         | Triton sgemm kernels + per-batch <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/utils.py#L12" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRABatchInfo</code></a> state.                                     |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/utils.py#L12" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRABatchInfo</code></a>                      | GPU state                      | Per-request (not per-token) adapter routing: `weight_indices`, `seg_indptr`, `lora_ranks`.                                                                                                                                                                          |

### 6.1 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_registry.py#L54" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRARegistry</code></a> — naming, not weights
{: #lora-registry }

The registry lives in the **TokenizerManager process** and is the single source of truth for "which adapter names exist". It is *not* where weights live. Its job is to (a) map human-friendly names to UUIDs, (b) hand out ID references to active requests, and (c) gate load/unload operations against in-flight requests.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> python/sglang/srt/lora/lora_registry.py:26-51 — LoRARef <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_registry.py#L26" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
@dataclass(frozen=True)
class LoRARef:
    """
    Reference record for a LoRA model.

    This object guarantees a unique ``lora_id`` and may include ``lora_name``, ``lora_path``, and ``pinned``.
    The ID eliminates conflicts from reused LoRA names or paths and can be used to generate deterministic cache
    keys (e.g., radix cache).
    """

    lora_id: str = field(default_factory=lambda: uuid4().hex)
    lora_name: Optional[str] = None
    lora_path: Optional[str] = None
    pinned: Optional[bool] = None
```

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora_registry.py:54-85 — LoRARegistry init <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_registry.py#L54" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
class LoRARegistry:
    """
    The central registry to keep track of available LoRA adapters and ongoing LoRA requests.

    The `LoRARegistry` resides in the tokenizer manager process and acts as the single source of
    truth for all available LoRA adapters. It supports concurrent inference and dynamic adapter
    updates through a two-phase update / eventual consistency model between the tokenizer manager
    process and the scheduler processes.
    """

    def __init__(self, lora_paths: Optional[List[LoRARef]] = None):
        # A read-write lock to ensure adapters loading / unloading operations are exclusive.
        self._registry_lock = RWLock()
        # An ordered dictionary holding LoRARef objects, name→LoRARef, in LRU order.
        self._registry: OrderedDict[str, LoRARef] = OrderedDict()
        # Counters for ongoing requests, mapping from LoRA ID to ConcurrentCounter.
        self._counters: Dict[str, ConcurrentCounter] = {}

        if lora_paths:
            for lora_ref in lora_paths:
                self._register_adapter(lora_ref)
```

The key methods used during a request's life:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora_registry.py:115-154 — acquire <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_registry.py#L115" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
async def acquire(self, lora_name: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Queries registry for LoRA IDs based on LoRA names and start tracking the usage of the
    corresponding LoRA adapters by incrementing its counter.
    """
    def _lookup(name: str) -> str:
        if name is None: return None
        lora_ref = self._registry.get(name, None)
        if lora_ref is None:
            raise ValueError(
                f"The following requested LoRA adapters are not loaded: {name}\n"
                f"Loaded adapters: {self._registry.keys()}."
            )
        self._registry.move_to_end(name)   # LRU touch
        return lora_ref.lora_id

    if isinstance(lora_name, str):
        async with self._registry_lock.writer_lock:
            lora_id = _lookup(lora_name)
        await self._counters[lora_id].increment(notify_all=False)
        return lora_id
    elif isinstance(lora_name, list):
        async with self._registry_lock.writer_lock:
            lora_ids = [_lookup(name) for name in lora_name]
        await asyncio.gather(*[self._counters[id].increment(notify_all=False)
                               for id in lora_ids if id is not None])
        return lora_ids
```

<div class="callout motiv" markdown="1">

#### Why is the lookup under `writer_lock`, not `reader_lock`?

Because `move_to_end` mutates the ordered dict (that's the LRU update). Lookup still needs serializing writes. But the counter increment happens *after* releasing the lock, so it doesn't block concurrent acquires; it only contests with other increments/decrements. Loading or unloading, in contrast, takes the writer lock and waits for `counter == 0` before removing an adapter. This two-phase consistency is what the class docstring promises: "concurrent inference and dynamic adapter updates through a two-phase update / eventual consistency model between the tokenizer manager process and the scheduler processes."

</div>

The `acquire` call is what turns a user-facing `"adapter0"` (or a list of them, for a batch of requests with different adapters) into a UUID that's attached to the request and shipped to the scheduler:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> tokenizer_manager.py:2450 (inside generate_request path) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py#L2450" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
obj.lora_id = await self.lora_registry.acquire(obj.lora_path)
```

The scheduler subprocesses never call the registry directly. They get `lora_id`s via IPC, look up the adapter weights in their own `LoRAManager.loras` dict, and report back when the request finishes (so the registry's counter gets decremented via `release`).

---

### 6.2 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L54" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAManager.__init__</code></a> → `init_state`
{: #lora-manager }

The scheduler's `ModelRunner.init_lora_manager()` (called during §5.1's `initialize`) constructs a <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L53" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAManager</code></a> after the model weights have loaded. The constructor's signature tells you what it needs to know up front:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora_manager.py:52-108 — LoRAManager.__init__ (excerpt) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L52" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
class LoRAManager:
    def __init__(
        self,
        base_model: torch.nn.Module,
        base_hf_config: AutoConfig,
        max_loras_per_batch: int,
        load_config: LoadConfig,
        dtype: torch.dtype,
        server_args: ServerArgs,
        lora_backend: str = "triton",
        tp_size: int = 1,
        tp_rank: int = 0,
        max_lora_rank: Optional[int] = None,
        target_modules: Optional[Iterable[str]] = None,
        lora_paths: Optional[List[LoRARef]] = None,
    ):
        self.base_model = base_model
        self.base_hf_config = (base_hf_config.get_text_config()
                               if hasattr(base_hf_config, "get_text_config")
                               else base_hf_config)
        self.max_loras_per_batch = max_loras_per_batch
        # ...
        self.eviction_policy = server_args.lora_eviction_policy
        self._experts_shared_outer_override = server_args.experts_shared_outer_loras
        self.lora_use_virtual_experts = server_args.lora_use_virtual_experts
        self.lora_strict_loading = getattr(server_args, "lora_strict_loading", False)

        # LoRA backend for running sgemm kernels
        logger.info(f"Using {lora_backend} as backend of LoRA kernels.")
        backend_type = get_backend_from_name(lora_backend)
        self.lora_backend: BaseLoRABackend = backend_type(
            max_loras_per_batch=max_loras_per_batch,
            device=self.device,
            server_args=server_args,
        )

        # Initialize mutable internal state of the LoRAManager.
        self.init_state(
            max_lora_rank=max_lora_rank,
            target_modules=target_modules,
            lora_paths=lora_paths,
        )
```

`init_state` runs the six steps that get every piece of LoRA infrastructure in place. Here's the orchestration:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora_manager.py:413-448 — init_state <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L413" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def init_state(
    self,
    max_lora_rank: Optional[int] = None,
    target_modules: Optional[Iterable[str]] = None,
    lora_paths: Optional[List[LoRARef]] = None,
):
    """Initialize the internal (mutable) state of the LoRAManager."""
    assert lora_paths or (
        max_lora_rank is not None and target_modules is not None
    ), ("When no initial --lora-paths is provided, you need to specify both "
        "--max-lora-rank and --lora-target-modules for LoRA initialization.")

    self.init_lora_adapters(lora_paths)           # 1. parse configs + load CPU weights
    self.init_lora_shapes(                        # 2. resolve max_lora_rank + target_modules
        max_lora_rank=max_lora_rank,
        target_modules=target_modules,
    )

    if self._experts_shared_outer_override is not None:
        self.experts_shared_outer_loras = self._experts_shared_outer_override
    else:
        self.experts_shared_outer_loras = self._detect_shared_outer_loras()  # 3. 3D-vs-4D auto-detect
    if self.experts_shared_outer_loras:
        logger.info("Shared outer LoRA mode enabled: gate_up lora_A and "
                    "down lora_B will be shared across experts (expert_dim=1).")

    self.init_lora_modules()    # 4. wrap every target nn.Module with BaseLayerWithLoRA subclass
    self.init_memory_pool()     # 5. allocate the 3D/4D GPU buffers
    self.update_lora_info()     # 6. plumb the buffers into each wrapped layer's set_lora_info()
```

#### Step 1 — `init_lora_adapters`: load CPU-side adapter weights

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora_manager.py:450-469 — init_lora_adapters <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L450" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def init_lora_adapters(self, lora_paths: Optional[List[LoRARef]] = None):
    # Configs of all active LoRA adapters, indexed by LoRA ID.
    self.configs: Dict[str, LoRAConfig] = {}

    # LoRA adapter weights cached in CPU memory, indexed by LoRA ID.
    self.loras: Dict[str, LoRAAdapter] = {}

    # Mapping from LoRA ID to LoRARef object.
    self.lora_refs: Dict[str, LoRARef] = {}

    self.num_pinned_loras: int = 0

    if lora_paths:
        for lora_ref in lora_paths:
            result = self.load_lora_adapter(lora_ref)
            if not result.success:
                raise RuntimeError(
                    f"Failed to load LoRA adapter {lora_ref.lora_name}: {result.error_message}"
                )
```

For each `--lora-paths adapter0=/path`, `load_lora_adapter` reads the adapter's `adapter_config.json` into a <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_config.py#L25" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAConfig</code></a>, validates compatibility (rank ≤ `max_lora_rank`, no new vocab tokens, target modules subset, etc.), then loads the adapter tensors into a <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora.py#L48" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAAdapter</code></a> kept in **CPU memory**. GPU memory is only touched later when the adapter gets paged into an active slot.

#### Step 3 — `_detect_shared_outer_loras`

Before looking at the auto-detect logic, it's worth spelling out what "shared outer" actually means, because the shape arithmetic is dense.

Recall the base LoRA factorization: each linear layer's weight update is `ΔW = B @ A`, where `A` has shape `(rank, input_dim)` and `B` has shape `(output_dim, rank)`. For a dense (non-MoE) module like `qkv_proj`, SGLang stores `A` and `B` as **3-D** tensors, adding a leading `num_loras` axis so one buffer holds all loaded adapters:

| Module                               | LoRA A shape                       | LoRA B shape                    |
|--------------------------------------|------------------------------------|---------------------------------|
| Dense (e.g. `qkv_proj`, `down_proj`) | `(num_loras, rank · c, input_dim)` | `(num_loras, output_dim, rank)` |

The `c` multiplier is from `get_stacked_multiply` — it's `3` for `qkv_proj` (q/k/v fused), `2` for `gate_up_proj` (w1/w3 fused), and `1` elsewhere. It expands the LoRA rank dimension so one fused buffer holds the LoRA-A rows for all the sub-projections.

For MoE modules this gets an extra dimension because there are `num_experts` distinct expert matrices. The layout promotes to **4-D** with a new expert axis. But there are **two valid conventions** for what goes into that expert axis:

##### Per-expert (not shared) — `expert_dim = num_experts`

Each expert has its own LoRA A and its own LoRA B. Total parameters scale linearly with `num_experts`. For Qwen3-30B-A3B (128 experts, hidden 2048, moe_intermediate 768, rank 16):

| Module             | LoRA A shape (per-expert)                                     | LoRA B shape (per-expert)                                    |
|--------------------|---------------------------------------------------------------|--------------------------------------------------------------|
| `gate_up_proj_moe` | `(num_loras, 128, 16·2, 2048)` = `(num_loras, 128, 32, 2048)` | `(num_loras, 128, 768·2, 16)` = `(num_loras, 128, 1536, 16)` |
| `down_proj_moe`    | `(num_loras, 128, 16, 768)`                                   | `(num_loras, 128, 2048, 16)`                                 |

Per-adapter parameter count for these two modules at rank 16, per layer: `128 × (32·2048 + 1536·16 + 16·768 + 2048·16)` = `128 × (65 536 + 24 576 + 12 288 + 32 768)` = `128 × 135 168` = ~**17.3 M parameters per layer**. Across 48 layers, ~830 M params per adapter just for MoE LoRA — and that's a small-rank (`r = 16`) adapter.

##### Shared outer — `expert_dim = 1`

All 128 experts share the same outer matrix. The expert axis in the stored tensor collapses to 1, and at runtime the kernel broadcasts that single matrix across every expert:

| Module             | LoRA A shape (shared)                                 | LoRA B shape (shared)                                  |
|--------------------|-------------------------------------------------------|--------------------------------------------------------|
| `gate_up_proj_moe` | `(num_loras, 1, 32, 2048)`                            | `(num_loras, 128, 1536, 16)` *(B is still per-expert)* |
| `down_proj_moe`    | `(num_loras, 128, 16, 768)` *(A is still per-expert)* | `(num_loras, 1, 2048, 16)` *(B is shared)*             |

This is the asymmetric detail that took me a moment to see in the code at <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/mem_pool.py#L203" target="_blank" rel="noopener noreferrer"><code>mem_pool.py:203</code></a> and `mem_pool.py:258`:

```python
if self.is_moe_module(module_name):
    num_experts = self._get_num_experts(base_model)
    expert_dim = num_experts
    if self.experts_shared_outer_loras and module_name == "gate_up_proj_moe":
        expert_dim = 1
    return (
        self.max_loras_per_batch,
        expert_dim,
        max_lora_dim * c,
        input_dim,
    )
```

For LoRA A, only `gate_up_proj_moe` gets collapsed to `expert_dim = 1`. For LoRA B, only `down_proj_moe` gets collapsed. The "outer" in the name refers specifically to the hidden-size–facing sides of the two projections: A of gate_up reads from `hidden_size`, B of down_proj writes to `hidden_size`. Both of these are the sides that, if shared, let you compute `ΔW·x` as a single non-MoE operation first and then dispatch the result into per-expert routing — which is exactly what makes the shared form cheaper to execute, not just to store.

Parameter savings in shared form, for the same rank-16 config:

|                           | Per-expert (4-D)           | Shared outer             | Savings |
|---------------------------|----------------------------|--------------------------|---------|
| `gate_up_proj_moe` LoRA A | `128 × 32 × 2048` = 8.39 M | `1 × 32 × 2048` = 65.5 K | ~128×   |
| `down_proj_moe` LoRA B    | `128 × 2048 × 16` = 4.19 M | `1 × 2048 × 16` = 32.8 K | ~128×   |

Total reduction per layer is ~12.5 M → ~100 K on the two collapsible tensors. The non-collapsible ones (gate_up B, down_proj A) stay at 128×-expert scale because they face the `moe_intermediate_size` side and carry per-expert information that can't be factored out.

##### The detector

Adapters on the HF Hub don't record which convention they were trained in — they just ship the tensors. SGLang peeks at the first 3-D `gate_up_proj lora_A` it finds and checks whether `weight.shape[0] == 1` (shared) or equals `num_experts` (per-expert). All adapters in a batch must agree, or it errors out:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora_manager.py:471-505 — _detect_shared_outer_loras <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L471" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def _detect_shared_outer_loras(self) -> bool:
    """Auto-detect shared outer LoRA format from loaded adapter weights.

    MoE adapters with shared outer experts store 3D tensors where
    dim[0]=1 indicates weights shared across all experts, while
    dim[0]=num_experts indicates per-expert weights.
    Returns True if gate_up lora_A has expert_dim=1 (shared).
    """
    shared_outer: Optional[bool] = None
    for adapter_id, adapter in self.loras.items():
        for layer in adapter.layers:
            for name, weight in layer.weights.items():
                if ("gate_up_proj" in name
                    and "lora_A" in name
                    and weight.dim() == 3):
                    is_shared = weight.shape[0] == 1
                    if shared_outer is None:
                        shared_outer = is_shared
                    elif shared_outer != is_shared:
                        raise RuntimeError(
                            "Mixed shared-outer LoRA formats detected across "
                            f"loaded adapters (conflict in adapter '{adapter_id}')."
                        )
    return bool(shared_outer) if shared_outer is not None else False
```

<div class="callout info" markdown="1">

#### Why `weight.dim() == 3` for a 4-D MoE module?

The stored adapter files flatten the `num_loras` dim per-adapter (each adapter file stores **one** adapter), so the tensor on disk is 3-D: `(expert_dim, rank, input_dim)`. The 4-D layout we discussed above is the *runtime buffer* where SGLang stacks multiple adapters together. The detector reads a single loaded adapter before the runtime buffers are created, so it sees 3-D. `weight.shape[0]` in this context is already the expert axis (0 for shared, or `num_experts` for per-expert).

</div>

#### Step 4 — `init_lora_modules`: wrap every target layer

This is the step that physically replaces <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L866" class="sym-link" target="_blank" rel="noopener noreferrer"><code>QKVParallelLinear</code></a> instances (and the like) with <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L526" class="sym-link" target="_blank" rel="noopener noreferrer"><code>QKVParallelLinearWithLoRA</code></a> wrappers. It walks the model's module tree, decides which nodes are LoRA targets, and calls `replace_submodule` to swap them in place.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora_manager.py:712-830 — init_lora_modules (excerpt) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L712" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def set_lora_module(self, module_name, module):
    """Wrap any module (standard or MoE) with LoRA support."""
    lora_module = get_lora_layer(module, self.lora_backend)
    replace_submodule(self.base_model, module_name, lora_module)
    return lora_module

def init_lora_modules(self):
    # Look-up table that essentially maps (layer_index, module_name) → LoRA module.
    self.lora_modules: List[Dict[str, BaseLayerWithLoRA]] = [
        {} for _ in range(self.base_hf_config.num_hidden_layers)
    ]

    self.embed_tokens_module: Optional[BaseLayerWithLoRA] = None
    self.lm_head_module: Optional[BaseLayerWithLoRA] = None

    # Tied-embeddings untie hack (for models with tie_word_embeddings=True) — N/A for Qwen3-30B-A3B-Instruct-2507
    # ... (omitted; see §1.1 — tie_word_embeddings=false here) ...

    for module_name, module in self.base_model.named_modules():
        if "embed_tokens" in module_name and "embed_tokens" in self.target_modules:
            if isinstance(module, VocabParallelEmbedding) and not isinstance(module, BaseLayerWithLoRA):
                lora_module = self.set_lora_module(module_name, module)
                self.embed_tokens_module = lora_module
                continue
        if "lm_head" in module_name and "lm_head" in self.target_modules:
            if isinstance(module, ParallelLMHead) and not isinstance(module, BaseLayerWithLoRA):
                lora_module = self.set_lora_module(module_name, module)
                self.lm_head_module = lora_module
                continue

        # DeepSeek MLA special case (fused_qkv_a_proj_with_mqa) — skip for Qwen3
        # ... omitted ...

        # The module should be converted if it is included in target_names
        if module_name.split(".")[-1] in self.target_modules:
            layer_id = get_layer_id(module_name)
            if layer_id is None:
                continue
            self.lora_modules[layer_id][module_name] = self.set_lora_module(module_name, module)
            continue

        if isinstance(module, FusedMoE) and all(
            x in self.target_modules for x in ["gate_up_proj", "down_proj"]
        ):
            layer_id = get_layer_id(module_name)
            lora_module = self.set_lora_module(module_name, module)
            lora_module.experts_shared_outer_loras = self.experts_shared_outer_loras
```

After this loop, for `target_modules={"all"}` (normalized internally to the full set for Qwen3), `self.lora_modules` looks like:

```python
self.lora_modules = [
    # layer 0:
    {
      "model.layers.0.self_attn.qkv_proj":    QKVParallelLinearWithLoRA(...),
      "model.layers.0.self_attn.o_proj":      RowParallelLinearWithLoRA(...),
      "model.layers.0.mlp.experts":           FusedMoEWithLoRA(...),
    },
    # layer 1: same structure ...
    ...
    # layer 47: same structure ...
]
self.embed_tokens_module = VocabParallelEmbeddingWithLoRA(...)   # if "embed_tokens" in target_modules
self.lm_head_module = ParallelLMHeadWithLoRA(...)                # if "lm_head" in target_modules
```

#### Step 5 — `init_memory_pool`

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora_manager.py:686-704 — init_memory_pool <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L686" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def init_memory_pool(self):
    """(Re)initialize the LoRA memory pool based on the current configurations."""
    self.memory_pool = LoRAMemoryPool(
        base_hf_config=self.base_hf_config,
        max_loras_per_batch=self.max_loras_per_batch,
        dtype=self.dtype,
        tp_size=self.tp_size,
        tp_rank=self.tp_rank,
        max_lora_rank=self.max_lora_rank,
        target_modules=self.target_modules,
        base_model=self.base_model,
        eviction_policy=self.eviction_policy,
        lora_added_tokens_size=self.lora_added_tokens_size,
        experts_shared_outer_loras=self.experts_shared_outer_loras,
        strict_loading=self.lora_strict_loading,
    )

    # Initializing memory pool with base model
    self.fetch_new_loras({None})
```

`None` is a sentinel meaning "the no-adapter slot" — one buffer index is reserved for requests that want the plain base model, and `fetch_new_loras({None})` zeros it out.

#### Step 6 — `update_lora_info`: plug buffers into every wrapper

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora_manager.py:332-411 — update_lora_info (excerpt) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L332" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def update_lora_info(self):
    """Update all LoRA modules to associate them with the latest memory buffer."""
    for layer_id, layer_modules in enumerate(self.lora_modules):
        for module_name, module in layer_modules.items():
            # Hack for FusedMoE layer
            if isinstance(module, FusedMoEWithLoRA) and all(
                x in self.target_modules for x in ["gate_up_proj", "down_proj"]
            ):
                gate_up_key = ("gate_up_proj_moe"
                               if "gate_up_proj_moe" in self.memory_pool.A_buffer
                               else "gate_up_proj")
                down_key = ("down_proj_moe"
                            if "down_proj_moe" in self.memory_pool.A_buffer
                            else "down_proj")
                gate_up_a = self.memory_pool.get_tensor(gate_up_key, layer_id, LoRAType.LORA_A)
                gate_up_b = self.memory_pool.get_tensor(gate_up_key, layer_id, LoRAType.LORA_B)
                down_a    = self.memory_pool.get_tensor(down_key,    layer_id, LoRAType.LORA_A)
                down_b    = self.memory_pool.get_tensor(down_key,    layer_id, LoRAType.LORA_B)
                module.set_lora_info(
                    gate_up_lora_a_weights=gate_up_a,
                    gate_up_lora_b_weights=gate_up_b,
                    down_lora_a_weights=down_a,
                    down_lora_b_weights=down_b,
                )
                continue

            target_module = get_target_module_name(module_name, self.memory_pool.target_modules)
            module.set_lora_info(
                self.memory_pool.get_tensor(target_module, layer_id, LoRAType.LORA_A),
                self.memory_pool.get_tensor(target_module, layer_id, LoRAType.LORA_B),
            )
    # lm_head / embed_tokens plumbing omitted
```

Each wrapped layer now holds direct `torch.Tensor` references to the right slice of the global memory pool. The forward pass will index *into* these tensors via the per-batch `weight_indices`, so no pointer-chasing happens on the GPU hot path.

---

### 6.3 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/mem_pool.py#L49" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAMemoryPool</code></a> — 3D dense + 4D MoE buffers
{: #lora-pool }

The memory pool is where GPU LoRA weights live. Its design answers three questions at once: how much GPU memory does LoRA cost? where are the per-adapter boundaries? and how does a kernel find the right slice for the request it's processing?

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora/mem_pool.py:49-95 — LoRAMemoryPool (excerpt) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/mem_pool.py#L49" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
class LoRAMemoryPool:
    """Class for memory pool management of lora modules"""

    def __init__(
        self,
        base_hf_config: AutoConfig,
        max_loras_per_batch: int,
        dtype: torch.dtype,
        tp_size: int,
        tp_rank: int,
        max_lora_rank: int,
        target_modules: Set[str],
        base_model: torch.nn.Module,
        eviction_policy: str,
        lora_added_tokens_size: int,
        experts_shared_outer_loras: bool = False,
        strict_loading: bool = False,
    ):
        self.num_layer: int = base_hf_config.num_hidden_layers
        self.max_loras_per_batch: int = max_loras_per_batch
        # ...

        # Both A_buffer and B_buffer maps lora weight names to its buffer space.
        # Standard LoRA (3D): [num_loras, rank, hidden_dim]
        # MoE LoRA (4D):      [num_loras, num_experts, rank, hidden_dim]
        # The dimensionality is determined by the module type (MoE vs standard)
        self.A_buffer: Dict[str, List[torch.Tensor]] = {}
        self.B_buffer: Dict[str, List[torch.Tensor]] = {}

        self.embedding_A_buffer: Dict[str, torch.Tensor] = {}
        self.embedding_B_buffer: Dict[str, torch.Tensor] = {}

        self.lm_head_A_buffer: Dict[str, torch.Tensor] = {}
        self.lm_head_B_buffer: Dict[str, torch.Tensor] = {}
        self.new_embeddings_buffer: Dict[str, torch.Tensor] = {}

        # Lora uid -> buffer idx in memory pool
        self.uid_to_buffer_id: Dict[Optional[str], int] = {}
        self.buffer_id_to_uid: List[Union[str, None, EmptySlot]] = [EMPTY_SLOT] * self.max_loras_per_batch
```

The buffer keys are target-module names (`"qkv_proj"`, `"o_proj"`, `"gate_up_proj_moe"`, `"down_proj_moe"`, etc.); the list is indexed by layer. Shape functions make it concrete:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> mem_pool.py:175-213 — get_lora_A_shape <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/mem_pool.py#L175" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def get_lora_A_shape(
    self, module_name: str, base_model: torch.nn.Module,
    max_lora_dim: int, layer_idx: int,
) -> Tuple[int]:
    """
    Get shape for LoRA A weights. Automatically returns 3D or 4D based on module type.

    Returns:
        - Standard: [num_loras, rank, hidden_dim]
        - MoE:      [num_loras, num_experts, rank, hidden_dim]
    """
    input_dim, _ = get_hidden_dim(module_name, self.base_hf_config, base_model, layer_idx)
    c = get_stacked_multiply(module_name)
    if (self.tp_size > 1
        and module_name in ROW_PARALLELISM_LINEAR_LORA_NAMES
        and module_name not in REPLICATED_LINEAR_LORA_NAMES):
        input_dim = divide(input_dim, self.tp_size)

    if self.is_moe_module(module_name):
        num_experts = self._get_num_experts(base_model)
        expert_dim = num_experts
        if self.experts_shared_outer_loras and module_name == "gate_up_proj_moe":
            expert_dim = 1
        return (self.max_loras_per_batch, expert_dim, max_lora_dim * c, input_dim)
    else:
        return (self.max_loras_per_batch, max_lora_dim * c, input_dim)

def get_lora_B_shape(
    self, module_name: str, base_model: torch.nn.Module,
    max_lora_dim: int, layer_idx: int,
) -> Tuple[int]:
    _, output_dim = get_hidden_dim(module_name, self.base_hf_config, base_model, layer_idx)
    if (self.tp_size > 1
        and module_name not in ROW_PARALLELISM_LINEAR_LORA_NAMES
        and module_name not in REPLICATED_LINEAR_LORA_NAMES):
        output_dim = divide(output_dim, self.tp_size)

    if self.is_moe_module(module_name):
        num_experts = self._get_num_experts(base_model)
        expert_dim = num_experts
        if self.experts_shared_outer_loras and module_name == "down_proj_moe":
            expert_dim = 1
        return (self.max_loras_per_batch, expert_dim, output_dim, max_lora_dim)
    else:
        return (self.max_loras_per_batch, output_dim, max_lora_dim)
```

Two constants do real work here: `c = get_stacked_multiply(module_name)` accounts for the pre-fused layers — `c=3` for `qkv_proj` (q+k+v stacked), `c=2` for `gate_up_proj` (gate+up stacked), `c=1` otherwise. And the `ROW_PARALLELISM_LINEAR_LORA_NAMES` set (`o_proj`, `down_proj`) controls which axis to shard.

Shapes for a concrete Qwen3-30B-A3B run with `--max-lora-rank 64 --max-loras-per-batch 4 --lora-target-modules all --tp 4`:

| Buffer key                                        | Shape formula                | Value (per rank, bf16) | Bytes/layer |
|---------------------------------------------------|------------------------------|------------------------|-------------|
| `qkv_proj` A                                      | (M, 3·r, H)                  | (4, 192, 2048)         | 3.0 MB      |
| `qkv_proj` B                                      | (M, q_shard + 2·kv_shard, r) | (4, 1280, 64)          | 0.63 MB     |
| `o_proj` A (row-parallel, input sharded)          | (M, r, H/TP)                 | (4, 64, 512)           | 0.25 MB     |
| `o_proj` B                                        | (M, H, r)                    | (4, 2048, 64)          | 1.0 MB      |
| `gate_up_proj_moe` A (4D, per-expert)             | (M, E, 2·r, H)               | (4, 128, 128, 2048)    | 256 MB      |
| `gate_up_proj_moe` B                              | (M, E, 2·I/TP, r)            | (4, 128, 384, 64)      | 24 MB       |
| `down_proj_moe` A (row-parallel, input I sharded) | (M, E, r, I/TP)              | (4, 128, 64, 192)      | 12 MB       |
| `down_proj_moe` B                                 | (M, E, H, r)                 | (4, 128, 2048, 64)     | 128 MB      |

Per-layer LoRA memory (standard + MoE) ≈ 425 MB per rank. Across 48 layers: **~20.4 GB per rank**. That's a significant cost — more than the base model's ~14.5 GB per rank. It's why you have to choose `max_lora_rank` and `target_modules` carefully: narrowing to just `qkv_proj,o_proj` (skip MoE expert LoRA) drops it to \< 200 MB total across all layers.

<div class="callout motiv" markdown="1">

#### Why allocate the full `max_loras_per_batch` slots up front?

Pre-allocating fixed-size buffers is core to S-LoRA / Punica. Dynamic allocation would (a) fragment the GPU allocator, (b) make kernel indexing variable-size (each request looks up adapter weights by buffer_id and indexes the same tensor), and (c) violate CUDA-graph capture, which requires static tensor addresses and shapes. The trade-off is that you pay for `max_loras_per_batch` slots whether or not you use them — but that lets the kernels be as simple as possible.

</div>

<div class="callout info" markdown="1">

#### The "ambiguous modules" special case (`gate_up_proj`, `down_proj`)

These two names appear in both dense (Llama-style MLP) and MoE contexts. For a model with MoE **and** shared experts, the pool allocates **both** a 3D `gate_up_proj` buffer (for the shared dense MLP) **and** a 4D `gate_up_proj_moe` buffer (for the routed expert MLPs). Qwen3-30B-A3B-Instruct-2507 has no shared experts (`shared_expert_intermediate_size` not set, `n_shared_experts` not set), so only the 4D MoE version is allocated — confirmed in `init_buffers` at `has_shared_experts = False`.

</div>

---

### 6.4 Adapter weight loading — Disk → CPU → GPU pool
{: #lora-load }

So far §6.2 mentioned "weights are loaded into CPU memory" and §6.3 described the pre-allocated GPU pool. The missing piece is the loader that bridges them. There are **two distinct phases**, and the split is the whole point of the S-LoRA-style architecture: you can cheaply hold hundreds of adapters in CPU RAM, and pay the GPU price only for the ones actually in use right now.

| Phase              | When                                                                                                                                                                                                                                                | Inputs                                                      | Outputs                                                                                                                                                                                                                                      |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 — Disk → CPU     | Startup (if `--lora-paths`) or `/load_lora_adapter` HTTP call                                                                                                                                                                                       | `adapter_model.safetensors` + `adapter_config.json`         | `self.loras[uid]`: a <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora.py#L48" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAAdapter</code></a> with CPU tensors, grouped by layer |
| 2 — CPU → GPU pool | Per forward batch, inside <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/forward_batch_info.py#L279" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ForwardBatch</code></a>`.init_new` | CPU-resident `LoRAAdapter` + pre-allocated GPU pool buffers | Pool slot populated; `uid_to_buffer_id[uid] = buffer_id`                                                                                                                                                                                     |

#### Phase 1 — Disk → CPU

Entry point: <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L151" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAManager.load_lora_adapter</code></a> is called for each `--lora-paths adapter0=/path` during init, and also by the `/load_lora_adapter` HTTP handler at runtime. It reads `adapter_config.json`, validates the adapter (rank ≤ `max_lora_rank`, target modules subset, etc.), then delegates weight loading to `load_lora_weights`:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora/lora_manager.py:613-631 — load_lora_weights <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L613" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def load_lora_weights(self, lora_ref: LoRARef):
    """Load the weights of a LoRA adapter to CPU memory and conducts post-loading validation."""
    lora_adapter = LoRAAdapter(
        lora_ref.lora_id,
        self.configs[lora_ref.lora_id],
        self.base_hf_config,
        self.load_config,
        self.lora_backend,
    )
    lora_adapter.initialize_weights()

    # If we want to overlap loading LoRA adapters with compute, they must be pinned in CPU memory
    if self.enable_lora_overlap_loading:
        lora_adapter.pin_weights_in_cpu()

    self.loras[lora_ref.lora_id] = lora_adapter
```

<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora.py#L76" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAAdapter.initialize_weights</code></a> is where safetensors actually get read from disk. The interesting choice: **it reuses the base model's loader** (`DefaultModelLoader` — see §5.4). From the loader's perspective, `adapter_model.safetensors` is just another safetensors archive:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora/lora.py:76-89 — LoRAAdapter.initialize_weights <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora.py#L76" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def initialize_weights(self):
    model_path = self.config.path
    loader = DefaultModelLoader(self.load_config)
    revision = getattr(self.config.hf_config, "revision", None)

    for name, loaded_weight in loader._get_weights_iterator(
        DefaultModelLoader.Source(
            model_path, revision=revision, fall_back_to_pt=True
        )
    ):
        self._process_weight(name, loaded_weight)

    self._normalize_weights()
```

`_process_weight` routes each tensor to one of three homes on the adapter instance — and immediately moves it to CPU with `loaded_weight.cpu()`:

- `self.layers[layer_id].weights[name]` — for per-layer modules (qkv, gate_up, down, o_proj, MoE experts). Indexed by layer so `load_lora_weight_to_buffer` can walk layers in order.
- `self.embedding_layers[name]` — for `embed_tokens` and `lm_head`.
- `self.added_tokens_embeddings[name]` — for adapters that extend the vocab. Currently unsupported in SGLang serving but the storage is here.

After every tensor is read, `_normalize_weights` runs a chain of renaming/fusing passes so the adapter's naming convention matches SGLang's fused-layer conventions:

- `normalize_qkv_proj` — if the adapter has separate `q_proj`/`k_proj`/`v_proj` LoRA tensors, concatenate them along dim 0 into a single `qkv_proj` tensor matching the base model's fused QKV layout.
- `_rename_expert_w_to_proj` — rename `w1` → `gate_proj`, `w3` → `up_proj`, `w2` → `down_proj` for adapters trained against DeepSeek/Mixtral-style expert weight names.
- `normalize_gate_up_proj` — analogous stacking for `gate_proj` + `up_proj` → `gate_up_proj`.
- `normalize_fused_qkv_a_proj` — MLA-specific, for DeepSeek `q_a_proj` + `kv_a_proj_with_mqa` fusion.

<div class="callout info" markdown="1">

#### What if the adapter only trained some of q, k, v?

Many LoRA adapters only tune q and v (it's a common setting). When `normalize_qkv_proj` runs, it checks which of q/k/v the adapter shipped. If `k_proj` is missing, it **initializes the k-portion of the fused qkv tensor to zeros** (`lora.py:171`): `torch.zeros_like(weights[v_name])`. This way the fused `qkv_proj` tensor has the right shape, and the zero k-portion contributes nothing to the output — exactly matching "no LoRA on K" behavior.

</div>

When Phase 1 finishes, the `LoRAAdapter` has all its tensors sitting in CPU memory (pinned if `enable_lora_overlap_loading` was set, so Phase 2's H2D copy can run async). Nothing has touched GPU yet.

#### Phase 2 — CPU → GPU pool

This fires per batch, lazily. From <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/forward_batch_info.py#L596" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ForwardBatch.init_new</code></a> at `forward_batch_info.py:596`:

```python
if model_runner.server_args.enable_lora:
    if not model_runner.server_args.enable_lora_overlap_loading:
        model_runner.lora_manager.fetch_new_loras(set(ret.lora_ids))

    model_runner.lora_manager.prepare_lora_batch(ret)
```

<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L284" class="sym-link" target="_blank" rel="noopener noreferrer"><code>fetch_new_loras</code></a> is the routing function. It passes the current batch's UIDs, the CPU-resident adapters dict, and the layer-wrapper map to the memory pool:

```python
def fetch_new_loras(self, new_loras, running_loras=set()):
    cur_uids = new_loras | running_loras
    assert len(cur_uids) <= self.max_loras_per_batch
    self.memory_pool.prepare_lora_batch(
        cur_uids=cur_uids,
        lora_adapters=self.loras,          # ← CPU-resident LoRAAdapters from Phase 1
        lora_modules=self.lora_modules,    # ← wrapper handles for TP slicing
        lora_refs=self.lora_refs.copy(),
        ...
    )
```

<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/mem_pool.py#L421" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAMemoryPool.prepare_lora_batch</code></a> at `mem_pool.py:421` does two things per UID that isn't already resident in a pool slot:

1. **Allocate a slot.** `get_available_buffer_slot()` first looks for an empty slot in the pool. If full, it picks a victim using `self.eviction_policy` (LRU by default), skipping adapters needed by the current batch and pinned adapters. One notable tiebreak: it **prefers evicting LoRA adapters over the `None` slot** (the placeholder for non-LoRA requests), so the `None` slot stays warm across batches.
2. **Copy the weights.** <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/mem_pool.py#L502" class="sym-link" target="_blank" rel="noopener noreferrer"><code>load_lora_weight_to_buffer</code></a> walks the adapter's layers and copies each tensor into its pool slot.

The copy itself has three interesting pieces of logic worth calling out (all inside `load_lora_weight_to_buffer`):

**1. The `None` UID zeros out A, not both A and B.** When a non-LoRA request gets a pool slot (for use in the rank-0 no-op path — see the dense-layer kernel short-circuit in §6.8), only LoRA-A gets zeroed:

```python
if uid is None:
    for i in range(self.num_layer):
        for k in self.A_buffer.keys():
            self.A_buffer[k][i][buffer_id] = 0
    for k in self.embedding_A_buffer.keys():
        self.embedding_A_buffer[k][buffer_id] = 0
    for k in self.lm_head_A_buffer.keys():
        self.lm_head_A_buffer[k][buffer_id] = 0
    return
```

The B buffer is left unchanged. This is safe because `ΔW · x = B @ (A @ x)` — if A is zero, the intermediate is zero regardless of B, so the delta contributes nothing. And the kernel's `if rank == 0: return` early-exit skips even the A-read anyway (§6.8). The zeroing is a belt-and-suspenders fallback; the early-exit is the real short-circuit.

**2. Per-weight MoE routing.** For MoE modules, the loader uses a regex to detect per-expert weights and group them into a dict keyed by expert index:

```python
expert_match = re.search(r"experts\.(\d+)\.", name)
if expert_match:
    # Per-expert MoE weight — 2D tensors, one per expert
    target_module = target_module + "_moe"
    ...
    expert_id = int(expert_match.group(1))
    if "lora_A" in name:
        temp_A_buffer[target_module][expert_id] = weights
    else:
        temp_B_buffer[target_module][expert_id] = weights
elif "experts" in name and weights.dim() == 3:
    # Shared outer MoE weight — 3D tensor [expert_dim, rank, hidden]
    target_module = target_module + "_moe"
    if "lora_A" in name:
        temp_A_buffer[target_module] = weights
    ...
```

The 3-D-tensor branch is for shared-outer adapters (§6.3). Per-expert 2-D tensors accumulate into a dict that later gets stacked into the pool's 4-D buffer.

**3. TP slicing happens at copy time via the wrapper's helpers.** Before the actual `copy_`, the CPU tensor is passed through the corresponding layer wrapper's `slice_lora_a_weights` / `slice_lora_b_weights`:

```python
temp_A_buffer[target_module] = module.slice_lora_a_weights(
    temp_A_buffer[target_module], self.tp_rank
)
temp_B_buffer[target_module] = module.slice_lora_b_weights(
    temp_B_buffer[target_module], self.tp_rank
)
```

For example, <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L574" class="sym-link" target="_blank" rel="noopener noreferrer"><code>QKVParallelLinearWithLoRA.slice_lora_b_weights</code></a> narrows a full-size LoRA-B tensor down to this TP rank's q/k/v slice. The sliced result is what lands in the pool buffer. After slicing, the actual H2D copy is the one-liner:

```python
buffer_view.copy_(weight, non_blocking=True)
```

`non_blocking=True` is why `pin_weights_in_cpu()` matters in Phase 1: if the source tensor is pinned, this H2D copy dispatches on a side CUDA stream, concurrent with ongoing compute. Without pinning it's blocking but still fast — a rank-16 adapter is only ~200 MB for Qwen3 sizes, and PCIe 5 moves that in milliseconds.

<div class="callout motiv" markdown="1">

#### Why is Phase 2 lazy?

The obvious alternative is "load every adapter to GPU at startup." It's rejected because it doesn't scale: if you have 100 adapters and each takes ~200 MB per rank, that's 20 GB per rank of weights that spend most of their time idle. The S-LoRA architecture (<a href="https://arxiv.org/abs/2311.03285" target="_blank" rel="noopener noreferrer">Sheng et al., 2023</a>) observed that real serving workloads have a small *working set* of adapters hot at any moment — so treating the GPU pool as a cache with LRU eviction lets you serve 100× more adapters than you could if they were all resident.

The per-batch check in `prepare_lora_batch` only pays for adapters whose UIDs changed since the last batch. Steady state: zero H2D cost once the working set is resident. Churn cost: one ~ms-scale H2D copy per new adapter introduced to the batch.

</div>

---

### 6.5 Layer wrappers & <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L1054" class="sym-link" target="_blank" rel="noopener noreferrer"><code>get_lora_layer</code></a>
{: #lora-layers }

Each wrapped layer inherits from <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L30" class="sym-link" target="_blank" rel="noopener noreferrer"><code>BaseLayerWithLoRA</code></a>, a thin base that holds a reference to the *original* layer and forwards to `base_layer.forward` by default:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora/layers.py:30-56 — BaseLayerWithLoRA <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L30" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
class BaseLayerWithLoRA(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        lora_backend: BaseLoRABackend,
    ):
        super().__init__()
        self.base_layer: nn.Module = base_layer
        self.set_lora: bool = False
        self.lora_backend: BaseLoRABackend = lora_backend
        if hasattr(self.base_layer, "weight"):
            self.weight = self.base_layer.weight
        if hasattr(self.base_layer, "bias") and self.base_layer.bias is not None:
            self.bias = self.base_layer.bias

    def forward(self, x: torch.Tensor):
        return self.base_layer.forward(x)

    def set_lora_info(self, *args):
        pass

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        pass

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        pass
```

<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L1054" class="sym-link" target="_blank" rel="noopener noreferrer"><code>get_lora_layer</code></a> dispatches on the base layer's type. Note: the order matters — <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L138" class="sym-link" target="_blank" rel="noopener noreferrer"><code>FusedMoE</code></a> is checked before every other type, because a <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L138" class="sym-link" target="_blank" rel="noopener noreferrer"><code>FusedMoE</code></a> wouldn't match a `nn.Linear`-shaped isinstance check anyway, but the comment warns that subclasses of <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L286" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ColumnParallelLinear</code></a> (e.g. <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L866" class="sym-link" target="_blank" rel="noopener noreferrer"><code>QKVParallelLinear</code></a> inherits from it) must be listed **before** their base class:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora/layers.py:1054-1072 — get_lora_layer <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L1054" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def get_lora_layer(
    layer: nn.Module, lora_backend: BaseLoRABackend
) -> BaseLayerWithLoRA:
    supported_layer_types = {
        # the order matters
        FusedMoE:                   FusedMoEWithLoRA,
        ParallelLMHead:             ParallelLMHeadWithLoRA,
        VocabParallelEmbedding:     VocabParallelEmbeddingWithLoRA,
        ReplicatedLinear:           ReplicatedLinearWithLoRA,
        QKVParallelLinear:          QKVParallelLinearWithLoRA,
        MergedColumnParallelLinear: MergedColumnParallelLinearWithLoRA,
        ColumnParallelLinear:       ColumnParallelLinearWithLoRA,
        RowParallelLinear:          RowParallelLinearWithLoRA,
    }
    for src_layer_type, lora_layer_type in supported_layer_types.items():
        if isinstance(layer, src_layer_type):  # pylint: disable=unidiomatic-typecheck
            ret = lora_layer_type(layer, lora_backend)
            return ret
    raise Exception(f"No corresponding LoRA layer supported for {type(layer)}.")
```

For Qwen3-30B-A3B at TP=4, each decoder layer's modules map to the following wrappers:

| Base module in SGLang                                                                                                                                                                                                    | Wrapper type                                                                                                                                                                                          | Count per layer                                               |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L866" class="sym-link" target="_blank" rel="noopener noreferrer"><code>QKVParallelLinear</code></a> (self_attn.qkv_proj)     | <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L526" class="sym-link" target="_blank" rel="noopener noreferrer"><code>QKVParallelLinearWithLoRA</code></a> | 1                                                             |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L1312" class="sym-link" target="_blank" rel="noopener noreferrer"><code>RowParallelLinear</code></a> (self_attn.o_proj)      | <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L603" class="sym-link" target="_blank" rel="noopener noreferrer"><code>RowParallelLinearWithLoRA</code></a> | 1                                                             |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L191" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ReplicatedLinear</code></a> (mlp.gate — the router)   | <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L694" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ReplicatedLinearWithLoRA</code></a>  | 1 <span class="sub">(only if "gate" in target_modules)</span> |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L138" class="sym-link" target="_blank" rel="noopener noreferrer"><code>FusedMoE</code></a> (mlp.experts) | <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L782" class="sym-link" target="_blank" rel="noopener noreferrer"><code>FusedMoEWithLoRA</code></a>          | 1                                                             |

Plus (outside the decoder stack): <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/vocab_parallel_embedding.py#L161" class="sym-link" target="_blank" rel="noopener noreferrer"><code>VocabParallelEmbedding</code></a> → <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L58" class="sym-link" target="_blank" rel="noopener noreferrer"><code>VocabParallelEmbeddingWithLoRA</code></a> and <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/vocab_parallel_embedding.py#L512" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ParallelLMHead</code></a> → <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L224" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ParallelLMHeadWithLoRA</code></a>.

Let's look at the QKV wrapper since that's the most representative standard-linear case:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora/layers.py:526-571 — QKVParallelLinearWithLoRA <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L526" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
class QKVParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(
        self,
        base_layer: QKVParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        q_proj_shard_size = self.base_layer.q_proj_shard_size
        kv_proj_shard_size = self.base_layer.kv_proj_shard_size
        self.output_offset = torch.tensor(
            [
                0,
                q_proj_shard_size,                      # end of Q
                q_proj_shard_size + kv_proj_shard_size, # end of K
                q_proj_shard_size + 2 * kv_proj_shard_size,  # end of V
            ],
            dtype=torch.int32,
            device=next(self.base_layer.parameters()).device,
        )
        self.output_offset_cpu = self.output_offset.cpu()
        # For computing number of launched blocks
        self.max_qkv_out_dim = max(q_proj_shard_size, kv_proj_shard_size)

    def set_lora_info(
        self,
        A_buffer_qkv: torch.Tensor,
        B_buffer_qkv: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer_qkv = A_buffer_qkv
        self.B_buffer_qkv = B_buffer_qkv

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_output = self.lora_backend.run_qkv_lora(
            x=x,
            qkv_lora_a=self.A_buffer_qkv,
            qkv_lora_b=self.B_buffer_qkv,
            base_output=base_output,
            output_offset=self.output_offset,
            output_offset_cpu=self.output_offset_cpu,
            max_qkv_out_dim=self.max_qkv_out_dim,
        )
        return lora_output
```

For Qwen3 TP=4: `output_offset = [0, 1024, 1152, 1280]` (end-of-Q, end-of-K, end-of-V in the fused output dim). `A_buffer_qkv` shape `(M=4, 3·64=192, H=2048)` contains per-adapter LoRA-A for all three of q, k, v stacked. `B_buffer_qkv` shape `(M=4, 1280, 64)` contains per-adapter LoRA-B matching the fused output layout.

`apply_lora` delegates to `lora_backend.run_qkv_lora`, which we'll see in §6.8. The key thing the wrapper holds is: **layer-specific offsets** (so the kernel knows where q/k/v live in the fused output) and **layer-specific slicing helpers** (`slice_lora_b_weights`) used at adapter-load time to narrow a full-size LoRA-B down to this rank's TP slice.

#### Where does the forward actually decide base vs LoRA-wrapped?

Nowhere — at runtime. **The decision was made once at model-load time by physically replacing the base module in the `nn.Module` tree with the LoRA-wrapped version.** After that, the normal model forward just calls whatever module is at `model.layers[i].self_attn.qkv_proj`, and that slot now holds a wrapped instance.

The swap itself is one `setattr` call:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> utils/common.py:1165-1173 — replace_submodule <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/common.py#L1165" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def replace_submodule(
    model: nn.Module, module_name: str, new_module: nn.Module
) -> nn.Module:
    """Replace a submodule in a model with a new module."""
    parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
    target_name = module_name.split(".")[-1]
    setattr(parent, target_name, new_module)
    return new_module
```

This is invoked by `LoRAManager.set_lora_module` (`lora_manager.py:707`) once per `(layer_id, target_module)` pair during `init_lora_modules`. After it finishes, every LoRA-target attribute on the base model points to a `BaseLayerWithLoRA` subclass, and the original base layer now lives as `wrapper.base_layer` inside the wrapper.

The wrapped forward always follows the same shape — see `ColumnParallelLinearWithLoRA.forward` at `layers.py:442` as the canonical example:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora/layers.py:442-457 — ColumnParallelLinearWithLoRA.forward <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L442" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def forward(self, input_: torch.Tensor):
    # duplicate the logic in ColumnParallelLinear
    bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
    output_parallel = self.base_layer.quant_method.apply(
        self.base_layer, input_, bias
    )

    if self.set_lora:
        output_parallel = self.apply_lora(output_parallel, input_)

    if self.base_layer.gather_output:
        output = tensor_model_parallel_all_gather(output_parallel)
    else:
        output = output_parallel
    output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
    return output, output_bias
```

Every wrapped layer follows the same four steps: (1) run the base layer's forward; (2) if `self.set_lora` is True, call `apply_lora(base_output, input_)` which invokes the backend's kernel; (3) do whatever post-processing the base layer needs (all-gather, add bias); (4) return.

<div class="callout info" markdown="1">

#### Two gating flags, used for different purposes

There are actually **two** different "is LoRA active?" checks, and they live at different levels:

- **`self.set_lora` — module-level, flips once.** A boolean on each `BaseLayerWithLoRA` instance. Starts `False`; flips to `True` the first time `set_lora_info` is called on the module (when LoRA buffers are bound to it). **Never flips back.** After the first batch that uses LoRA, every subsequent forward pass takes the `apply_lora` branch — even batches where no request actually uses a LoRA adapter.
- **`batch_info.has_active_lora` — batch-level, set by `prepare_lora_batch`.** Computed CPU-side at `lora_manager.py:328` as `any(lora_ranks[wi] > 0 for wi in weight_indices)`. True when at least one request in the current batch has a non-zero rank (i.e. actually uses a LoRA adapter). **Only the MoE LoRA runner reads this flag** — see the early-return at `lora_moe_runners.py:463`. Dense wrappers do not check it.

So what happens when a batch has no LoRA requests? For **dense layers**, `self.set_lora` is still True, so `apply_lora` still runs and the Triton kernel is still launched. But inside the kernel, every program reads `lora_ranks[weight_indices[batch_id]] == 0` and early-exits before doing any compute or weight load. From `sgemm_lora_a.py:56-62`:

```python
batch_id = tl.program_id(axis=1)
w_index = tl.load(weight_indices + batch_id)
rank = tl.load(lora_ranks + w_index)

# If rank is 0, this kernel becomes a no-op as the output is always trivially correct.
if rank == 0:
    return
```

So the cost of a no-LoRA batch going through dense LoRA wrappers is one kernel launch per layer (≈5-10 μs each) with zero compute inside. For **MoE layers**, the Python-level `has_active_lora` gate skips the kernel launch entirely, because MoE LoRA kernels are heavier and worth the CPU check to avoid.

</div>

---

### 6.6 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L782" class="sym-link" target="_blank" rel="noopener noreferrer"><code>FusedMoEWithLoRA</code></a> — LoRA deltas inside the MoE forward
{: #lora-moe }

This is the tricky one. Naïvely, LoRA is "add `B(A(x))` after the base linear". For an MoE layer, though, there's no single base linear — `hidden → gate_up → activation → down → sum` is a sequence where activation is nonlinear. Where do the LoRA deltas go?

The class docstring gives the answer:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora/layers.py:782-794 — FusedMoEWithLoRA docstring <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L782" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
class FusedMoEWithLoRA(BaseLayerWithLoRA):
    """
    Wrapper around FusedMoE that integrates LoRA into the MoE computation.

    Design: LoRA deltas are added at specific points in the MoE forward pass:
    1. After gate_up projection, BEFORE activation (halfway through)
    2. After down projection, BEFORE final reduction

    This follows the vLLM/HF approach where LoRA is fused into the computation
    rather than computed independently and added at the end.
    """
```

So the gated MLP math changes from (no LoRA):

```python
y = silu(gate(x)) * up(x)
out = down(y)
```

to (with LoRA, per expert):

```python
gate_part = gate(x) + gate_lora_B(gate_lora_A(x))    # LoRA point 1 — before silu
up_part   = up(x)   + up_lora_B(up_lora_A(x))        # LoRA point 1 — before silu
y = silu(gate_part) * up_part
out_partial = down(y) + down_lora_B(down_lora_A(y))  # LoRA point 2 — before TP reduce
# ... then allreduce across TP group if RowParallel
```

Packing this into SGLang's <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L138" class="sym-link" target="_blank" rel="noopener noreferrer"><code>FusedMoE</code></a> flow (where all experts run inside a single Triton kernel via grouped GEMM) requires threading the LoRA tensors through the same dispatch/combine machinery the base MoE uses. Here's the constructor that sets this up:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora/layers.py:794-856 — FusedMoEWithLoRA.__init__ <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L794" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def __init__(
    self,
    base_layer: FusedMoE,
    lora_backend: BaseLoRABackend,
):
    super().__init__(base_layer, lora_backend)

    self.experts_shared_outer_loras: bool = False
    self.lora_use_virtual_experts: bool = False
    self.quant_method = base_layer.quant_method

    self.tp_size = getattr(base_layer, "moe_tp_size", 1)
    self.tp_rank = getattr(base_layer, "moe_tp_rank", 0)
    self.intermediate_size_per_partition = getattr(
        base_layer, "intermediate_size_per_partition", None
    )
    self._uses_interleaved_gate_up = (
        getattr(base_layer.moe_runner_config, "gemm1_alpha", None) is not None
    )

    # Initialize triton_lora moe runner for batches with lora enabled
    from sglang.srt.layers.moe import MoeRunnerBackend
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
    from sglang.srt.layers.moe.utils import get_moe_runner_backend

    # Determine runner backend: prefer server arg, fall back to quant method's runner
    global_backend = get_moe_runner_backend()
    if not global_backend.is_auto():
        runner_backend = global_backend
    elif (hasattr(base_layer.quant_method, "runner")
          and base_layer.quant_method.runner is not None):
        runner_backend = base_layer.quant_method.runner.runner_backend
    else:
        runner_backend = MoeRunnerBackend.TRITON

    self._lora_runner = MoeRunner(
        runner_backend,
        base_layer.moe_runner_config,
        lora_enabled=True,
    )
    # ... triton / marlin quant info branch ...
```

The forward pass uses the base layer's *dispatch* and *combine* (which handle routing and token permutation), but substitutes its own runner — `self._lora_runner` — in the middle:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora/layers.py:913-956 — FusedMoEWithLoRA.forward <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L913" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput, **kwargs):
    """
    Forward pass with integrated LoRA computation.

    LoRA deltas are added at the correct points inside the MoE computation:
    1. After gate_up projection, before activation
    2. After down projection, before final reduction
    """
    # Build LoRA info for this batch
    lora_info = self._get_lora_info()
    # run lora moe_runner
    return self._forward_with_lora(hidden_states, topk_output, lora_info, **kwargs)

def _forward_with_lora(
    self, hidden_states: torch.Tensor, topk_output: TopKOutput, lora_info, **kwargs,
):
    base_layer = self.base_layer
    # Dispatch tokens (doesn't do much in the LoRA case)
    dispatch_output = base_layer.dispatcher.dispatch(
        hidden_states=hidden_states, topk_output=topk_output
    )
    quant_info = self._quant_info
    # Run the only lora moe runner (Triton)
    combine_input = self._lora_runner.run(
        dispatch_output, quant_info, lora_info=lora_info
    )
    final_hidden_states = base_layer.dispatcher.combine(combine_input=combine_input)
    return final_hidden_states
```

And `_get_lora_info` bundles the per-batch state (including the precomputed `adapter_enabled` mask and all four LoRA weight pointers) into a <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_moe_runners.py#L257" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAInfo</code></a> struct:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora/layers.py:870-912 — _get_lora_info <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L870" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def _get_lora_info(self):
    """Build LoRAInfo for the current batch."""
    from sglang.srt.lora.lora_moe_runners import LoRAInfo

    batch_info = self.lora_backend.batch_info
    lora_ranks = batch_info.lora_ranks

    max_lora_rank = self.down_lora_a_weights.shape[2]

    cg_buffers = getattr(self.lora_backend, "moe_cg_buffers", None)
    if cg_buffers is not None and batch_info.use_cuda_graph:
        adapter_enabled = cg_buffers["adapter_enabled"]
        adapter_enabled.zero_()
        idx_buf = cg_buffers["weight_indices_long"]
        idx_buf[: batch_info.bs] = batch_info.weight_indices[: batch_info.bs]
        adapter_enabled.index_fill_(0, idx_buf[: batch_info.bs], 1)
    else:
        adapter_enabled = torch.zeros(len(lora_ranks), dtype=torch.int32, device=lora_ranks.device)
        adapter_enabled.index_fill_(0, batch_info.weight_indices.long(), 1)

    return LoRAInfo(
        gate_up_lora_a_weights=self.gate_up_lora_a_weights,
        gate_up_lora_b_weights=self.gate_up_lora_b_weights,
        down_lora_a_weights=self.down_lora_a_weights,
        down_lora_b_weights=self.down_lora_b_weights,
        seg_indptr=batch_info.seg_indptr,
        req_to_lora=batch_info.weight_indices,
        lora_ranks=lora_ranks,
        adapter_enabled=adapter_enabled,
        max_lora_rank=max_lora_rank,
        num_experts=self.base_layer.num_experts,
        experts_shared_outer_loras=self.experts_shared_outer_loras,
        cg_buffers=cg_buffers,
        tp_size=self.tp_size, tp_rank=self.tp_rank,
        hidden_size=getattr(self.base_layer, "hidden_size", 0),
        lora_use_virtual_experts=self.lora_use_virtual_experts,
    )
```

<div class="callout info" markdown="1">

#### Why a separate `adapter_enabled` mask?

`adapter_enabled[i]` = 1 if adapter slot `i` is actually used by at least one request in the current batch, else 0. The fused MoE+LoRA kernel uses this to skip the entire LoRA-delta fused multiply for adapter slots that contribute zero tokens — saving time when only a subset of the `max_loras_per_batch` slots are active. In CUDA-graph mode this tensor is pre-allocated once (in `moe_cg_buffers`) and updated in-place each batch so the graph capture stays valid.

</div>

For the Triton MoE-LoRA kernel itself (`fused_moe_lora_kernel.py` in `sglang/srt/lora/triton_ops/`), the math per expert *e* touched by token *t* with adapter *a* is:

```text
gate_up_A:  gate_up_lora_a_weights[a, e, :, :]    shape [2·r, H]   # or shape [1, ...] if shared outer
gate_up_B:  gate_up_lora_b_weights[a, e, :, :]    shape [2·I/TP, r]
down_A:     down_lora_a_weights[a, e, :, :]       shape [r, I/TP]
down_B:     down_lora_b_weights[a, e, :, :]       shape [H, r]

# Fused gate-up LoRA delta, before activation:
ga = x @ gate_up_A.T                           # [num_tokens_for_e, 2·r]
gb = ga @ gate_up_B.T                          # [num_tokens_for_e, 2·I/TP]
gate_and_up_delta = gb.chunk(2)                # split along the fused dim
(gate_base + gate_delta) * silu(up_base + up_delta) = y

# Fused down LoRA delta, before allreduce:
da = y  @ down_A.T                             # [num_tokens_for_e, r]
db = da @ down_B.T                             # [num_tokens_for_e, H]
out_partial = down_base(y) + db

# (then TP allreduce across ranks)
```

---

### 6.7 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/utils.py#L12" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRABatchInfo</code></a> — routing state per batch
{: #lora-batch }

One piece of state persists across all LoRA layers in a forward pass: <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/utils.py#L12" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRABatchInfo</code></a>. It lives on the backend (`self.lora_backend.batch_info`) and every wrapped layer reads from it.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora/utils.py:12-49 — LoRABatchInfo <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/utils.py#L12" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
@dataclass
class LoRABatchInfo:
    # The forward mode is using CUDA Graph.
    use_cuda_graph: bool

    # Batch size
    bs: int

    # Number of segments. For triton backend, it is equal to batch size.
    num_segments: int

    # Indice pointers of each segment in shape (num_segments + 1, )
    seg_indptr: torch.Tensor

    # The index of lora adapter used by each segment, in shape (num_segments,)
    weight_indices: torch.Tensor

    # ranks of each lora adapter, in shape (lora_num,)
    lora_ranks: torch.Tensor

    # scaling of each lora adapter, in shape (lora_num,)
    scalings: torch.Tensor

    # Maximum segment length of current batch
    max_len: Optional[int]

    # Lengths of each segments in shape (num_segments,)
    seg_lens: Optional[torch.Tensor]

    # The logical (re)ordering of input rows (tokens), in shape (num_tokens,)
    permutation: Optional[torch.Tensor]

    # Total number of tokens this batch info expects (host-side int).
    # Used by lm_head LoRA to validate input shape without GPU sync.
    expected_tokens: Optional[int] = None

    # CPU-side flag: True when at least one request uses a LoRA adapter.
    # Computed from Python lists in prepare_lora_batch to avoid GPU sync.
    has_active_lora: bool = False
```

<div class="callout warn" markdown="1">

#### Important: `weight_indices` is **per-segment**, and a segment is **per-request** in this layout

The field is `weight_indices`, not `lora_indices_per_token` — its size is `num_segments`, which in this default layout equals `bs` (batch size in requests). All tokens of request *i* share the same adapter buffer slot `weight_indices[i]`. The kernels walk `seg_indptr` to find each request's token range and index `weight_indices` once per segment — saving a full `O(num_tokens)` worth of integer comparisons compared to a per-token layout. (There's a *second* layout where a segment corresponds to an adapter rather than a request; see the next subsection.)

</div>

#### What "segment" actually means — and the two layouts

A **segment** is a contiguous range of input token rows that share one piece of LoRA routing info. The metadata describing N segments is three parallel arrays:

- `seg_indptr` — CSR-style prefix sum of length `num_segments + 1`. Segment *i* covers token rows `[seg_indptr[i], seg_indptr[i+1])`.
- `seg_lens` — equivalent to `seg_indptr[i+1] - seg_indptr[i]` for convenience.
- `weight_indices[i]` — which LoRA adapter slot segment *i* uses.

The Triton kernel's grid is `(num_tiles_per_segment, num_segments)` — axis 1 launches one set of programs *per segment*. Each program loads the weights at `weight_indices[batch_id]`, then does its block of GEMM work covering `seg_lens[batch_id]` rows. So "what's a segment?" is entirely a function of how you populate these three arrays — the kernel doesn't care.

SGLang uses two segmentations. The default one built by `prepare_lora_batch` above treats **one segment per request**. The second one, built only for decode batches by <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/backend/triton_backend.py#L184" class="sym-link" target="_blank" rel="noopener noreferrer"><code>compute_sgemm_routing</code></a>, treats **one segment per adapter**. The trigger is `triton_backend.py:301-305`:

```python
# Biggest win is in decode.
is_decode = not forward_batch.forward_mode.is_extend()
if is_decode:
    self.compute_sgemm_routing(use_cuda_graph)
else:
    self.sgemm_batch_info = None
```

Both layouts live in `LoRABatchInfo` objects and are consumed by the same kernel. The backend's getter at `triton_backend.py:51-55` picks which one to hand the SGEMM helpers:

```python
def _sgemm_info(self, pruned_batch_info=None):
    """Return the sgemm batch_info (merged segments when available)."""
    if pruned_batch_info is not None:
        return pruned_batch_info
    return getattr(self, "sgemm_batch_info", None) or self.batch_info
```

The `or self.batch_info` fallback is what handles prefill: when `sgemm_batch_info` is `None`, the SGEMM kernel uses the per-request layout, and `permutation=None` tells it to read sequentially without gathers. In decode, `sgemm_batch_info` is populated with a merged-by-adapter layout and `permutation` tells the kernel where to gather rows from.

#### Worked example — the two layouts for the same batch

Suppose 8 decode requests arrive, each with one token, using adapters `[A, B, A, C, None, A, B, None]`. Assume the memory-pool assigns `A→slot 0, B→1, C→2, None→3` and `max_loras_per_batch = 8`.

**Layout 1 — per-request (`batch_info`), what every batch starts as:**

| Field            | Value                                                     |
|------------------|-----------------------------------------------------------|
| `num_segments`   | 8 (one per request)                                       |
| `seg_lens`       | `[1, 1, 1, 1, 1, 1, 1, 1]` (decode → all 1s)              |
| `seg_indptr`     | `[0, 1, 2, 3, 4, 5, 6, 7, 8]`                             |
| `weight_indices` | `[0, 1, 0, 2, 3, 0, 1, 3]` (the adapter slot per request) |
| `permutation`    | `None` (sequential reads)                                 |

If the kernel ran over this, it would launch 8 programs on grid axis 1. Each program loads its slot's weights once from HBM. **Adapter A's weights would be loaded 3 separate times** (by programs 0, 2, 5), B's 2 times, and so on — repeated HBM traffic on the same ~1 MB of adapter weights.

**Layout 2 — per-adapter (`sgemm_batch_info`), built by `compute_sgemm_routing` in decode:**

Starting from `weight_indices = [0, 1, 0, 2, 3, 0, 1, 3]`:

```python
# perm = argsort(weight_indices) stable
perm       = [0, 2, 5, 1, 6, 3, 4, 7]
#             └─ A ─┘ └─ B ┘ C  └None┘
sorted_wi  = [0, 0, 0, 1, 1, 2, 3, 3]

# For each adapter slot 0..7 (=max_loras_per_batch)
seg_starts = searchsorted(sorted_wi, [0,1,2,3,4,5,6,7])       # [0, 3, 5, 6, 8, 8, 8, 8]
seg_ends   = searchsorted(sorted_wi, [0,1,2,3,4,5,6,7], 'r')  # [3, 5, 6, 8, 8, 8, 8, 8]
seg_lens   = seg_ends - seg_starts                            # [3, 2, 1, 2, 0, 0, 0, 0]
```

Resulting `sgemm_batch_info`:

| Field            | Value                                                          |
|------------------|----------------------------------------------------------------|
| `num_segments`   | 8 (= `max_loras_per_batch`, one per adapter slot)              |
| `seg_lens`       | `[3, 2, 1, 2, 0, 0, 0, 0]`                                     |
| `seg_indptr`     | `[0, 3, 5, 6, 8, 8, 8, 8, 8]`                                  |
| `weight_indices` | `[0, 1, 2, 3, 4, 5, 6, 7]` — segment *i* uses adapter slot *i* |
| `permutation`    | `[0, 2, 5, 1, 6, 3, 4, 7]`                                     |

Now when the same Triton kernel runs over this metadata, the programs are:

- Program 0: loads A's weights once, processes rows `perm[0:3] = [0, 2, 5]` (3 rows in one GEMM)
- Program 1: loads B's weights once, processes rows `perm[3:5] = [1, 6]` (2 rows)
- Program 2: loads C's weights once, processes row `perm[5:6] = [3]` (1 row)
- Program 3: rank is 0 for the "none" slot, early-exits without any work
- Programs 4-7: `seg_lens[i] = 0`, early-exit immediately

Adapter A's weights now get read **once**, not 3 times. That's the decode win the code comment alludes to (*"Biggest win is in decode"*) — HBM bandwidth savings scale with how many requests reuse the same adapter.

<div class="callout motiv" markdown="1">

#### Why prefill doesn't merge

Prefill batches already have big segments (hundreds to thousands of rows per request), so each per-request program is already a fat GEMM that uses the GPU efficiently. The weight-load cost is amortized across many rows. Plus, prefill batches are usually only 1-2 requests at a time — there's nothing to merge. And the sort + permutation setup itself has a non-trivial cost. So for prefill, SGLang just uses the per-request layout directly, and `sgemm_batch_info` stays `None`.

</div>

For `extend` mode (prefill with variable prompt lengths), `seg_lens = extend_seq_lens`. For `decode` mode, `seg_lens = ones(bs)` (one token per request). The per-batch setup is:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora/backend/triton_backend.py:227-275 — prepare_lora_batch (excerpt) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/backend/triton_backend.py#L227" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def prepare_lora_batch(
    self,
    forward_batch: ForwardBatch,
    weight_indices: list[int],
    lora_ranks: list[int],
    scalings: list[float],
    use_cuda_graph: bool,
):
    # Use pinned memory to avoid synchronizations during host-to-device transfer
    weight_indices_tensor = torch.tensor(
        weight_indices, dtype=torch.int32, pin_memory=True, device="cpu"
    )
    lora_ranks_tensor = torch.tensor(
        lora_ranks, dtype=torch.int32, pin_memory=True, device="cpu"
    )
    scalings_tensor = torch.tensor(
        scalings, dtype=torch.float, pin_memory=True, device="cpu"
    )

    bs = forward_batch.batch_size

    if use_cuda_graph:
        batch_info = self.cuda_graph_batch_info
        batch_info.bs = forward_batch.batch_size
        batch_info.num_segments = forward_batch.batch_size
    else:
        max_len = (max(forward_batch.extend_seq_lens_cpu)
                   if forward_batch.forward_mode.is_extend() else 1)
        seg_lens = (forward_batch.extend_seq_lens
                    if forward_batch.forward_mode.is_extend()
                    else torch.ones(bs, dtype=torch.int32, device=self.device))
        seg_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=self.device)
        seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)

        batch_info = LoRABatchInfo(
            bs=forward_batch.batch_size,
            num_segments=forward_batch.batch_size,
            max_len=max_len,
            use_cuda_graph=False,
            seg_lens=seg_lens,
            # ...
        )
```

Pinned-memory CPU tensors avoid the implicit GPU synchronization that a direct CPU-list→GPU-tensor upload would trigger. In CUDA-graph mode, even this CPU→GPU copy happens in-place into the pre-allocated `cuda_graph_batch_info`; the graph capture sees the same tensor addresses every batch.

---

### 6.8 Triton kernels + per-batch routing
{: #lora-kernels }

The standard (non-MoE) LoRA kernels live in `sglang/srt/lora/triton_ops/` and are mostly just grouped sgemvs. There are six top-level ones:

| Kernel file                                    | Used by                       | What it computes                                                                                         |
|------------------------------------------------|-------------------------------|----------------------------------------------------------------------------------------------------------|
| `sgmv_lora_a.py` → `sgemm_lora_a_fwd`          | all standard linear LoRAs     | `out[t] = x[t] @ A[adapter(t)]^T` (per-token grouped by adapter).                                        |
| `sgmv_lora_b.py` → `sgemm_lora_b_fwd`          | o_proj, down_proj, gate, etc. | `out[t] = base_out[t] + (ga[t] @ B[adapter(t)]^T) × scaling`                                             |
| `qkv_lora_b.py` → `qkv_lora_b_fwd`             | qkv_proj                      | B-projection for the fused q/k/v output layout, adding deltas at the right offsets inside `base_output`. |
| `gate_up_lora_b.py` → `gate_up_lora_b_fwd`     | dense gate_up_proj (non-MoE)  | B-projection for the fused gate+up output layout.                                                        |
| `embedding_lora_a.py` → `embedding_lora_a_fwd` | embed_tokens                  | LoRA-A embedding lookup (not a matmul — direct `A[x]`).                                                  |
| `chunked_sgmv_lora_b.py`                       | lm_head LoRA                  | Chunked B-projection for the possibly-huge-vocab lm_head output.                                         |

The backend's top-level methods tie these together:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora/backend/triton_backend.py:57-108 — run_lora_a/b_sgemm and run_qkv_lora <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/backend/triton_backend.py#L57" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def run_lora_a_sgemm(
    self, x, weights, pruned_batch_info=None, stack_num: int = 1, *args, **kwargs,
) -> torch.Tensor:
    return sgemm_lora_a_fwd(
        x, weights, self._sgemm_info(pruned_batch_info), stack_num=stack_num
    )

def run_lora_b_sgemm(
    self, x, weights, base_output=None, pruned_batch_info=None, *args, **kwargs,
) -> torch.Tensor:
    return sgemm_lora_b_fwd(
        x, weights, self._sgemm_info(pruned_batch_info), base_output
    )

def run_qkv_lora(
    self, x, qkv_lora_a, qkv_lora_b, output_offset, max_qkv_out_dim,
    base_output=None, *args, **kwargs,
) -> torch.Tensor:
    # x: (s, input_dim)
    # qkv_lora_a: (num_lora, 3 * r, input_dim)
    # qkv_lora_b: (num_lora, output_dim_q + 2 * output_dim_kv, r)
    sgemm_info = self._sgemm_info()
    lora_a_output = sgemm_lora_a_fwd(x, qkv_lora_a, sgemm_info, stack_num=3)
    lora_output = qkv_lora_b_fwd(
        lora_a_output, qkv_lora_b, sgemm_info,
        output_offset, max_qkv_out_dim, base_output,
    )
    return lora_output
```

#### Per-batch adapter routing: <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/backend/triton_backend.py#L184" class="sym-link" target="_blank" rel="noopener noreferrer"><code>compute_sgemm_routing</code></a>

The token ordering delivered to each layer is arbitrary (whatever order requests arrived). For the Triton sgemm kernels to be efficient **in decode**, tokens belonging to the same adapter need to be grouped so each adapter's weights are loaded once instead of once per request. <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/backend/triton_backend.py#L184" class="sym-link" target="_blank" rel="noopener noreferrer"><code>compute_sgemm_routing</code></a> produces a permutation that groups them, and a per-adapter segment table. See §6.7 for the worked example that walks through the layout transformation step-by-step.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> lora/backend/triton_backend.py:184-225 — compute_sgemm_routing <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/backend/triton_backend.py#L184" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def compute_sgemm_routing(self, use_cuda_graph: bool):
    """Sort tokens by adapter and build merged segments for sgemm LoRA."""
    bi = self.batch_info
    bs = bi.bs
    mlpb = self.max_loras_per_batch
    wi = bi.weight_indices[:bs]

    perm = torch.argsort(wi, stable=True).to(torch.int32)
    sorted_wi = wi[perm]
    adapter_ids = torch.arange(mlpb, device=wi.device, dtype=torch.int32)
    seg_starts = torch.searchsorted(sorted_wi, adapter_ids)
    seg_ends   = torch.searchsorted(sorted_wi, adapter_ids, right=True)
    seg_lens   = seg_ends - seg_starts

    if use_cuda_graph:
        sgemm = self.cuda_graph_sgemm_batch_info
        sgemm.permutation[:bs] = perm
        sgemm.seg_lens[:] = seg_lens
        sgemm.seg_indptr[0] = 0
        torch.cumsum(sgemm.seg_lens, dim=0, out=sgemm.seg_indptr[1:])
        sgemm.max_len = bs
        sgemm.lora_ranks[:mlpb] = bi.lora_ranks[:mlpb]
        sgemm.scalings[:mlpb] = bi.scalings[:mlpb]
    else:
        # ... eager-mode branch: construct a fresh LoRABatchInfo ...
    self.sgemm_batch_info = sgemm
```

The `stable=True` flag on `argsort` preserves relative order within an adapter group — important because KV-cache positions depend on original token order. After routing, the sgemm kernels use the **permutation** to gather tokens into adapter-contiguous layout, run one GEMM per adapter segment, then scatter results back via the inverse permutation.

The call is gated on `is_decode` at the bottom of `prepare_lora_batch`:

```python
# Biggest win is in decode.
is_decode = not forward_batch.forward_mode.is_extend()
if is_decode:
    self.compute_sgemm_routing(use_cuda_graph)
else:
    self.sgemm_batch_info = None
```

Prefill batches stay on the per-request layout (see §6.7). The SGEMM helpers' getter returns whichever batch-info is available (`sgemm_batch_info` if populated, else `batch_info`), and `permutation=None` on the per-request layout tells the kernel to read rows sequentially.

<div class="callout info" markdown="1">

#### How does performance scale with number of unique adapters in a batch?

The Triton kernel's grid always uses axis 1 = `num_segments`. In decode with the merged layout, `num_segments = max_loras_per_batch` (default 8) regardless of how many are actually used — unused slots have `seg_lens[i] = 0` and those programs early-exit. So the kernel-grid cost is bounded. What varies with the number of *unique* adapters is:

- **HBM bandwidth.** Each active program loads its adapter's weights once. More unique adapters → more unique weight reads. LoRA-A for qkv at rank 16 is about 192 KB of weights per adapter; 8 unique adapters = 1.5 MB of weight reads per layer per forward.
- **Per-adapter SM occupancy.** A single adapter with 32 rows of work gets every SM focused on that one GEMM shape. With 8 adapters × 4 rows each, each program has a tiny M dimension and under-utilizes SM tiles — the GEMM kernel is tuned for certain tile sizes and small M hurts throughput.

The worst case is "batch size ≤ `max_loras_per_batch` with one request per unique adapter" — the merged layout collapses to the per-request layout and gains nothing. The best case is "batch size \>\> `max_loras_per_batch` with heavy adapter reuse" — one adapter covers many rows and the weight load is amortized. In practice, serving workloads with a handful of popular adapters at batch size 32-128 get the bulk of the win.

The hard ceiling is `max_loras_per_batch` itself (default 8). If a batch arrives with 9 unique adapters, the scheduler has to split it — you can't fit 9 adapters in the sgemm metadata buffer. Raising this value lets you run more adapter diversity at the cost of per-adapter efficiency; tune based on workload.

</div>

---

### 6.9 Two-phase CUDA-graph init
{: #lora-graph }

We now know *what* the pre-allocated `LoRABatchInfo` buffers hold (§6.7). The remaining question is **when** they get allocated. The answer involves two phases with different timing, because they're sized against different budgets:

| Phase                        | When                                                                                                                                                                                                                                                               | What it allocates                                                                                                                                                                                                                                                  | Entry point                                                                                                                                                                                                               |
|------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 — MoE intermediate buffers | Before `init_memory_pool`, so KV auto-sizing sees the cost                                                                                                                                                                                                         | Large activation scratch for the MoE+LoRA fused kernel, plus the `adapter_enabled` and `weight_indices_long` tensors the MoE LoRA runner reads at replay                                                                                                           | <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L124" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAManager.init_cuda_graph_moe_buffers</code></a> |
| 2 — Dense batch metadata     | Inside <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/cuda_graph_runner.py#L515" class="sym-link" target="_blank" rel="noopener noreferrer"><code>CudaGraphRunner.__init__</code></a>, after the KV pool has been sized | The two pre-sized <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/utils.py#L12" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRABatchInfo</code></a> objects (per-request and per-adapter layouts — see §6.7) | <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L110" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAManager.init_cuda_graph_batch_info</code></a>  |

The two-phase split exists because of a sizing dependency: Phase 2's `max_bs_in_cuda_graph` and `num_tokens_per_bs` depend on how much KV cache the system ended up with, which isn't known until `init_memory_pool` runs. But Phase 1 has to happen *before* the KV profile, so that the KV allocator subtracts the LoRA MoE scratch from its available budget and doesn't over-commit.

The allocation code for Phase 2 is a straightforward call to both layouts — you've seen this already in §6.7 and §6.8, so it's not repeated here. The key observation is that **both** layouts are allocated up front, because the same captured graph has to work whether the replay batch is prefill (uses per-request layout) or decode (uses per-adapter layout).

---

### 6.10 LoRA × RadixCache compatibility (PR #7216)
{: #lora-radix }

Historically, enabling LoRA forced `disable_radix_cache=True`. The reason: two different requests with the same prompt *but different LoRA adapters* have different KV caches — because the LoRA-modified attention projections produce different K and V values for the same prompt tokens. Treating them as the same prefix would silently corrupt attention.

<a href="https://github.com/sgl-project/sglang/pull/7216" target="_blank" rel="noopener noreferrer">PR #7216</a> (Aug 2025, by <a href="https://github.com/Fridge003" target="_blank" rel="noopener noreferrer">@Fridge003</a>) fixed this by adding the `lora_id` to the radix-tree key. Two requests can share a cached prefix only if they use the same adapter (or both use none).

The enabling piece in the PR is in <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_registry.py#L27" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRARef</code></a>'s docstring, already quoted in §6.1:

> "The ID eliminates conflicts from reused LoRA names or paths and can be used to generate deterministic cache keys (e.g., radix cache)."

Since `lora_id` is a UUID baked at adapter-load time, its bytes are what the tree actually hashes on. The <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py#L558" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Req</code></a> struct carries `lora_id` end-to-end, and <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py#L374" class="sym-link" target="_blank" rel="noopener noreferrer"><code>RadixCache.match_prefix</code></a> incorporates it. For a user this is transparent: the only change they notice is that radix caching now works with LoRA on, and concurrent requests with *different* adapters don't poison each other's caches. See the PR for the sequence-of-key-bytes implementation details.

---

<p class="bridge" markdown="span">*We've now covered every component in isolation. Time to put them back together and watch a single request flow through the whole stack — from HTTP arrival to token streaming out.*</p>

## 7 · A request, end to end
{: #request }

Now that every piece is on GPU and every wire is connected, let's trace a single `POST /v1/chat/completions` request through the three processes. Say the user sends:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B-Instruct-2507:adapter0",
    "messages": [{"role": "user", "content": "Explain MoE routing in 3 sentences."}],
    "max_tokens": 128
  }'
```

The `:adapter0` suffix is SGLang's convention for attaching a named LoRA adapter — the `:` is why `served_model_name` can't contain a colon (§2.3's assertion).

### 7.1 Step A — HTTP → TokenizerManager

FastAPI's `/v1/chat/completions` handler transforms the OpenAI-shaped payload into SGLang's internal <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/io_struct.py#L134" class="sym-link" target="_blank" rel="noopener noreferrer"><code>GenerateReqInput</code></a> and calls `tokenizer_manager.generate_request(obj)`.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> tokenizer_manager.py:515 — generate_request (entry) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py#L515" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
async def generate_request(
    self,
    obj: Union[GenerateReqInput, EmbeddingReqInput],
    request: Optional[fastapi.Request] = None,
):
    ...
    obj.normalize_batch_and_arguments()
    ...
    # LoRA name → ID
    if self.server_args.enable_lora and obj.lora_path:
        obj.lora_id = await self.lora_registry.acquire(obj.lora_path)
    ...
    # Tokenize
    obj = await self._tokenize_one_request(obj)
    ...
    # Ship to scheduler
    self._send_one_request(obj, state, created_time)
```

`normalize_batch_and_arguments` handles edge cases (single vs batch of requests, normalizing `lora_path` to a list, fixing `sampling_params`, etc.). `lora_registry.acquire("adapter0")` returns the UUID and increments its in-flight counter. `_tokenize_one_request` turns text into `input_ids`. `_send_one_request` pushes a <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/io_struct.py#L694" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TokenizedGenerateReqInput</code></a> onto the ZMQ socket.

### 7.2 Step B — ZMQ → Scheduler.recv_requests

The scheduler's event loop (§4.4) calls `recv_requests()` at the top of every iteration. Non-blocking NOWAIT drain:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> scheduler.py:1506 — recv_requests (excerpt) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L1506" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def recv_requests(self) -> List[Any]:
    """Receive requests from the tokenizer manager over ZMQ."""
    if self.attn_tp_rank == 0 and self.pp_rank == 0:
        recv_reqs = []
        while True:
            try:
                recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            recv_reqs.append(recv_req)
    else:
        recv_reqs = None
    # Then tp-broadcast to other ranks:
    if self.tp_size > 1:
        work = broadcast_pyobj(recv_reqs, src=attn_tp_rank_0, group=self.tp_cpu_group)
        recv_reqs = work
    return recv_reqs
```

Only attention-TP rank 0 reads from the socket; the list is broadcast to the other TP ranks over the CPU process group (PyTorch Gloo backend, not NCCL — see §2.8), so all ranks process the same set of requests at the same step.

`process_input_requests(recv_reqs)` then classifies each message by type (<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/io_struct.py#L694" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TokenizedGenerateReqInput</code></a> → new generation; <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/io_struct.py#L1593" class="sym-link" target="_blank" rel="noopener noreferrer"><code>AbortReq</code></a> → cancellation; `LoadLoRAAdapterReq` → dynamic adapter load; etc.) and dispatches to the appropriate handler. For a new generation, it creates a <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py#L558" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Req</code></a> object with `lora_id` attached and appends it to `waiting_queue`.

### 7.3 Step C — <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L2302" class="sym-link" target="_blank" rel="noopener noreferrer"><code>get_next_batch_to_run</code></a> → prefill vs decode

This function decides what to run this iteration. Big picture:

1. Merge the last-iteration prefill batch into `running_batch` (they become decode-ready after one forward).
2. Call `get_new_batch_prefill()` to try to build a new prefill batch from `waiting_queue`, subject to radix-cache hits, KV budget, and LoRA constraints (max 1 adapter's worth of LoRA memory available per batch).
3. If there's a prefill batch, return it (prefill pre-empts decode). Otherwise return the `running_batch` for a decode step.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> scheduler.py:2302 — get_next_batch_to_run (top) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L2302" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
    self._abort_on_waiting_timeout()
    self._abort_on_running_timeout()
    ...
    # Merge the prefill batch into the running batch
    chunked_req_to_exclude = set()
    ...
    if (not self.enable_hisparse
        and self.last_batch
        and self.last_batch.forward_mode.is_extend()):
        ...
        if not self.last_batch.is_empty():
            if self.running_batch.is_empty():
                self.running_batch = self.last_batch
            else:
                self.running_batch.merge_batch(self.last_batch)
    ...
    if self.dllm_config is not None:
        new_batch = self.get_new_batch_dllm()
    else:
        new_batch = self.get_new_batch_prefill()
    ...
    if new_batch is not None:
        return new_batch
    return self.running_batch
```

<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L2417" class="sym-link" target="_blank" rel="noopener noreferrer"><code>get_new_batch_prefill</code></a> is where the radix-cache prefix matcher runs. For each waiting request it walks the tree: tokens already cached in GPU become `prefix_indices` (reuse KV), only the uncached suffix becomes "extend" tokens (compute new KV). With LoRA+radix (§6.10), the prefix walk is gated on both token equality and `lora_id` equality.

### 7.4 Step D — <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L2754" class="sym-link" target="_blank" rel="noopener noreferrer"><code>run_batch</code></a> → forward

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> scheduler.py:2754 — run_batch (excerpt) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L2754" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def run_batch(
    self, batch: ScheduleBatch, pp_proxy_tensors=None,
) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
    """Run a batch."""
    self.forward_ct += 1
    ...
    # Run forward
    if self.is_generation:
        if self.spec_algorithm.is_none() or self.enable_overlap:
            worker_batch_or_batch = batch.get_model_worker_batch()
        else:
            worker_batch_or_batch = batch
        if self.enable_overlap:
            ...
            with self.forward_stream_ctx, self.record_bubble_metrics(batch):
                self.forward_stream.wait_stream(self.schedule_stream)
                self.future_map.resolve_future(model_worker_batch)
                batch_result = self.model_worker.forward_batch_generation(model_worker_batch)
            ...
```

`batch.get_model_worker_batch()` flattens the <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py#L1321" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ScheduleBatch</code></a> into a <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py#L2585" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelWorkerBatch</code></a> — pure tensors, no Python objects — ready to be handed to the GPU. Key fields the model runner will see: `input_ids`, `positions`, `extend_seq_lens`, `seq_lens`, `req_pool_indices`, `out_cache_loc`, `sampling_info`, **`lora_ids`**.

### 7.5 Step E — <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L2791" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelRunner.forward_extend</code></a> / `forward_decode`

<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tp_worker.py#L443" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TpModelWorker.forward_batch_generation</code></a> routes to <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L2882" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelRunner.forward</code></a>, which branches on `forward_mode`:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> model_runner.py:2882 — forward dispatch <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L2882" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def forward(
    self, forward_batch: ForwardBatch,
    skip_attn_backend_init=False, pp_proxy_tensors=None,
    reinit_attn_backend=False, split_forward_count=1,
) -> ModelRunnerOutput:
    self.forward_pass_id += 1
    with get_global_expert_distribution_recorder().with_forward_pass(
        self.forward_pass_id, forward_batch,
    ) as recorder_outputs:
        output = self._forward_raw(
            forward_batch, skip_attn_backend_init, pp_proxy_tensors,
            reinit_attn_backend, split_forward_count,
        )
        ...
    return output
```

The pre-hook path in `_forward_raw` populates LoRA state right before forward runs:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> model_runner.py:2470-2480 — LoRA + attn metadata hook <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L2470" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
if lora_ids is not None:
    self.lora_manager.prepare_lora_batch(forward_batch)

self.attn_backend.init_forward_metadata(forward_batch)
```

<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L300" class="sym-link" target="_blank" rel="noopener noreferrer"><code>prepare_lora_batch</code></a> (§6.7) uploads the per-batch `weight_indices`, `lora_ranks`, `scalings` tensors and sets `batch_info.has_active_lora`. `init_forward_metadata` builds the attention backend's per-batch state (position tables, page tables, varlen indices for FA3).

Then `self.model.forward(input_ids, positions, forward_batch)` runs the actual network. For Qwen3-30B-A3B-Instruct-2507, that's 48 iterations of the decoder layer:

```python
for layer in model.layers:             # 48 layers
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)

    # === attention ===
    qkv, _ = layer.self_attn.qkv_proj(hidden_states)     # QKVParallelLinearWithLoRA fires here
    q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)
    q = layer.self_attn.q_norm(q.view(..., head_dim))    # per-head Q RMSNorm
    k = layer.self_attn.k_norm(k.view(..., head_dim))    # per-head K RMSNorm
    q, k = rotary_emb(q, k, positions)
    attn_out = radix_attention(q, k, v, forward_batch)   # reads+writes KV pool
    hidden_states, _ = layer.self_attn.o_proj(attn_out)  # RowParallelLinearWithLoRA fires here

    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)

    # === MoE ===
    router_logits = layer.mlp.gate(hidden_states)
    topk_out = layer.mlp.topk(router_logits)             # top-8 of 128
    hidden_states = layer.mlp.experts(hidden_states, topk_out)  # FusedMoEWithLoRA fires here

    hidden_states = residual + hidden_states

hidden_states = model.norm(hidden_states)
logits = lm_head(hidden_states)                          # ParallelLMHeadWithLoRA if enabled
```

Inside each `*WithLoRA` wrapper's `forward`, the base projection runs first, then (if `self.set_lora` and `batch_info.has_active_lora`) the LoRA delta is added in-place. The backend kernels read from `batch_info` and the preloaded A/B buffers.

### 7.6 Step F — sampling and result

The logits from `lm_head` go through `self.sampler.sample(...)` (§5.1), which applies temperature/top-p/top-k/grammar constraints and returns `next_token_ids`. <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L2937" class="sym-link" target="_blank" rel="noopener noreferrer"><code>process_batch_result</code></a> updates each <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py#L558" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Req</code></a>'s state:

- Append the sampled token to the request's `output_ids`.
- Write the new K/V into the KV pool at `out_cache_loc`.
- If the request is done (hit EOS, max_tokens, stop sequence), mark `finished()` and release its KV slots.
- Enqueue a `BatchTokenIDOut` to the detokenizer.

### 7.7 Step G — DetokenizerManager → user

The <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/detokenizer_manager.py#L73" class="sym-link" target="_blank" rel="noopener noreferrer"><code>DetokenizerManager</code></a> subprocess picks up the `BatchTokenIDOut`, looks up each request's token stream so far, calls the tokenizer's `decode()` (or streaming incremental decode), and sends a `BatchStrOut` back to the TokenizerManager via ZMQ. The TokenizerManager resolves the awaiting `asyncio.Event` in `state`, the FastAPI handler yields the result, and the HTTP response streams out.

Once the last token of this request has been sent, the TokenizerManager calls `lora_registry.release(lora_id)`, which decrements the counter — freeing the adapter to be unloaded if requested.

---

<p class="bridge" markdown="span">*The walkthrough so far assumed `--tp 4` and glossed over exactly what "tensor parallel" means. The next two Parts unpack every parallelism dimension SGLang supports, starting with the two most load-bearing for Qwen3-MoE.*</p>

## 8 · Parallelism — TP & EP in depth
{: #parallelism }

This part pulls together everything about how SGLang partitions a model across GPUs. It's deliberately at the end because it cross-cuts parts 1, 5, and 6 — you'll see forward-pointers to sections you've already read.

Two orthogonal axes of parallelism are relevant for Qwen3 and DeepSeek-class MoE models:

| Axis                     | What it partitions                                                                  | What communication it adds                                               | CLI flag                               |
|--------------------------|-------------------------------------------------------------------------------------|--------------------------------------------------------------------------|----------------------------------------|
| **Tensor parallel (TP)** | Each dense matrix is sliced across N GPUs (row-parallel or column-parallel).        | One all-reduce per dense block that ends with a row-parallel projection. | `--tp N`                               |
| **Expert parallel (EP)** | MoE experts are partitioned across N GPUs; each GPU owns `num_experts / N` experts. | Two all-to-all calls per MoE block (dispatch, combine).                  | `--moe-a2a-backend deepep --ep-size N` |

Pipeline parallel (`--pp`), context-parallel (`--cp`), and data-parallel (`--dp`) also exist but aren't needed for a single-node Qwen3 run. Everything below assumes `--pp 1 --dp 1 --cp 1`.

### 8.1 TP: three patterns & the Qwen3 attention flow
{: #par-tp-story }

TP for transformer inference is, under the hood, an application of three very old tricks for partitioning matrix products — published in the original Megatron-LM paper (<a href="https://arxiv.org/abs/1909.08053" target="_blank" rel="noopener noreferrer">Shoeybi et al., 2019</a>). For an `output = input @ W`:

| Pattern                                                                                                                                                                                                                  | How `W` is sliced                                    | Per-rank compute                                                                            | Collective at end                                                                                                             |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|---------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| **Replicated**                                                                                                                                                                                                           | Not sliced. Every rank holds the full `W`.           | Full `input @ W`.                                                                           | None.                                                                                                                         |
| **Column-parallel** (<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L286" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ColumnParallelLinear</code></a>) | Split along the output dim: `W = [W₁, W₂, ..., Wₚ]`. | `input @ Wᵢ` → partial output of size `output_size / tp_size`.                              | Optional `all_gather` if downstream needs full output; otherwise none (leave partitioned for a following row-parallel layer). |
| **Row-parallel** (<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L1312" class="sym-link" target="_blank" rel="noopener noreferrer"><code>RowParallelLinear</code></a>)      | Split along the input dim: `W = [W₁; W₂; ...; Wₚ]ᵀ`. | `inputᵢ @ Wᵢ` → partial **sum** of size `output_size` (each rank has partial contribution). | `all_reduce` to sum partials across ranks.                                                                                    |

The critical fact: when a column-parallel layer feeds directly into a row-parallel layer, you save the `all_gather` at the column-parallel output (the row-parallel layer happens to need exactly the per-partition slice that column-parallel produced). The "column → row" pair results in **one all-reduce** total across the two matrix multiplies.

Here's the column-parallel constructor, to make it concrete:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> layers/linear.py:286-345 — ColumnParallelLinear docstring + init <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L286" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
class ColumnParallelLinear(LinearBase):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        ...
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        ...
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        ...
    ):
        ...
        if tp_rank is None:
            tp_rank = get_tensor_model_parallel_rank()
        if tp_size is None:
            tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.output_size_per_partition = divide(self.output_size, tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, tp_size) for output_size in self.output_sizes
            ]
```

And here's the row-parallel forward, which is where the actual collective gets invoked:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> layers/linear.py:1492-1525 — RowParallelLinear.forward <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L1492" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def forward(self, input_, skip_all_reduce=False):
    if self.input_is_parallel:
        input_parallel = input_
    else:
        splitted_input = split_tensor_along_last_dim(
            input_, num_partitions=self.tp_size
        )
        input_parallel = splitted_input[self.tp_rank].contiguous()

    # Matrix multiply.
    assert self.quant_method is not None
    # Only fuse bias add into GEMM for rank 0 (this ensures that
    # bias will not get added more than once in TP>1 case)
    bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
    if self.use_dp_attention_reduce:
        symm_ctx = use_symmetric_memory(get_attention_tp_group())
    else:
        symm_ctx = use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        )
    with symm_ctx:
        output_parallel = self.quant_method.apply(self, input_parallel, bias=bias_)

    if self.reduce_results and self.tp_size > 1 and not skip_all_reduce:
        if self.use_dp_attention_reduce:
            output = get_attention_tp_group().all_reduce(output_parallel)
        else:
            output = tensor_model_parallel_all_reduce(output_parallel)
    else:
        output = output_parallel
    ...
    return output, output_bias
```

Three details worth pointing out in that code:

1. **Bias on rank 0 only.** If every rank added the bias after their partial GEMM, and then we all-reduced, the bias would be summed `tp_size` times. So the code sets `bias_` to `None` on ranks \> 0 and only rank 0's GEMM includes the bias.
2. **Symmetric memory.** `use_symmetric_memory` enables NCCL's "symmetric memory" allreduce path on machines where the allocator can guarantee symmetric allocations (same virtual address on all ranks). This is significantly faster than regular ring-allreduce on NVLink-connected GPUs. If symmetry can't be verified (`is_allocation_symmetric()` is False), the context manager falls back to standard NCCL.
3. **`reduce_results` flag.** Some attention backends (those that internally handle the allreduce as part of a fused kernel) pass `reduce_results=False` to the o_proj to skip the explicit all-reduce here. You rarely set it yourself.

The top-level all-reduce helper is a one-liner dispatch:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> distributed/communication_op.py:16-19 — tensor_model_parallel_all_reduce <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/communication_op.py#L16" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)
```

#### Qwen3 attention flow at TP=4, end-to-end

Here's how one attention block's TP flow plays out for Qwen3, drawing on §5.6:

```python
# Entering the block with `x` (full hidden size 2048), replicated on all 4 ranks:
x_full = ...  # shape (T, 2048), same on all ranks

# Step 1. RMSNorm — elementwise, no comms.
h = input_layernorm(x_full)   # (T, 2048) — same on all ranks

# Step 2. QKV projection — column-parallel; each rank computes its partition.
qkv_partial = qkv_proj.weight @ h.T
# On rank 0: qkv_proj.weight = [q0, k0, v0] shape (1024+128+128, 2048)
#   result: (T, 1280) containing q[heads 0-7] + k[head 0] + v[head 0]
# On rank 1: q[heads 8-15], k[head 1], v[head 1] ... etc.
# NO COMMUNICATION. Each rank has its own 1/4 slice.

q_part, k_part, v_part = split_qkv(qkv_partial)

# Step 3. Per-head q_norm / k_norm — elementwise along head_dim, no comms.
q_part = q_norm(q_part)   # operates per-head; same on all ranks for their heads
k_part = k_norm(k_part)

# Step 4. RoPE — elementwise, no comms.
q_part, k_part = rotary_emb(q_part, k_part, positions)

# Step 5. Attention compute — local to each rank's heads.
#   Each rank stores its own KV cache slice (1 KV head per rank).
#   No cross-rank communication needed inside attention.
attn_out_partial = radix_attention(q_part, k_part, v_part, forward_batch)
# Shape: (T, 8 * 128) = (T, 1024)  — partial output, this rank's Q-head slice.

# Step 6. o_proj — row-parallel.
#   o_proj.weight per rank: (2048, 1024). Input is (T, 1024).
#   Per-rank product: (T, 2048) — but it's a PARTIAL SUM, not the full o_proj output.
o_partial = o_proj_weight @ attn_out_partial.T   # no bias on ranks > 0
# NOW the all-reduce happens:
o_full = tensor_model_parallel_all_reduce(o_partial)   # ONE NCCL all_reduce, shape (T, 2048)

# Step 7. Residual. Elementwise, no comms.
x_full = x_full + o_full
```

One NCCL all-reduce per attention block. The MoE block has the same pattern — column-parallel on gate_up, row-parallel on down_proj, one all-reduce at the end:

```python
# After post_attention_layernorm:
h = post_norm(x_full)   # (T, 2048), same on all ranks (replicated)

# Router — REPLICATED (gate.weight is small, each rank runs it fully).
router_logits = gate_weight @ h.T   # (T, 128) — same on all ranks
topk_ids, topk_probs = top_k_softmax(router_logits)

# Experts — column-parallel on each expert's gate_up, row-parallel on down.
# For Qwen3 TP=4: each rank holds expert i's (gate_proj, up_proj) shape (384, 2048)  (1/4 of intermediate)
#                 and expert i's down_proj shape (2048, 192)  (1/4 of intermediate row).
# Fused MoE runs all 128 experts in a single grouped Triton kernel per rank.
moe_out_partial = fused_moe(h, topk_ids, topk_probs)   # (T, 2048), partial sum
# Final all-reduce on the MoE block output:
moe_out_full = tensor_model_parallel_all_reduce(moe_out_partial)   # ONE all_reduce
x_full = x_full + moe_out_full
```

<div class="callout info" markdown="1">

#### Why is the router replicated rather than partitioned?

The router's weight is tiny: `(128, 2048)` = 256 K parameters = 512 KB in bf16. Partitioning it across 4 ranks would save 384 KB but require an all-gather on the logits before the top-k, which costs more than the memory saved. Small "decision" networks like the router are pretty much always left replicated. Qwen3 declares this with <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L191" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ReplicatedLinear</code></a> (see <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L191" target="_blank" rel="noopener noreferrer">linear.py:191</a>).

</div>

### 8.2 Counting NCCL calls per layer (TP=4)
{: #par-tp-count }

Per decoder layer, for Qwen3-30B-A3B-Instruct-2507 with no DP-attention and no fused RMSNorm+allreduce:

| Sub-block                 | Collective       | Tensor size (bf16, per rank, T tokens) |
|---------------------------|------------------|----------------------------------------|
| attention → o_proj output | 1× all_reduce    | T × 2048 × 2 B                         |
| MoE → down_proj output    | 1× all_reduce    | T × 2048 × 2 B                         |
| **Per layer**             | **2 all_reduce** | T × 8 KB total                         |

Across all 48 layers: **96 all_reduce calls per forward pass**, each moving ~`T × 2048 × 2 B`. For a decode batch of `T=64` tokens, that's 256 KB × 96 = 24 MB total per forward — a few milliseconds on NVLink-4 H200.

With `--enable-flashinfer-allreduce-fusion` (FlashInfer's fused allreduce+RMSNorm kernel), the post-attention and post-MoE RMSNorms fuse *into* their preceding all-reduces, turning the next layer's input RMSNorm into a no-op. That shaves a small but non-trivial number of kernel launches per forward. Enabled by default when available.

With `--enable-dp-attention`, each request is assigned to one DP rank's attention group (a smaller subset of the TP group), reducing collective sizes but requiring an all-gather at the dense-MoE boundary. This is a common choice for serving pipelines with low batch-size variance; it trades one big collective for several smaller ones that parallelize better.

### 8.3 EP: why a second parallelism dimension exists
{: #par-ep-why }

TP handles "matrix is too big to fit on one GPU." EP handles a different problem: **"there are too many experts, and replicating all of them across TP ranks wastes memory."**

Consider the memory math for the MoE weights of Qwen3 vs DeepSeek-V3:

| Model         | Experts        | Intermediate per expert | Hidden | MoE params per layer | Per rank, TP=8, bf16 |
|---------------|----------------|-------------------------|--------|----------------------|----------------------|
| Qwen3-30B-A3B | 128            | 768                     | 2048   | ~0.6 B               | ~144 MB              |
| DeepSeek-V3   | 256 + 1 shared | 2048                    | 7168   | ~11.3 B              | ~2.7 GB              |
| Kimi K2       | 384 + 1 shared | 2048                    | 7168   | ~17 B                | ~4 GB                |

For Qwen3 at TP=4, MoE takes ~288 MB per rank per layer (§5.7). For DeepSeek-V3 at TP=8, MoE takes 2.7 GB per rank per layer. DeepSeek-V3 has 61 layers. That's ~165 GB per rank just for MoE weights — it doesn't fit on even an H200 (141 GB HBM).

EP fixes this by *partitioning experts themselves* instead of slicing every expert's weight. With `moe_ep_size = 8`, rank 0 owns experts \[0..31\], rank 1 owns \[32..63\], and so on. Each rank's memory is now `256/8 × 11.3 B / 256 ≈ 1.4 GB` per layer — a clean 8× reduction.

The cost: a token that needs to execute expert 150 doesn't live on the same rank as its expert. Before the MoE compute can run, every rank's tokens have to be shuffled to the rank owning their assigned experts (dispatch); after compute, results have to be shuffled back (combine). Two all-to-all collectives per MoE block, instead of one all-reduce.

<div class="callout motiv" markdown="1">

#### When does EP win?

EP wins when the expert weights are large enough that the saved HBM footprint (allowing larger KV cache, longer context, or more concurrent requests) outweighs the added A2A latency. That crossover happens for models where `num_experts × expert_size` is a large fraction of the model — basically every DeepSeek-style MoE. For Qwen3-30B-A3B (small experts, lots of them), EP typically loses to pure TP on a single node, which is why the example commands we've been using don't specify `--ep-size`.

</div>

### 8.4 How EP & TP compose for MoE
{: #par-ep-compose }

EP doesn't replace TP; it operates alongside it. In SGLang's model, the world is divided into orthogonal groups for the MoE layer:

```text
world_size = moe_tp_size × moe_ep_size           (for MoE layers)
world_size = tp_size                              (for attention layers, since EP is MoE-only)
```

So if you run `--tp 8 --ep-size 4` on an 8-GPU node:

- Attention: TP=8 (every rank has 1/8 of each attention matrix)
- MoE: moe_tp=2 × moe_ep=4. Each rank has 1/2 of the intermediate dim of 1/4 of the experts.

FusedMoE's constructor encodes exactly this:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> layers/moe/fused_moe_triton/layer.py:197-215 — FusedMoE EP fields <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L197" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
self.moe_ep_size = get_moe_expert_parallel_world_size()
self.moe_ep_rank = get_moe_expert_parallel_rank()
self.moe_tp_size = get_moe_tensor_parallel_world_size()
self.moe_tp_rank = get_moe_tensor_parallel_rank()

# DeepEP: each rank has its own shared expert slot, so total shared
# weight slots = num_fused_shared_experts * ep_size.
# AMD/Standard: shared experts are global, slots = num_fused_shared_experts.
if num_fused_shared_experts > 0 and is_deepep_class_backend():
    num_shared_slots = num_fused_shared_experts * self.moe_ep_size
else:
    num_shared_slots = num_fused_shared_experts

assert (num_experts - num_shared_slots) % self.moe_ep_size == 0
self._num_global_routed = num_experts - num_shared_slots
self._num_local_routed = self._num_global_routed // self.moe_ep_size
self.num_local_experts = self._num_local_routed + num_fused_shared_experts
self._has_fused_shared = num_fused_shared_experts > 0
```

So for DeepSeek-V3 at `moe_ep_size=4`: `_num_global_routed = 256`, `_num_local_routed = 64`. This rank owns 64 experts out of 256.

The weight-loading remap for EP is a clean bit of arithmetic:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> layers/moe/fused_moe_triton/layer.py:573-581 — _map_global_expert_id_to_local_expert_id <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L573" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
    start_idx = self.moe_ep_rank * self._num_local_routed
    end_idx = start_idx + self._num_local_routed
    if start_idx <= expert_id < end_idx:
        return expert_id - start_idx
    elif self._has_fused_shared and expert_id >= self._num_global_routed:
        return expert_id - self._num_global_routed + self._num_local_routed
    else:
        return -1
```

At weight-load time (§5.7), when the loader hands a global-numbered expert tensor (e.g. expert 150) to the weight_loader, this function returns the local index (e.g. 22 if this rank owns experts \[128..191\]) — or `-1` meaning "not my expert, skip this tensor." The tensor gets dropped and its bytes never land on this GPU.

### 8.5 DeepEP — the dispatcher
{: #par-deepep }

DeepEP (<a href="https://github.com/deepseek-ai/DeepEP" target="_blank" rel="noopener noreferrer">github.com/deepseek-ai/DeepEP</a>) is DeepSeek's open-source MoE all-to-all library. SGLang wraps it in <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L744" class="sym-link" target="_blank" rel="noopener noreferrer"><code>DeepEPDispatcher</code></a>, a single class that chooses between two very different modes at dispatch time.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> layers/moe/token_dispatcher/deepep.py:744-785 — DeepEPDispatcher __init__ <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L744" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
class DeepEPDispatcher(BaseDispatcher):
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        super().__init__()
        self.deepep_mode = deepep_mode

        common_kwargs = dict(
            group=group,
            router_topk=router_topk,
            permute_fusion=permute_fusion,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            deepep_mode=deepep_mode,
        )

        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher = _DeepEPDispatcherImplLowLatency(
                return_recv_hook=return_recv_hook,
                **common_kwargs,
            )
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher = _DeepEPDispatcherImplNormal(
                async_finish=async_finish,
                **common_kwargs,
            )

        self._stage = _Stage.INITIAL
        self._deepep_dispatch_hooks = DeepEPPDispatchHooks()
```

The mode selection logic is literally "is this a prefill or a decode?":

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> layers/moe/utils.py:116-144 — DeepEPMode <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/utils.py#L116" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
class DeepEPMode(Enum):

    NORMAL = "normal"
    LOW_LATENCY = "low_latency"
    AUTO = "auto"

    def enable_normal(self) -> bool:
        return self in [DeepEPMode.NORMAL, DeepEPMode.AUTO]

    def enable_low_latency(self) -> bool:
        return self in [DeepEPMode.LOW_LATENCY, DeepEPMode.AUTO]

    def resolve(self, is_extend_in_batch: bool) -> DeepEPMode:
        if self != DeepEPMode.AUTO:
            return self
        if is_extend_in_batch:
            return DeepEPMode.NORMAL
        else:
            return DeepEPMode.LOW_LATENCY
```

In `AUTO` mode (the default), both dispatchers are constructed, and `_get_impl()` picks the right one per batch via `get_is_extend_in_batch()`. **Prefill batches use Normal**; **decode batches use Low-Latency**. Why? Prefill sees large token counts (hundreds or thousands per request) with throughput-sensitive comm; decode sees tiny token counts (1-8 per request) with latency-sensitive comm. The two modes use genuinely different CUDA kernels optimized for those two regimes.

The top-level `dispatch` / `combine` API is split into `_a` / `_b` halves:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> layers/moe/token_dispatcher/deepep.py:806-845 — DeepEPDispatcher.dispatch & combine <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L806" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def dispatch(
    self,
    hidden_states: torch.Tensor,
    topk_output: TopKOutput,
) -> DispatchOutput:
    self.dispatch_a(hidden_states, topk_output)
    if self._deepep_dispatch_hooks is not None:
        self._deepep_dispatch_hooks(self)
    ret = self.dispatch_b()
    return ret

def dispatch_a(
    self,
    hidden_states: torch.Tensor,
    topk_output: TopKOutput,
):
    self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
    inner_state = self._get_impl().dispatch_a(
        hidden_states=hidden_states,
        topk_output=topk_output,
    )
    self._dispatch_intermediate_state = inner_state

def dispatch_b(self):
    self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
    inner_state = self._dispatch_intermediate_state
    del self._dispatch_intermediate_state
    return self._get_impl().dispatch_b(*inner_state)
```

<div class="callout motiv" markdown="1">

#### Why two-phase?

`dispatch_a` *launches* the all-to-all on the comm stream and returns immediately. `dispatch_b` *waits* for it and extracts the permuted tensors. This split gives SGLang's two-batch overlap (TBO) mode a hook point: while batch A's dispatch is in-flight, batch B's pre-dispatch compute can run on the default stream. Same for combine. Explained in detail in the <a href="https://lmsys.org/blog/2025-05-05-large-scale-ep/" target="_blank" rel="noopener noreferrer">SGLang large-scale EP blog post</a>.

</div>

### 8.6 Normal mode — prefill path
{: #par-deepep-normal }

Normal mode is DeepEP's throughput-optimized path. It's used for prefill, where latency per request is dominated by token count, not collective round-trip time. Here's `dispatch_a`:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> layers/moe/token_dispatcher/deepep.py:389-410 — Normal dispatch_a <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L389" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def dispatch_a(
    self,
    hidden_states: torch.Tensor,
    topk_output: TopKOutput,
):
    topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
    topk_ids = topk_ids.to(torch.int64)
    if (
        deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
        and not get_moe_runner_backend().is_cutlass()
        and not envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
    ):
        # TODO hard code 128 block quant, use fp8 communication
        hidden_states = sglang_per_token_group_quant_fp8(
            hidden_states,
            128,
            column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
    previous_event = Buffer.capture() if self.async_finish else None
    return hidden_states, topk_ids, topk_weights, previous_event
```

<div class="callout info" markdown="1">

#### FP8 compression at the comm boundary

When DeepGEMM is enabled (the default on modern NVIDIA), hidden states are quantized to FP8 with per-token-group scales *before* the all-to-all. This halves the A2A payload compared to bf16 and, critically, feeds directly into the FP8-native DeepGEMM MoE kernel on the receiver side. No dequant needed until the expert's output. Group size is hardcoded 128 because that's DeepGEMM's optimal tile.

</div>

Then `dispatch_b` → `_dispatch_core`:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> layers/moe/token_dispatcher/deepep.py:435-495 — Normal _dispatch_core <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L435" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def _dispatch_core(
    self,
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    previous_event,
):
    buffer = self._get_buffer()
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        previous_event,
    ) = buffer.get_dispatch_layout(
        topk_ids,
        self.num_experts,
        previous_event=previous_event,
        async_finish=self.async_finish,
        allocate_on_comm_stream=previous_event is not None,
    )

    _deepep_precompile_tp_barrier()
    (
        recv_x,
        recv_topk_ids,
        recv_topk_weights,
        num_recv_tokens_per_expert,
        self.handle,
        event,
    ) = buffer.dispatch(
        x,
        topk_idx=topk_ids,
        topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=previous_event,
        async_finish=self.async_finish,
        allocate_on_comm_stream=(previous_event is not None) and self.async_finish,
        expert_alignment=128 if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM else 1,
        config=DeepEPConfig.get_instance().normal_dispatch_config,
    )
    ...
    return (
        recv_x,
        recv_topk_ids,
        recv_topk_weights,
        num_recv_tokens_per_expert,
        event,
    )
```

Two DeepEP library calls:

1. `buffer.get_dispatch_layout(topk_ids, num_experts)` — computes "how many tokens does this rank send to each other rank and to each expert," fully on-device via a CUDA kernel over `topk_ids`. No Python, no blocking comm.
2. `buffer.dispatch(...)` — the actual all-to-all. Sends each token's hidden vector to the rank owning that token's selected experts. `expert_alignment=128` pads expert groups to 128 tokens so DeepGEMM's grouped-GEMM kernel can run unmasked.

Normal combine is the symmetric backward path:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> layers/moe/token_dispatcher/deepep.py:519-529 — Normal _combine_core <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L519" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def _combine_core(self, x: torch.Tensor, previous_event):
    buffer = self._get_buffer()
    _deepep_precompile_tp_barrier()
    combined_x, _, event = buffer.combine(
        x,
        self.handle,
        async_finish=self.async_finish,
        previous_event=previous_event,
        allocate_on_comm_stream=previous_event is not None,
        config=DeepEPConfig.get_instance().normal_combine_config,
    )
    return combined_x, event
```

The `self.handle` from dispatch is what threads source/destination metadata through to combine, so the inverse permutation doesn't have to be recomputed.

### 8.7 Low-latency mode — decode path
{: #par-deepep-ll }

Decode only processes 1 token per active request per step — often ≤ 256 tokens in the whole global batch. At those sizes, a ring-based all-to-all has enormous overhead: the ring setup cost dominates the actual payload transfer. DeepEP's *low-latency* mode uses a completely different kernel: direct point-to-point RDMA writes with per-expert token buckets, no bulk ring construction. Conceptually it's like a bunch of small scatter/gathers rather than one big all-to-all.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> layers/moe/token_dispatcher/deepep.py:614-648 — Low-latency _dispatch_core <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L614" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def _dispatch_core(
    self,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
):
    use_nvfp4 = use_fp8 = False
    input_global_scale = self.quant_config.get("input_global_scale", None)
    if input_global_scale is not None:
        use_nvfp4 = True
    else:
        use_fp8 = True

    buffer = self._get_buffer()
    _deepep_precompile_tp_barrier()
    packed_recv_hidden, self.packed_recv_count, self.handle, event, hook = (
        buffer.low_latency_dispatch(
            hidden_states,
            topk_ids,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            use_fp8=use_fp8,
            **(dict(use_nvfp4=True) if use_nvfp4 else dict()),
            ...
            async_finish=not self.return_recv_hook,
            return_recv_hook=self.return_recv_hook,
            round_scale=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,
            use_ue8m0=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,
        )
    )
    return packed_recv_hidden, self.packed_recv_count, event, hook
```

Key differences vs. normal mode:

- **Output is packed, not per-token.** `packed_recv_hidden` is laid out as `[num_local_experts, max_tokens_per_expert, hidden]`, with unused slots zero-padded. This is *already* in the shape DeepGEMM's masked grouped-GEMM wants, so no post-dispatch permute is needed.
- **FP8 (or NVFP4 on Blackwell) is mandatory**; there's no bf16 variant of the low-latency kernel.
- **Result count via `packed_recv_count`** — a small int tensor saying "expert i received N tokens"; the grouped-GEMM kernel uses this as a mask to skip past padding.
- **Optional `return_recv_hook`.** Instead of an event-based wait, the caller gets a closure it can invoke at the optimal point to overlap the RDMA completion with other compute. Used by two-batch overlap (TBO) mode.

The low-latency combine goes further still, with first-class overlap hooks:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> layers/moe/token_dispatcher/deepep.py:676-720 — Low-latency _combine_core <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L676" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def _combine_core(
    self,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
):
    buffer = self._get_buffer()
    overlap_args = self.overlap_args
    meta_overlap_args = self.meta_overlap_args

    ctx = nullcontext()
    if overlap_args is not None:
        overlap_args.stream.wait_event(overlap_args.wait_event)
        ctx = torch.cuda.stream(overlap_args.stream)

        if is_blackwell():
            overlap_args_dict = dict(
                overlap=overlap_args.overlap,
                src_signals=overlap_args.signal,
                src_signal_expect_value=overlap_args.threshold,
            )
        else:
            overlap_args_dict = dict(
                overlap=overlap_args.overlap,
                packed_recv_count=self.packed_recv_count,
                comp_signal=overlap_args.signal,
                block_m=meta_overlap_args["block_m"],
                threshold=meta_overlap_args["threshold"],
                num_sms=overlap_args.num_sms,
            )
    else:
        overlap_args_dict = {}

    with ctx:
        _deepep_precompile_tp_barrier()
        combined_hidden_states, event, hook = buffer.low_latency_combine(
            x=hidden_states,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            handle=self.handle,
            async_finish=not self.return_recv_hook,
            return_recv_hook=self.return_recv_hook,
            **overlap_args_dict,
        )
```

The `overlap_args` branch enables **computation-communication fusion**: the MoE's down_proj GEMM and the combine all-to-all run simultaneously on different SMs of the same GPU, with a signaling scheme (GPU-side atomic counter on Hopper; dedicated signaling SMs on Blackwell) that tells the combine kernel "start fetching expert *i*'s output now — the down_proj block for *i* just finished." Net effect: combine latency is almost entirely hidden behind GEMM compute.

### 8.8 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L145" class="sym-link" target="_blank" rel="noopener noreferrer"><code>DeepEPBuffer</code></a> & NVLink/RDMA sizing
{: #par-deepep-buffer }

Both modes share a persistent `Buffer` object — DeepEP's core state holding pinned/registered memory for NVLink and RDMA transports. It's allocated once per process and reused across every MoE layer (there are typically dozens).

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> layers/moe/token_dispatcher/deepep.py:152-241 — DeepEPBuffer.get_deepep_buffer <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L152" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
@classmethod
def get_deepep_buffer(
    cls,
    group: dist.ProcessGroup,
    hidden_size: int,
    param_bytes: int,
    deepep_mode: DeepEPMode,
    num_max_dispatch_tokens_per_rank: int = -1,
    num_experts: int = -1,
):
    if cls._buffer is not None:
        return cls._buffer

    cls._hidden_size = hidden_size
    cls._num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
    cls._num_experts = num_experts

    num_nvl_bytes, num_rdma_bytes = 0, 0
    if deepep_mode.enable_normal():
        hidden_bytes = hidden_size * param_bytes
        for config in (
            DeepEPConfig.get_instance().normal_dispatch_config
            or Buffer.get_dispatch_config(group.size()),
            DeepEPConfig.get_instance().normal_combine_config
            or Buffer.get_combine_config(group.size()),
        ):
            num_nvl_bytes = max(
                config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
                num_nvl_bytes,
            )
            num_rdma_bytes = max(
                config.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
                num_rdma_bytes,
            )
    if deepep_mode.enable_low_latency():
        assert num_max_dispatch_tokens_per_rank != -1
        assert num_experts != -1 and num_experts % group.size() == 0
        num_rdma_bytes = max(
            Buffer.get_low_latency_rdma_size_hint(
                num_max_dispatch_tokens_per_rank,
                hidden_size,
                group.size(),
                num_experts,
            ),
            num_rdma_bytes,
        )

    # We should calculate num_qps_per_rank consistently with DeepEP's test script logic:
    if deepep_mode == DeepEPMode.NORMAL:
        num_qps_per_rank = DeepEPConfig.get_instance().num_sms
    elif deepep_mode == DeepEPMode.LOW_LATENCY:
        num_qps_per_rank = num_experts // group.size()
    elif deepep_mode == DeepEPMode.AUTO:
        num_qps_per_rank = max(
            DeepEPConfig.get_instance().num_sms, num_experts // group.size()
        )
    else:
        raise NotImplementedError

    ...

    cls._buffer = Buffer(
        group,
        num_nvl_bytes,
        num_rdma_bytes,
        low_latency_mode=deepep_mode.enable_low_latency(),
        num_qps_per_rank=num_qps_per_rank,
        allow_mnnvl=True,
    )
    return cls._buffer
```

Three quantities get computed here, and each matters:

- **`num_nvl_bytes`** — pinned GPU memory for NVLink transport inside a node. Normal mode only; low-latency uses RDMA even within a node.
- **`num_rdma_bytes`** — RDMA staging buffers. For internode topologies these are registered with the NIC; for intranode low-latency they're still used because DeepEP's LL path uses GPU-registered memory for RDMA-style point-to-point.
- **`num_qps_per_rank`** — how many RDMA queue pairs (independent concurrent message streams) each rank opens. Normal mode matches it to `num_sms` (parallelize comm over SMs); low-latency matches it to `num_experts / ep_size` (one QP per locally-owned expert so each expert's receive bucket gets its own stream).

<div class="callout info" markdown="1">

#### Why the SM count matters for normal-mode DeepEP

Normal-mode DeepEP uses *dedicated SMs* to run its dispatch/combine kernels. The SMs spent on comm are subtracted from what's available for the MoE grouped GEMM. `--deepep-config.num_sms=24` is the default on H100/H200 (out of 132 SMs on H200); reducing it gives the GEMM more SMs but makes comm slower. The warning at the end of `get_deepep_buffer` fires if you configure fewer than half the SMs — that's almost always suboptimal.

</div>

Here's a simplified end-to-end picture for a DeepEP-enabled MoE block:

```text
          rank 0              rank 1              rank 2              rank 3
        ┌──────────┐        ┌──────────┐        ┌──────────┐        ┌──────────┐
        │ hidden_0 │        │ hidden_1 │        │ hidden_2 │        │ hidden_3 │
        │ topk_0   │        │ topk_1   │        │ topk_2   │        │ topk_3   │
        └────┬─────┘        └────┬─────┘        └────┬─────┘        └────┬─────┘
             │                   │                   │                   │
             │  [all-to-all dispatch: tokens go to ranks owning their experts]
             │                   │                   │                   │
        ┌────▼─────┐        ┌────▼─────┐        ┌────▼─────┐        ┌────▼─────┐
        │ experts  │        │ experts  │        │ experts  │        │ experts  │
        │ 0..63    │        │ 64..127  │        │ 128..191 │        │ 192..255 │
        │ grouped  │        │ grouped  │        │ grouped  │        │ grouped  │
        │ GEMM     │        │ GEMM     │        │ GEMM     │        │ GEMM     │
        └────┬─────┘        └────┬─────┘        └────┬─────┘        └────┬─────┘
             │                   │                   │                   │
             │  [all-to-all combine: expert outputs return to original ranks,
             │                              weighted by topk_probs]
             │                   │                   │                   │
        ┌────▼─────┐        ┌────▼─────┐        ┌────▼─────┐        ┌────▼─────┐
        │ out_0    │        │ out_1    │        │ out_2    │        │ out_3    │
        └──────────┘        └──────────┘        └──────────┘        └──────────┘
```

### 8.9 Does Qwen3-30B-A3B-Instruct-2507 benefit from EP?
{: #par-qwen3-ep }

Running the numbers from §8.3:

| Config                             | MoE params per rank per layer (bf16) | MoE total per rank (48 layers) |
|------------------------------------|--------------------------------------|--------------------------------|
| Qwen3 --tp 4 (no EP)               | ~288 MB                              | ~13.8 GB                       |
| Qwen3 --tp 4 --ep-size 4 (full EP) | ~72 MB                               | ~3.5 GB                        |

EP saves ~10.3 GB per rank — noticeable, but on H200's 141 GB that's a 7% improvement on the total memory budget. In exchange you add two all-to-all collectives per MoE block (96 per forward). At decode-time with a ~100-token batch and H200 NVLink, those two A2As add something on the order of ~300-500 μs **per layer** in low-latency mode — so 15-25 ms per forward pass at 48 layers, which is very substantial for decoding.

That's why for Qwen3-30B-A3B-Instruct-2507, **EP is usually a loss on a single node**. You'd enable it only in two scenarios:

1. **Multi-node deployment.** If you're running across 2+ nodes (rare for a 30B model), EP lets each rank hold a smaller slice, reducing the cross-node KV cache contention.
2. **Extreme long-context serving.** If you need every byte of HBM for KV cache (say, 1M-context batched inference), saving ~10 GB per rank for KV may be worth the 20ms/step overhead.

For DeepSeek-V3 and Kimi K2 it's a clear win: EP is essentially required on a single 8×H200 node since the MoE weights don't fit otherwise. That's why every reference deployment for those models uses `--moe-a2a-backend deepep --ep-size 8 --enable-ep-moe`.

<div class="callout motiv" markdown="1">

#### When in doubt, benchmark

SGLang's `benchmark/kernels/moe_dispatch/` directory has standalone DeepEP benchmarks you can run to measure the A2A latency for your exact GPU topology, hidden size, and num_experts. If the A2A per-layer latency times the number of layers exceeds the MoE FLOP latency savings from smaller weights, EP is a loss. For the `--tp 4` Qwen3 case it's a loss; for `--tp 8 --ep-size 8` DeepSeek it's a big win.

</div>

#### Further reading on SGLang's EP story

- <a href="https://lmsys.org/blog/2025-05-05-large-scale-ep/" target="_blank" rel="noopener noreferrer">SGLang large-scale expert-parallel blog (May 2025)</a> — origin of the two-batch overlap design, EPLB details, DeepGEMM integration.
- <a href="https://github.com/deepseek-ai/DeepEP" target="_blank" rel="noopener noreferrer">DeepEP GitHub repo</a> — the library SGLang wraps. README has benchmark numbers and kernel-level detail.
- <a href="https://arxiv.org/abs/2412.19437" target="_blank" rel="noopener noreferrer">DeepSeek-V3 tech report</a> — the original EP-based MoE design that motivated DeepEP.
- <a href="https://arxiv.org/abs/1909.08053" target="_blank" rel="noopener noreferrer">Megatron-LM paper (Shoeybi et al., 2019)</a> — the original column/row-parallel TP design SGLang inherits.

---

<p class="bridge" markdown="span">*TP and EP handle within-a-replica parallelism. The remaining three dimensions — pipeline, context, and data parallelism — plus the two request routers on top address scale-out across replicas and very-long-context workloads.*</p>

## 9 · PP, CP, DP & the routers
{: #parallelism2 }

Part 10 covered the two parallelism axes you need for a single-node Qwen3 run. This part fills in the rest: pipeline parallelism (layer-dim sharding across nodes), context parallelism (sequence-dim sharding for long-context prefill), and data parallelism — which in SGLang actually means **two different things** that both get called DP, plus the two different **routers** that coordinate requests across DP groups.

### 9.1 Pipeline Parallel — `event_loop_pp` & <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/forward_batch_info.py#L1080" class="sym-link" target="_blank" rel="noopener noreferrer"><code>PPProxyTensors</code></a>
{: #par-pp }

PP splits the model along the **layer dimension**. For `--pp 4`, each rank owns 12 of the 48 decoder layers, and hidden states flow through the ranks sequentially: stage 0 runs layers 0–11, sends its hidden states to stage 1 (which runs 12–23), and so on. The last stage applies `lm_head` and samples.

Unlike TP (same work, split tensors, collective per layer) and EP (different experts per rank, A2A per MoE block), PP has **no collective** between ranks — just point-to-point sends of hidden states. The bottleneck is straggler / bubble time, not bandwidth.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> managers/scheduler_pp_mixin.py:47-66 — event_loop_pp (excerpt + docstring) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler_pp_mixin.py#L47" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
class SchedulerPPMixin:
    @DynamicGradMode()
    def event_loop_pp(self: Scheduler):
        """
        A scheduler loop for pipeline parallelism.
        Notes:
        1. Each stage runs in the same order and is notified by the previous stage.
        2. We use async send but sync recv to avoid desynchronization
           while minimizing the communication overhead.
        3. We can use async batch depth to buffer the outputs in the last stage
           to allow overlapping the GPU computation and CPU processing
           and avoid last PP rank straggler.

        Unified Schedule:
        ====================================================================
        Stage P
        recv ith req from previous stage
        recv ith proxy from previous stage
        run ith batch
        recv prev (i+1)% mb_size th outputs
        process batch result of prev (i+1)% mb_size th batch
            (can be run in parallel with the curr batch GPU computation)
        send ith req to next stage
        send ith proxy to next stage
        send current stage's outputs to next stage
            (can be stashed and delayed to send later)
        ====================================================================
        """
        self.init_pp_loop_state()
        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                ...
```

Three mechanisms make PP work in SGLang:

1. **<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/forward_batch_info.py#L1080" class="sym-link" target="_blank" rel="noopener noreferrer"><code>PPProxyTensors</code></a>.** A thin wrapper around a `Dict[str, torch.Tensor]`. The hidden-state tensors a stage produces ("residual", "hidden_states", etc.) get packed into one of these and shipped to the next stage. Adapted from vLLM — see the comment at the top of the class.
2. **Async send + sync recv.** Each stage posts a non-blocking P2P send of its output, then blocks on the recv from the previous stage. This ordering prevents livelock (ranks waiting for each other's sends) while still allowing the send to overlap with the next compute.
3. **`pp_async_batch_depth` look-ahead.** On the last rank, sampling output is normally the critical-path tail (tokenization, detokenization IPC). With async depth \> 0, the last rank buffers N+1 batches' outputs so its send of batch *i* can overlap with GPU compute of batch *i+1*. Tunable via `--pp-async-batch-depth`.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> model_executor/forward_batch_info.py:1080 — PPProxyTensors <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/forward_batch_info.py#L1080" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
class PPProxyTensors:
    # adapted from https://github.com/vllm-project/vllm/blob/d14e98d924724b284dc5eaf8070d935e214e50c0/vllm/sequence.py#L1103
    tensors: Dict[str, torch.Tensor]

    def __init__(self, tensors):
        # manually define this function, so that
        # Dynamo knows `IntermediateTensors()` comes from this file.
        self.tensors = tensors

    def __getitem__(self, key: Union[str, slice]):
        if isinstance(key, str):
            return self.tensors[key]
        elif isinstance(key, slice):
            return self.__class__({k: v[key] for k, v in self.tensors.items()})

    def __setitem__(self, key: str, value: torch.Tensor):
        self.tensors[key] = value
```

The dispatch table from §4.4 already shows PP's three event loops: `event_loop_pp` (normal), `event_loop_pp_disagg_prefill`, and `event_loop_pp_disagg_decode` (the latter two combine PP with prefill-decode disaggregation). Each microbatch slot (`mb_id`) maintains its own <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py#L1321" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ScheduleBatch</code></a> and `last_batch` state, rotating through `pp_loop_size` slots per iteration.

<div class="callout warn" markdown="1">

#### PP incompatibilities

From `server_args.check_server_args()` at `server_args.py:6467`: "Pipeline parallelism is not compatible with overlap schedule, speculative decoding." Also, context parallelism: `_handle_context_parallelism` asserts `pp_size == 1`. In practice you enable PP only for models so large that TP on one node won't fit — typically 400 B+ dense models or very-large-expert MoE — and you lose the overlap and spec-decode optimizations.

</div>

<div class="callout motiv" markdown="1">

#### Why PP exists alongside TP

TP's all-reduce cost scales with `tp_size` (latency-bound on small tensors, bandwidth-bound on large ones). Past 8 ranks on typical NVLink topologies, TP starts hurting. PP's per-layer P2P cost is **constant regardless of `pp_size`** (each stage just sends one message per forward), so PP scales better across nodes. Typical deployment for \>100B models: TP within a node (8 H200s), PP across nodes. The trade-off is PP's bubble: rank P has nothing to do until rank P-1 sends it something, so the first `pp_size-1` microbatches are idle time.

</div>

### 9.2 Context Parallel — prefill attention over long sequences
{: #par-cp }

CP splits the **sequence dimension** across ranks. For a 256K-token prompt on 4 CP ranks, each rank holds 64K tokens' worth of Q/K/V activations, runs attention on its shard, and then (in current SGLang) all-gathers K/V so each rank can compute its Q against the full K/V context.

CP is narrower in scope than TP/EP/PP. In SGLang it is:

- **Prefill-only.** Decode doesn't use CP — at decode time each request processes 1 token, so there's nothing to shard along the sequence axis.
- **Attention-only.** The feedforward / MoE portion runs unchanged on whichever TP or EP topology you chose.
- **Used alongside DP-attention.** The arithmetic in `_handle_context_parallelism` enforces `tp_size % (dp_size × attn_cp_size) == 0`, meaning CP eats into the TP group after DP-attention has claimed its portion.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> server_args.py:2750-2760 — _handle_context_parallelism (CP constraints) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L2750" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def _handle_context_parallelism(self):
    if self.attn_cp_size > 1:
        # The tp_size is the world size, not the real tensor parallel size
        assert (
            self.tp_size % self.attn_cp_size == 0
        ), "tp_size must be divisible by attn_cp_size"
        assert (
            self.tp_size % (self.dp_size * self.attn_cp_size) == 0
        ), "tp_size must be divisible by dp_size * attn_cp_size"

        assert (
            not self.enable_aiter_allreduce_fusion
        ), "Aiter allreduce fusion is not supported with context parallelism"
    ...
    if self.moe_dp_size > 1:
        ...
        assert self.pp_size == 1, "PP is not supported with context parallelism"
```

#### Two sharding strategies

SGLang's CP implementation (`layers/utils/cp_utils.py`) supports two ways to slice the sequence:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> layers/utils/cp_utils.py:213-260 — cp_all_gather_rerange_output (docstring) <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/utils/cp_utils.py#L213" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def cp_all_gather_rerange_output(input_tensor, cp_size, forward_batch, stream):
    """
    # for in-seq-split
    |   +-----------before allgather------------+|
    |   | dp_atten_tp0: block0, block7 |
    |   | dp_atten_tp1: block1, block6 |
    |   | dp_atten_tp2: block2, block5 |
    |   | dp_atten_tp3: block3, block4 |
    |
    |   +----------before rerange---------------+|
    | block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4 |
    |
    |   +--------------result-------------------+
    | block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7 |

    # for round-robin-split
    |   +-----------before allgather------------+|
    | dp_atten_tp0: token0, token4, token8, token12, token16, ... |
    | dp_atten_tp1: token1, token5, token9, token13, token17, ... |
    | dp_atten_tp2: token2, token6, token10, token14, token18, ... |
    | dp_atten_tp3: token3, token7, token11, token15, token19, ... |
    |
    |   +--------------result-------------------+
    | token0, token1, token2, token3, token4, token5, token6, token7, ...
    """
```

- **In-seq-split (zigzag).** Each rank holds *two* non-contiguous blocks, arranged so that rank *i* and rank `cp_size-1-i` together cover the full sequence symmetrically. This keeps per-rank compute balanced even when later tokens attend to more context (causal attention).
- **Round-robin-split.** Tokens are strided — rank *i* gets tokens *i, i+cp_size, i+2·cp_size, …*. Used by NSA (Native Sparse Attention) prefill, where the sparse pattern distributes work evenly without needing zigzag balancing.

The core dataclass tying it all together:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> layers/utils/cp_utils.py:20-40 — ContextParallelMetadata <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/utils/cp_utils.py#L20" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
@dataclass
class ContextParallelMetadata:
    split_list: List[int] = None
    max_rank_len: List[int] = None
    zigzag_index: List[int] = None
    per_rank_actual_token: List[int] = None
    reverse_split_len: List[int] = None
    cp_reverse_index: List[int] = None

    # metadata for attention
    kv_len_prev: int = -1
    kv_len_next: int = -1
    actual_seq_q_prev: int = -1
    actual_seq_q_next: int = -1
    kv_len_prev_tensor: torch.Tensor = None
    kv_len_next_tensor: torch.Tensor = None
    actual_seq_q_prev_tensor: torch.Tensor = None
    actual_seq_q_next_tensor: torch.Tensor = None

    total_seq_lens: torch.Tensor = None
```

The `_prev`/`_next` fields come from the zigzag pattern: each rank's Q is split into "prev half" (attending to tokens before this rank's first block) and "next half" (attending to tokens that include this rank's second block). Two separate attention kernel calls, one per half, with different `cu_seqlens_q` and `cache_seqlens`. The backend-specific `cp_attn_forward_extend` helper dispatches both halves and concatenates the result.

#### The K/V all-gather — where the bandwidth goes

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> layers/utils/cp_utils.py:323-352 — cp_allgather_and_save_kv_cache <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/utils/cp_utils.py#L323" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def cp_allgather_and_save_kv_cache(forward_batch, layer, k, v, cp_size):
    """
    Allgather KV cache from all CP ranks and write the full result
    into each rank's local memory pool.
    """
    cache_loc = (
        forward_batch.out_cache_loc
        if not layer.is_cross_attention
        else forward_batch.encoder_out_cache_loc
    )

    k = k.contiguous()
    v = v.contiguous()

    key_cache_full = cp_all_gather_rerange_kv_cache(
        k, cp_size, forward_batch, torch.cuda.current_stream()
    )
    value_cache_full = cp_all_gather_rerange_kv_cache(
        v, cp_size, forward_batch, torch.cuda.current_stream()
    )

    forward_batch.token_to_kv_pool.set_kv_buffer(
        layer,
        cache_loc,
        key_cache_full,
        value_cache_full,
        layer.k_scale,
        layer.v_scale,
    )
```

So per CP-enabled attention block: **two all-gather collectives** (one for K, one for V), each moving roughly `T × num_kv_heads × head_dim × 2 bytes / cp_size` bytes off this rank (it sends its shard, receives everyone else's). For DeepSeek-V3 with 1 KV head (MLA) at 256 K context, that's ~32 MB per layer of K + another ~32 MB of V. Across 61 layers, that's ~4 GB of communication per prefill — substantial but paid once per request, amortized across all subsequent decode tokens.

<div class="callout motiv" markdown="1">

#### Why does CP exist alongside TP and DP-attention?

For very long prompts on DeepSeek-class MLA models, the *attention compute* itself becomes the bottleneck (it's O(T²)) — not the weight movement. TP doesn't help because MLA already has only 1 KV head; DP-attention doesn't help because it shards *requests*, not *tokens within a request*. CP is the only axis that parallelizes the T² compute. Enabled via `--enable-prefill-context-parallel --prefill-cp-mode in-seq-split`; typical config sets `attn_cp_size = tp_size / dp_size` so each DP-attention group internally CP-splits its long prompts.

</div>

### 9.3 Two things called "DP": attention vs replication
{: #par-dp-two }

SGLang uses the term "data parallel" for two completely different things. You need to know which one someone's talking about.

|                     | DP attention                                                   | DP replication                                                                                                                                                                                                                              |
|---------------------|----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CLI flag            | `--enable-dp-attention`                                        | `--dp-size N`                                                                                                                                                                                                                               |
| What it replicates  | Attention compute only (MoE is shared).                        | The entire model, N times.                                                                                                                                                                                                                  |
| What it partitions  | Tokens across attention groups.                                | Requests across groups.                                                                                                                                                                                                                     |
| Process count       | Same as `tp_size` (one process per GPU).                       | `dp_size × tp_size × pp_size` processes, plus a controller.                                                                                                                                                                                 |
| Router / dispatcher | Handled inside each scheduler.                                 | Dedicated <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/data_parallel_controller.py#L118" class="sym-link" target="_blank" rel="noopener noreferrer"><code>DataParallelController</code></a> process. |
| When it wins        | MLA-style models (1 KV head) where pure TP over-replicates KV. | Small-to-medium models where you want more concurrent requests per node than a single engine can handle.                                                                                                                                    |

The two can combine: `--tp 8 --dp-size 2 --enable-dp-attention` means two full model replicas (DP replication), each using DP attention internally across 4 of its 8 TP ranks.

### 9.4 DP attention — `1/dp_size` tokens per attention group
{: #par-dp-attn }

DP attention is the modern SGLang design used for DeepSeek-V3, Kimi K2, GLM-4.5, and anything with MLA or very few KV heads. It was introduced in <a href="https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/" target="_blank" rel="noopener noreferrer">SGLang v0.4</a> specifically for DeepSeek, where MLA's single KV head made straight TP waste almost all GPU memory on duplicated KV cache.

Core idea: inside one forward pass, the attention compute runs on **just this DP group's requests**, but the MoE compute is shared across the full TP group. Before attention, tokens get scattered to their home DP group; after attention, they get all-gathered back for MoE.

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> layers/dp_attention.py:237-252 — compute_dp_attention_world_info <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/dp_attention.py#L237" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def compute_dp_attention_world_info(
    enable_dp_attention, tp_rank, tp_size, dp_size, attn_cp_size: int = 1
):
    attn_dp_size = dp_size if enable_dp_attention else 1
    attn_tp_size = tp_size // attn_dp_size // attn_cp_size
    attn_tp_rank = tp_rank % attn_tp_size

    if not enable_dp_attention:
        attn_dp_rank = 0
    else:
        # Rank layout is (dp, cp, tp) where tp is the fastest-changing dim:
        # tp_rank = (attn_dp_rank * attn_cp_size + attn_cp_rank) * attn_tp_size + attn_tp_rank
        attn_dp_rank = tp_rank // (attn_tp_size * attn_cp_size)

    return attn_tp_rank, attn_tp_size, attn_dp_rank
```

For `--tp 8 --dp-size 2 --enable-dp-attention` (and no CP):

- `attn_tp_size = 8 / 2 / 1 = 4` — each DP-attention group has 4 TP ranks.
- `attn_dp_rank = tp_rank // 4` — ranks 0–3 are DP group 0; ranks 4–7 are DP group 1.
- `attn_tp_rank = tp_rank % 4` — within each DP group, ranks are numbered 0–3 for attention-TP collectives.

At runtime, DeepSeek-V3's model code calls `cp_all_gather_rerange_output` (or its DP-attention sibling) right before entering the MoE block to re-assemble the full token set. The TP group for the MoE is still size 8, so MoE sees the concatenated output of both DP groups.

One concrete result: a DeepSeek-V3 deployment on 8× H200 with `--dp-size 8 --enable-dp-attention` stores the MLA KV cache on just **1 rank** per request (instead of duplicated across 8 TP ranks), freeing ~8× more HBM for KV — which is exactly what long-context DeepSeek serving needs.

### 9.5 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/data_parallel_controller.py#L118" class="sym-link" target="_blank" rel="noopener noreferrer"><code>DataParallelController</code></a> — the in-process router
{: #par-dp-controller }

DP replication is the older, simpler DP design. It runs `dp_size` complete scheduler subgroups (each internally TP/PP-parallel), with a **controller process** sitting in front of them that receives tokenized requests from the TokenizerManager and decides which DP group to send each request to.

The controller is a full separate process spawned at startup, with its own event loop:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> managers/data_parallel_controller.py:118-178 — DataParallelController.__init__ <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/data_parallel_controller.py#L118" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
class DataParallelController:
    """A controller that dispatches requests to multiple data parallel workers."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        run_scheduler_process_func: Callable,
    ) -> None:
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )
        self.run_scheduler_process_func = run_scheduler_process_func

        # Init inter-process communication
        self.context = zmq.Context(1 + server_args.dp_size)
        if server_args.node_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                self.context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )

        # Dispatch method
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM: self.follow_bootstrap_room_scheduler,
            LoadBalanceMethod.TOTAL_REQUESTS: self.total_requests_scheduler,
            LoadBalanceMethod.TOTAL_TOKENS: self.total_tokens_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]

        # Load balance budget
        self.dp_budget = DPBudget(server_args.dp_size)

        self.scheduler_procs = []
        self.workers: List[zmq.Socket] = [None] * server_args.dp_size
        self.status: List[bool] = [True] * server_args.dp_size

        if server_args.enable_dp_attention:
            self.launch_dp_attention_schedulers(server_args, port_args)
            local_ctrl = server_args.enable_dp_attention_local_control_broadcast
            self.control_message_step = 1 if local_ctrl else server_args.tp_size
        else:
            self.launch_dp_schedulers(server_args, port_args)
            self.control_message_step = 1

        self.init_dispatcher()
```

Two key pieces: the **dispatch method** (one of four load-balance policies) and the **<a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/data_parallel_controller.py#L87" class="sym-link" target="_blank" rel="noopener noreferrer"><code>DPBudget</code></a>** (running counters of load per group).

#### The four load-balance methods

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> managers/data_parallel_controller.py:70-85 — LoadBalanceMethod <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/data_parallel_controller.py#L70" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
    FOLLOW_BOOTSTRAP_ROOM = auto()
    TOTAL_REQUESTS = auto()
    TOTAL_TOKENS = auto()
```

Each has its own dispatcher method:

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> managers/data_parallel_controller.py:541-590 — four dispatcher methods <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/data_parallel_controller.py#L541" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
def round_robin_scheduler(self, req: Req):
    if self.maybe_external_dp_rank_routing(req):
        return

    while True:
        if self.status[self.round_robin_counter]:
            logger.debug(f"Choose worker {self.round_robin_counter}")
            self.workers[self.round_robin_counter].send_pyobj(req)
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                self.workers
            )
            break
        self.round_robin_counter = (self.round_robin_counter + 1) % len(self.workers)

def follow_bootstrap_room_scheduler(self, req: Req):
    if self.maybe_external_dp_rank_routing(req):
        return

    if (
        req.bootstrap_room is None
        and self.server_args.disaggregation_transfer_backend == "fake"
    ):
        req.bootstrap_room = self.round_robin_counter
        self.round_robin_counter = (self.round_robin_counter + 1) % len(self.workers)

    assert req.bootstrap_room is not None, (
        "req.bootstrap_room should not be None. Do not send requests directly to "
        "prefill or decode instances; send to the router instead."
    )
    target_rank = req.bootstrap_room % len(self.workers)
    self.workers[target_rank].send_pyobj(req)

def total_requests_scheduler(self, req: Req):
    if self.maybe_external_dp_rank_routing(req):
        return
    target_worker = self.dp_budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
    self.workers[target_worker].send_pyobj(req)

def total_tokens_scheduler(self, req: Req):
    if self.maybe_external_dp_rank_routing(req):
        return
    target_worker = self.dp_budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS)
    self.workers[target_worker].send_pyobj(req)
```

| Method                  | When to use                                                                  | How it picks a worker                                                                                                                                                                                                                                          |
|-------------------------|------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ROUND_ROBIN`           | Default. Uniform workload, no cache affinity matters.                        | Monotonic counter mod dp_size, skipping dead workers.                                                                                                                                                                                                          |
| `FOLLOW_BOOTSTRAP_ROOM` | PD-disaggregation. Request must go to the same DP rank as its prefill phase. | `req.bootstrap_room % dp_size`. The bootstrap_room is assigned by the prefill instance and carried through the request.                                                                                                                                        |
| `TOTAL_REQUESTS`        | Variable request sizes. Want to balance queue depth.                         | Pick the worker with fewest in-flight requests from <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/data_parallel_controller.py#L87" class="sym-link" target="_blank" rel="noopener noreferrer"><code>DPBudget</code></a>. |
| `TOTAL_TOKENS`          | Very variable context lengths. Want to balance KV pressure.                  | Pick the worker with fewest total tokens (running + waiting). Tie-break on request count.                                                                                                                                                                      |

#### DPBudget — the running load counters

<div class="code-head" markdown="span">
<span class="badge badge-sg">SG</span> managers/data_parallel_controller.py:87-116 — DPBudget <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/data_parallel_controller.py#L87" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>

```python
class DPBudget:
    def __init__(self, dp_size: int):
        self.dp_size = dp_size
        self.total_requests = [0] * dp_size
        self.total_tokens = [0] * dp_size

    def update_budget(self, load_update: WatchLoadUpdateReq):
        """Update the budget."""
        for load in load_update.loads:
            self.total_requests[load.dp_rank] = (
                load.num_running_reqs + load.num_waiting_reqs
            )
            self.total_tokens[load.dp_rank] = load.num_total_tokens

    def dispatch(self, method: LoadBalanceMethod):
        if method == LoadBalanceMethod.TOTAL_REQUESTS:
            target_rank = self.total_requests.index(min(self.total_requests))
        elif method == LoadBalanceMethod.TOTAL_TOKENS:
            # Use total_requests as a tie-breaker when total_tokens are equal
            target_rank = min(
                range(self.dp_size),
                key=lambda i: (self.total_tokens[i], self.total_requests[i]),
            )
        else:
            return None

        # Increment the load of that worker by one as a heuristic
        self.total_requests[target_rank] += 1
        return target_rank
```

Each scheduler subprocess periodically publishes a <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/io_struct.py#L2012" class="sym-link" target="_blank" rel="noopener noreferrer"><code>WatchLoadUpdateReq</code></a> containing its current running/waiting/token counts; the controller's `handle_load_update_req` updates <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/data_parallel_controller.py#L87" class="sym-link" target="_blank" rel="noopener noreferrer"><code>DPBudget</code></a>. The heuristic increment (`self.total_requests[target_rank] += 1` right after picking) prevents the obvious thundering-herd: if 100 requests arrive in the same tick before any load update returns, they won't all go to the same worker — the counter advances fake-optimistically.

<div class="callout info" markdown="1">

#### External DP routing override

Every dispatcher checks `maybe_external_dp_rank_routing(req)` first: if the request has `routed_dp_rank` set (by an upstream router), the controller just honors that and skips its own scheduling. This is the bridge to the external `sgl-router` — when the Rust router has already decided which DP group a request should go to (usually for cache-aware reasons), the in-process controller must respect that choice to avoid undoing the cache-affinity routing.

</div>

<div class="callout motiv" markdown="1">

#### Why these four policies and not, say, cache-aware?

The in-process controller runs entirely CPU-side and has no visibility into individual radix-tree state on its workers — that would require streaming the per-worker tree back to the controller every update, which is prohibitively expensive. So the in-process controller stays at the level of coarse load metrics. Cache-aware routing is the *external* router's job (§9.7) because the external router can afford to maintain its own approximate radix trees. FOLLOW_BOOTSTRAP_ROOM is the exception: it's not cache-aware, just a hashed affinity for PD-disagg correctness.

</div>

### 9.6 `sgl-router` — the external Rust HTTP gateway
{: #par-sgl-router }

**This is a separate project from the main SGLang repo.** It lives at `sgl-project/sglang` under the `sgl-model-gateway/` subdirectory, ships on PyPI as `sglang-router`, and is implemented in Rust. Originally called "SGLang Router," it was rebranded to **"SGLang Model Gateway"** as its scope expanded.

Unlike the in-process <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/data_parallel_controller.py#L118" class="sym-link" target="_blank" rel="noopener noreferrer"><code>DataParallelController</code></a> (which lives inside one SGLang server and dispatches across that server's DP groups), the model gateway sits **in front of many independent SGLang server instances**. You launch, say, 4 separate `python -m sglang.launch_server` processes (each with their own `--tp`), then point one `python -m sglang_router.launch_router` at all four:

```bash
# Start 4 workers (each is a full SGLang server with its own scheduler/model)
python -m sglang.launch_server --model Qwen/Qwen3-30B-A3B-Instruct-2507 --tp 4 --port 30001 &
python -m sglang.launch_server --model Qwen/Qwen3-30B-A3B-Instruct-2507 --tp 4 --port 30002 &
python -m sglang.launch_server --model Qwen/Qwen3-30B-A3B-Instruct-2507 --tp 4 --port 30003 &
python -m sglang.launch_server --model Qwen/Qwen3-30B-A3B-Instruct-2507 --tp 4 --port 30004 &

# Start the router in front
python -m sglang_router.launch_router \
  --worker-urls http://localhost:30001 http://localhost:30002 http://localhost:30003 http://localhost:30004 \
  --policy cache_aware \
  --host 0.0.0.0 --port 8000
```

Clients send OpenAI-compatible requests to `localhost:8000`; the router picks a worker and proxies the request.

#### When to use the external router vs the in-process controller

| Scenario                                                                             | Use                                                                                                   |
|--------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| One node, fits the model at `--tp N`, want more concurrent requests                  | In-process `--dp-size M`.                                                                             |
| Multiple independent servers (possibly on different nodes), want unified entry point | External `sgl-router`.                                                                                |
| Prefill-decode disaggregation (separate prefill and decode pools)                    | External router with `--pd-disaggregation` (the in-process controller only supports monolithic mode). |
| Kubernetes deployment with service discovery                                         | External router with `--service-discovery --selector app=sglang-worker`.                              |
| Need cache-aware routing across servers                                              | External router (in-process controller doesn't do cache-aware).                                       |
| Mixing models / multi-model gateway                                                  | External router.                                                                                      |
| Production observability, circuit breakers, retries                                  | External router.                                                                                      |

#### The nine load-balance policies

As of `sglang-router 0.3.x`, the policies exposed are (from the PyPI description and the Python binding):

| Policy               | How it picks                                                                                                                   |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `random`             | Uniform random across healthy workers. Baseline.                                                                               |
| `round_robin`        | Monotonic counter mod num_workers.                                                                                             |
| `cache_aware`        | Approximate-radix-tree longest-prefix match. The key feature — see §9.7.                                                       |
| `power_of_two` (P2C) | Sample two workers at random, pick the less-loaded one. Proven to reduce tail latency vs pure random at minimal tracking cost. |
| `bucket`             | Hash-bucket routing: partition workers into buckets and pick by request attribute.                                             |
| `manual`             | Client specifies the target worker explicitly.                                                                                 |
| `consistent_hashing` | Hash request (usually by session ID) to a stable worker — sticky sessions for conversation affinity.                           |
| `prefix_hash`        | Hash the **first N tokens** of the prompt to pick a worker. Cheap approximation of cache-awareness without a tree.             |

PD-disaggregation mode takes **two** policies (one for prefill pool, one for decode pool). The canonical recipe from the docs is `--prefill-policy cache_aware --decode-policy power_of_two`: cache-awareness matters most for prefill (that's where KV gets built), while decode is mostly uniform work and benefits more from load-balancing.

#### Reliability layer

The router adds production-grade primitives that the in-process controller doesn't have:

- **Per-worker circuit breakers.** If a worker fails K times in a row, it's marked unhealthy and requests route around it; periodic "half-open" probes test recovery.
- **Retries with exponential backoff + full jitter.** Transient failures (timeouts, 5xx) automatically retry on another worker with capped exponential backoff.
- **Token-bucket rate limiting.** Global, per-tenant, or per-engine buckets with dynamic scaling based on worker load.
- **Request queuing.** When the system is saturated, requests queue up (with a max size) rather than being rejected immediately.

### 9.7 Cache-aware routing — the approximate radix tree
{: #par-sgl-router-cache }

This is the router's signature feature, introduced in <a href="https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/" target="_blank" rel="noopener noreferrer">SGLang v0.4 (Dec 2024)</a>. The claim: 1.9× throughput and 3.8× hit-rate vs round-robin, scaling better as worker count grows.

#### The idea

Each SGLang worker maintains its own radix tree of cached KV prefixes (§4.3). That tree lives on the worker's GPU — the router can't see it. What the router *can* do is **maintain its own approximate copy** of each worker's tree, lazily updated whenever it routes a new request there.

On each request:

1. Run longest-prefix-match of the new prompt's tokens against every worker's approximate tree.
2. Pick the worker with the longest match (i.e., the one most likely to have the biggest KV cache hit).
3. Insert the new prompt's tokens into that worker's approximate tree.
4. Forward the request.

Since the router's tree is just an approximation — it doesn't see GPU-side evictions in real time — it can get out of sync with the actual cache state. The system tolerates this because: (a) cache-aware routing makes it more likely the worker *does* still have the prefix; (b) the worker's KV cache is its own ground truth and will just do a regular match anyway; (c) the router periodically prunes stale entries via LRU.

#### The dynamic switch between cache-aware and pure load-balancing

Pure cache-aware routing has a failure mode: if worker A has a popular system prompt cached, **every** request with that system prompt goes to A, overloading it while B/C/D idle. The router defends against this with two thresholds:

- `balance_abs_threshold` (default 32): if the busiest worker has more than *balance_abs* requests more than the least-busy, switch to pure load-balancing for this decision.
- `balance_rel_threshold` (default 1.0001): if `max_load / min_load >` this ratio, switch to load-balancing.

In load-balancing fallback mode the router picks the smallest worker (lowest tree size, lowest load). Once balance returns, cache-aware kicks back in. So the routing is formally:

```python
# Simplified logic of the PD router (from PR #7031 diagram):
def route(req):
    if not load_balanced(balance_abs_threshold, balance_rel_threshold):
        return select_least_loaded_worker(req)

    target, match_rate = find_longest_prefix_match(req.tokens)
    if match_rate > cache_threshold:
        worker = target
    else:
        worker = select_smallest_tree_worker()

    worker.radix_tree.insert(req.tokens)
    worker.load_counter += 1
    return worker
```

For PD-disagg mode the decision is split into two stages: the prefill-pool decision uses cache-aware (because prefill builds the KV), and the decode-pool decision uses power-of-two (because decode is mostly uniform).

#### Memory bounds

The approximate radix tree on the router isn't free — every routed prompt adds nodes. Bounded by:

- `--max-tree-size N`: hard cap on total nodes per worker's tree.
- `--eviction-interval-secs S`: periodic background LRU purge.
- `--cache-threshold T`: minimum match rate to actually use the cache-aware suggestion (0.5 = only route by prefix if at least half the prompt matches).

#### Known limitations (Router Roadmap, Issue #10341)

- **Multi-replica routers don't sync.** If you run three router pods behind a Kubernetes service, each has its own independent approximate radix tree. Cache hit rate drops — requests for the same prefix may hit different routers each time. The <a href="https://github.com/sgl-project/sglang/issues/10341" target="_blank" rel="noopener noreferrer">Router Roadmap (Issue #10341)</a> plans a gRPC-mesh sync layer to fix this.
- **Router tree can disagree with worker cache.** Worker evicts a prefix under memory pressure → router still thinks it's there → routes the request, which now has to recompute. Addressed by "precise prefix-cache aware routing" (a proposed future mode where workers publish KV-cache events to the router).
- **Session affinity workaround.** If cache efficiency matters and you need multi-replica routers, the docs recommend configuring your L4 load balancer for session-affinity-by-user-hash so requests from the same user consistently hit the same router replica.

#### Further reading

- <a href="https://docs.sglang.ai/advanced_features/router.html" target="_blank" rel="noopener noreferrer">SGLang Model Gateway docs</a> — authoritative reference for CLI flags and deployment patterns.
- <a href="https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/" target="_blank" rel="noopener noreferrer">SGLang v0.4 blog</a> — original cache-aware router announcement with benchmark numbers.
- <a href="https://github.com/sgl-project/sglang/issues/10341" target="_blank" rel="noopener noreferrer">Router Roadmap Issue #10341</a> — multi-model, HA, gRPC-mesh sync plans.
- <a href="https://github.com/sgl-project/sglang/issues/7031" target="_blank" rel="noopener noreferrer">PR #7031</a> — PD router merge design with the routing-decision flow diagram.

---

---

<p class="bridge" markdown="span">*All of the deep-dive material is behind us. The remaining two Parts are practical references for when you sit down to actually modify SGLang.*</p>

## 10 · Where to change things
{: #dev }

A cheat sheet, mapped by "what you want to do" → "where the change actually goes". All paths relative to `python/sglang/srt/`.

### 8.1 Add a new model architecture

1. Create `models/my_new_model.py`. Implement `MyNewForCausalLM(nn.Module)` and define its `load_weights(self, weights)` method (see §5.5 for the template).
2. Declare `packed_modules_mapping` if your weights need fusing (e.g. q/k/v → qkv).
3. At bottom of file: `EntryClass = MyNewForCausalLM` (§5.3).
4. That's it — `pkgutil.iter_modules` on `sglang.srt.models` picks it up next server start. No manual registration needed.
5. Optional: add `models/my_new_model_test.py` with weight-loading round-trip, a forward-equivalence check against HF, and a small generation test.

### 8.2 Add a new attention backend

1. Create `layers/attention/my_backend.py`. Subclass <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/base_attn_backend.py#L18" class="sym-link" target="_blank" rel="noopener noreferrer"><code>AttentionBackend</code></a> from `layers/attention/base_attn_backend.py` and implement `forward_extend`, `forward_decode`, `init_forward_metadata`, `init_cuda_graph_state`.
2. Register it in the `ATTENTION_BACKENDS` dict (in `model_executor/model_runner.py` near `_get_attention_backend_from_str`).
3. Add a `--attention-backend my_backend` CLI option by extending the choices list in `server_args.py`.
4. Optional: update `_handle_attention_backend_compatibility` to auto-select your backend for specific architectures or SMs.

### 8.3 Add a new LoRA kernel backend

1. Create `lora/backend/my_backend.py`. Subclass <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/backend/base_backend.py#L9" class="sym-link" target="_blank" rel="noopener noreferrer"><code>BaseLoRABackend</code></a> and implement `run_lora_a_sgemm`, `run_lora_b_sgemm`, `run_qkv_lora`, `run_gate_up_lora`, <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L300" class="sym-link" target="_blank" rel="noopener noreferrer"><code>prepare_lora_batch</code></a>, <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L110" class="sym-link" target="_blank" rel="noopener noreferrer"><code>init_cuda_graph_batch_info</code></a>, `init_cuda_graph_moe_buffers`.
2. Register in `lora/backend/lora_registry.py`'s name→class map.
3. Use `--lora-backend my_backend`.

### 8.4 Change what LoRA shapes are allocated

Edit `lora/mem_pool.py`. `get_lora_A_shape` / `get_lora_B_shape` are the authoritative shape functions (§6.3). Both branch on `self.is_moe_module(module_name)`. If your new target module has a different fusion multiplier, update `lora/utils.py`'s `get_stacked_multiply`.

### 8.5 Change how many GPU bytes LoRA costs for a given config

The three knobs:

- `--max-lora-rank` (scales everything linearly)
- `--max-loras-per-batch` (scales everything linearly)
- `--lora-target-modules` (changes which buffers exist — dropping `gate_up_proj`/`down_proj` eliminates the 4D MoE buffers which are by far the biggest).

For Qwen3-30B-A3B-Instruct-2507 TP=4 at rank=64, max_loras=4: dropping MoE targets takes per-rank LoRA memory from ~20.4 GB to ~240 MB.

### 8.6 Debug a LoRA correctness issue

1. Run with `SGLANG_LOG_LEVEL=debug`. The <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L53" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAManager</code></a> logs every adapter load/unload and detected shape.
2. Set a breakpoint in <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L300" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAManager.prepare_lora_batch</code></a>. Print `forward_batch.lora_ids` and `weight_indices` — verify each request's expected adapter is present.
3. Set a breakpoint in the relevant `*WithLoRA.apply_lora` (e.g. <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L559" class="sym-link" target="_blank" rel="noopener noreferrer"><code>QKVParallelLinearWithLoRA.apply_lora</code></a>). Verify `self.A_buffer_qkv.shape`, `self.B_buffer_qkv.shape` match what <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/mem_pool.py#L49" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAMemoryPool</code></a> should have allocated.
4. Numerical check: run the same prompt with `--disable-cuda-graph`. If results differ, a CG-capture issue (probably a non-static tensor address).
5. Set `--lora-strict-loading`. This makes adapter-config validation errors hard failures instead of silent warnings, which catches a lot of "adapter targets module X but server wasn't started with X in target_modules" footguns.

### 8.7 Profile

- `--enable-profile-cuda-graph` captures per-batch-size PyTorch profiler traces at CUDA-graph capture time. Look for LoRA kernel hot spots there.
- For live serving, `POST /start_profile` / `POST /stop_profile` HTTP endpoints wrap `torch.profiler` around a window of live traffic.
- NSight Systems works too — launch with `nsys profile -t cuda,osrt,nvtx python -m sglang.launch_server ...` and open the `.nsys-rep` to see kernel timelines.

### 8.8 Key environment variables

| Env var                                      | Effect                                                                                                                   |
|----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `SGLANG_LOG_LEVEL=debug`                     | Verbose logs across all managers.                                                                                        |
| `SGLANG_DISABLED_MODEL_ARCHS`                | Comma list of model module names to skip during registry scan (lets you avoid a broken file during dev).                 |
| `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY` | After each forward, verifies KV pool invariants. Slower but catches leaks.                                               |
| `SGLANG_USE_BREAKABLE_CUDA_GRAPH`            | Uses a "breakable" CG capture so you can insert Python-side breakpoints inside what would normally be a captured region. |
| `SGLANG_PLUGIN_PACKAGES`                     | Comma list of packages to `importlib.import` before argparse — lets third-party code register models/backends.           |

---

<p class="bridge" markdown="span">*And finally, everything the doc cites — every symbol, every code block, every PR — collected in one place for quick lookup.*</p>

## 11 · Reference index
{: #refs }

Every code block this doc cites, grouped by area, with the function and file:line it came from. All at commit `1ebe1c57eddd0ea7915b408e35a1b9b33cd10c41` of SGLang.

### Launch / server args

| Symbol                                                                                                                                                                                                                          | File:Line                         |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| `sglang.launch_server` (CLI entry)                                                                                                                                                                                              | `python/sglang/launch_server.py`  |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L748" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ServerArgs.__post_init__</code></a>                            | `server_args.py:748`              |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L2406" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ServerArgs._handle_attention_backend_compatibility</code></a> | `server_args.py:2406`             |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L6467" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ServerArgs.check_server_args</code></a>                       | `server_args.py:6467`             |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L6659" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ServerArgs.check_lora_server_args</code></a>                  | `server_args.py:6659`             |
| `http_server.launch_server`                                                                                                                                                                                                     | `entrypoints/http_server.py:2313` |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/engine.py#L633" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Engine._launch_subprocesses</code></a>                  | `entrypoints/engine.py:633`       |

### Tokenizer / request path

| Symbol                                                                                                                                                                                                                             | File:Line                            |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py#L218" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TokenizerManager.__init__</code></a>               | `managers/tokenizer_manager.py:215`  |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py#L344" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TokenizerManager.init_ipc_channels</code></a>      | `managers/tokenizer_manager.py:344`  |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py#L420" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TokenizerManager.init_lora</code></a>              | `managers/tokenizer_manager.py:420`  |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py#L515" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TokenizerManager.generate_request</code></a>       | `managers/tokenizer_manager.py:515`  |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py#L700" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TokenizerManager._tokenize_one_request</code></a>  | `managers/tokenizer_manager.py:700`  |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py#L1361" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TokenizerManager._handle_batch_request</code></a> | `managers/tokenizer_manager.py:1361` |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_registry.py#L54" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRARegistry</code></a>                                     | `lora/lora_registry.py:54`           |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_registry.py#L27" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRARef</code></a>                                          | `lora/lora_registry.py:26`           |

### Scheduler

| Symbol                                                                                                                                                                                                                  | File:Line                    |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------|
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L3738" class="sym-link" target="_blank" rel="noopener noreferrer"><code>run_scheduler_process</code></a>               | `managers/scheduler.py:3738` |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L332" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Scheduler.__init__</code></a> (worker creation) | `managers/scheduler.py:~633` |
| Tree-cache selector                                                                                                                                                                                                     | `managers/scheduler.py:820`  |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L1373" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Scheduler.run_event_loop</code></a>            | `managers/scheduler.py:1373` |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L3652" class="sym-link" target="_blank" rel="noopener noreferrer"><code>dispatch_event_loop</code></a>                 | `managers/scheduler.py:3652` |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L1386" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Scheduler.event_loop_normal</code></a>         | `managers/scheduler.py:1386` |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L1506" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Scheduler.recv_requests</code></a>             | `managers/scheduler.py:1506` |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L1693" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Scheduler.process_input_requests</code></a>    | `managers/scheduler.py:1693` |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L2302" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Scheduler.get_next_batch_to_run</code></a>     | `managers/scheduler.py:2302` |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L2754" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Scheduler.run_batch</code></a>                 | `managers/scheduler.py:2754` |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L2937" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Scheduler.process_batch_result</code></a>      | `managers/scheduler.py:2937` |

### Model loading

| Symbol                                                                                                                                                                                                                       | File:Line                            |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/model_config.py#L97" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelConfig.__init__</code></a>                     | `configs/model_config.py:96`         |
| `get_config` (HF AutoConfig wrapper)                                                                                                                                                                                         | `utils/hf_transformers/config.py:52` |
| `ModelRegistry.resolve_model_cls`                                                                                                                                                                                            | `models/registry.py:78`              |
| `import_model_classes`                                                                                                                                                                                                       | `models/registry.py:92`              |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py#L675" class="sym-link" target="_blank" rel="noopener noreferrer"><code>DefaultModelLoader.load_model</code></a>            | `model_loader/loader.py:675`         |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py#L385" class="sym-link" target="_blank" rel="noopener noreferrer"><code>DefaultModelLoader._prepare_weights</code></a>      | `model_loader/loader.py:385`         |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py#L481" class="sym-link" target="_blank" rel="noopener noreferrer"><code>DefaultModelLoader._get_weights_iterator</code></a> | `model_loader/loader.py:480`         |
| `_initialize_model`                                                                                                                                                                                                          | `model_loader/loader.py:261`         |
| `safetensors_weights_iterator`                                                                                                                                                                                               | `model_loader/weight_utils.py:819`   |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/weight_utils.py#L1137" class="sym-link" target="_blank" rel="noopener noreferrer"><code>default_weight_loader</code></a>             | `model_loader/weight_utils.py:1137`  |

### Model (Qwen3MoE)

| Symbol                                                                                                                                                                                                             | File:Line                  |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------|
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_moe.py#L1099" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Qwen3MoeForCausalLM.load_weights</code></a> | `models/qwen3_moe.py:1099` |
| `Qwen3MoeForCausalLM.packed_modules_mapping`                                                                                                                                                                       | `models/qwen3_moe.py:940`  |
| `EntryClass = Qwen3MoeForCausalLM`                                                                                                                                                                                 | `models/qwen3_moe.py:1223` |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_moe.py#L233" class="sym-link" target="_blank" rel="noopener noreferrer"><code>Qwen3MoeSparseMoeBlock</code></a>            | `models/qwen3_moe.py:233`  |

### Parallel layers

| Symbol                                                                                                                                                                                                                                 | File:Line                                   |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------|
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L892" class="sym-link" target="_blank" rel="noopener noreferrer"><code>QKVParallelLinear.__init__</code></a>                               | `layers/linear.py:866`                      |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py#L1095" class="sym-link" target="_blank" rel="noopener noreferrer"><code>QKVParallelLinear.weight_loader</code></a>                         | `layers/linear.py:538`                      |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L159" class="sym-link" target="_blank" rel="noopener noreferrer"><code>FusedMoE.__init__</code></a>                    | `layers/moe/fused_moe_triton/layer.py:159`  |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L415" class="sym-link" target="_blank" rel="noopener noreferrer"><code>FusedMoE._load_w13</code></a>                   | `layers/moe/fused_moe_triton/layer.py:415`  |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L477" class="sym-link" target="_blank" rel="noopener noreferrer"><code>FusedMoE._load_w2</code></a>                    | `layers/moe/fused_moe_triton/layer.py:477`  |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L1050" class="sym-link" target="_blank" rel="noopener noreferrer"><code>FusedMoE.make_expert_params_mapping</code></a> | `layers/moe/fused_moe_triton/layer.py:1050` |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/unquant.py#L176" class="sym-link" target="_blank" rel="noopener noreferrer"><code>UnquantizedFusedMoEMethod.create_weights</code></a>   | `layers/quantization/unquant.py:163`        |

### ModelRunner / memory / graphs

| Symbol                                                                                                                                                                                                                                              | File:Line                                           |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L526" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelRunner.initialize</code></a>                                  | `model_executor/model_runner.py:526`                |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L1167" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelRunner.load_model</code></a>                                 | `model_executor/model_runner.py:1167`               |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L2026" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelRunner.configure_kv_cache_dtype</code></a>                   | `model_executor/model_runner.py:2026`               |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L2083" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelRunner.init_attention_backend</code></a>                     | `model_executor/model_runner.py:2083`               |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L2554" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelRunner.init_device_graphs</code></a>                         | `model_executor/model_runner.py` (init section)     |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L2882" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelRunner.forward</code></a>                                    | `model_executor/model_runner.py:2882`               |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L2941" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelRunner._forward_raw</code></a>                               | `model_executor/model_runner.py:2941`               |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py#L754" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ModelRunnerKVCacheMixin.init_memory_pool</code></a> | `model_executor/model_runner_kv_cache_mixin.py:754` |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool.py#L744" class="sym-link" target="_blank" rel="noopener noreferrer"><code>MHATokenToKVPool.__init__</code></a>                                     | `mem_cache/memory_pool.py:742`                      |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool.py#L850" class="sym-link" target="_blank" rel="noopener noreferrer"><code>MHATokenToKVPool._create_buffers</code></a>                              | `mem_cache/memory_pool.py:~849`                     |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/cuda_graph_runner.py#L515" class="sym-link" target="_blank" rel="noopener noreferrer"><code>CudaGraphRunner.__init__</code></a>                           | `model_executor/cuda_graph_runner.py:512`           |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/cuda_graph_runner.py#L761" class="sym-link" target="_blank" rel="noopener noreferrer"><code>CudaGraphRunner.capture</code></a>                            | `model_executor/cuda_graph_runner.py:761`           |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/cuda_graph_runner.py#L864" class="sym-link" target="_blank" rel="noopener noreferrer"><code>CudaGraphRunner.capture_one_batch_size</code></a>             | `model_executor/cuda_graph_runner.py:864`           |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/cuda_graph_runner.py#L666" class="sym-link" target="_blank" rel="noopener noreferrer"><code>CudaGraphRunner.can_run</code></a>                            | `model_executor/cuda_graph_runner.py:666`           |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/cuda_graph_runner.py#L1193" class="sym-link" target="_blank" rel="noopener noreferrer"><code>CudaGraphRunner.replay</code></a>                            | `model_executor/cuda_graph_runner.py:1193`          |

### LoRA

| Symbol                                                                                                                                                                                                                                   | File:Line                            |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L54" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAManager.__init__</code></a>                                    | `lora/lora_manager.py:52`            |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L413" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAManager.init_state</code></a>                                 | `lora/lora_manager.py:413`           |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L450" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAManager.init_lora_adapters</code></a>                         | `lora/lora_manager.py:450`           |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L471" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAManager._detect_shared_outer_loras</code></a>                 | `lora/lora_manager.py:471`           |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L508" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAManager.init_lora_shapes</code></a>                           | `lora/lora_manager.py:508`           |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L712" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAManager.init_lora_modules</code></a>                          | `lora/lora_manager.py:712`           |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L686" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAManager.init_memory_pool</code></a>                           | `lora/lora_manager.py:686`           |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L332" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAManager.update_lora_info</code></a>                           | `lora/lora_manager.py:332`           |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L300" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAManager.prepare_lora_batch</code></a>                         | `lora/lora_manager.py:300`           |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L124" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAManager.init_cuda_graph_moe_buffers</code></a>                | `lora/lora_manager.py:~123`          |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py#L110" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAManager.init_cuda_graph_batch_info</code></a>                 | `lora/lora_manager.py:~109`          |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/mem_pool.py#L52" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAMemoryPool.__init__</code></a>                                     | `lora/mem_pool.py:49`                |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/mem_pool.py#L176" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAMemoryPool.get_lora_A_shape</code></a>                            | `lora/mem_pool.py:175`               |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/mem_pool.py#L233" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAMemoryPool.get_lora_B_shape</code></a>                            | `lora/mem_pool.py:~232`              |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/mem_pool.py#L290" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAMemoryPool.init_buffers</code></a>                                | `lora/mem_pool.py:290`               |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/mem_pool.py#L421" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRAMemoryPool.prepare_lora_batch</code></a>                          | `lora/mem_pool.py:421`               |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L30" class="sym-link" target="_blank" rel="noopener noreferrer"><code>BaseLayerWithLoRA</code></a>                                             | `lora/layers.py:30`                  |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L58" class="sym-link" target="_blank" rel="noopener noreferrer"><code>VocabParallelEmbeddingWithLoRA</code></a>                                | `lora/layers.py:58`                  |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L224" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ParallelLMHeadWithLoRA</code></a>                                       | `lora/layers.py:224`                 |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L406" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ColumnParallelLinearWithLoRA</code></a>                                 | `lora/layers.py:406`                 |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L470" class="sym-link" target="_blank" rel="noopener noreferrer"><code>MergedColumnParallelLinearWithLoRA</code></a>                           | `lora/layers.py:470`                 |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L526" class="sym-link" target="_blank" rel="noopener noreferrer"><code>QKVParallelLinearWithLoRA</code></a>                                    | `lora/layers.py:526`                 |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L603" class="sym-link" target="_blank" rel="noopener noreferrer"><code>RowParallelLinearWithLoRA</code></a>                                    | `lora/layers.py:603`                 |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L694" class="sym-link" target="_blank" rel="noopener noreferrer"><code>ReplicatedLinearWithLoRA</code></a>                                     | `lora/layers.py:694`                 |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L782" class="sym-link" target="_blank" rel="noopener noreferrer"><code>FusedMoEWithLoRA</code></a>                                             | `lora/layers.py:782`                 |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L872" class="sym-link" target="_blank" rel="noopener noreferrer"><code>FusedMoEWithLoRA._get_lora_info</code></a>                              | `lora/layers.py:870`                 |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L913" class="sym-link" target="_blank" rel="noopener noreferrer"><code>FusedMoEWithLoRA.forward</code></a>                                     | `lora/layers.py:913`                 |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py#L1054" class="sym-link" target="_blank" rel="noopener noreferrer"><code>get_lora_layer</code></a>                                              | `lora/layers.py:1054`                |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/utils.py#L12" class="sym-link" target="_blank" rel="noopener noreferrer"><code>LoRABatchInfo</code></a>                                                  | `lora/utils.py:12`                   |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/backend/triton_backend.py#L22" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TritonLoRABackend</code></a>                             | `lora/backend/triton_backend.py:22`  |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/backend/triton_backend.py#L83" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TritonLoRABackend.run_qkv_lora</code></a>                | `lora/backend/triton_backend.py:83`  |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/backend/triton_backend.py#L140" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TritonLoRABackend.init_cuda_graph_batch_info</code></a> | `lora/backend/triton_backend.py:140` |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/backend/triton_backend.py#L184" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TritonLoRABackend.compute_sgemm_routing</code></a>      | `lora/backend/triton_backend.py:184` |
| <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/backend/triton_backend.py#L227" class="sym-link" target="_blank" rel="noopener noreferrer"><code>TritonLoRABackend.prepare_lora_batch</code></a>         | `lora/backend/triton_backend.py:227` |
| Triton kernels (sgemm_a, sgemm_b, qkv_b, gate_up_b, embedding_a, chunked_sgmv, fused_moe_lora)                                                                                                                                           | `lora/triton_ops/`                   |

### External references

- **SGLang repo:** <a href="https://github.com/sgl-project/sglang" target="_blank" rel="noopener noreferrer">github.com/sgl-project/sglang</a>
- **Qwen3-30B-A3B-Instruct-2507:** <a href="https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507" target="_blank" rel="noopener noreferrer">huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507</a>
- **HF transformers Qwen3MoE:** <a href="https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_moe" target="_blank" rel="noopener noreferrer">src/transformers/models/qwen3_moe/</a>
- **RadixAttention blog (Jan 2024):** <a href="https://www.lmsys.org/blog/2024-01-17-sglang/" target="_blank" rel="noopener noreferrer">lmsys.org/blog/2024-01-17-sglang/</a>
- **SGLang v0.4 zero-overhead batching:** <a href="https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/" target="_blank" rel="noopener noreferrer">lmsys.org/blog/2024-12-04-sglang-v0-4/</a>
- **S-LoRA paper:** <a href="https://arxiv.org/abs/2311.03285" target="_blank" rel="noopener noreferrer">arXiv:2311.03285</a>
- **Punica paper:** <a href="https://arxiv.org/abs/2310.18547" target="_blank" rel="noopener noreferrer">arXiv:2310.18547</a>
- **SwiGLU paper (w1/w2/w3 convention):** <a href="https://arxiv.org/abs/2002.05202" target="_blank" rel="noopener noreferrer">arXiv:2002.05202</a>
- **FlashAttention-3 paper (Hopper):** <a href="https://arxiv.org/abs/2407.08608" target="_blank" rel="noopener noreferrer">arXiv:2407.08608</a>
- **PR #7216 — LoRA × RadixCache compat:** <a href="https://github.com/sgl-project/sglang/pull/7216" target="_blank" rel="noopener noreferrer">sgl-project/sglang#7216</a>
- **Discussion #2141 — old LoRA limitations:** <a href="https://github.com/sgl-project/sglang/discussions/2141" target="_blank" rel="noopener noreferrer">sgl-project/sglang#2141</a>

---

— end of document —  
<span class="sub">Audited against commit `1ebe1c57eddd0ea7915b408e35a1b9b33cd10c41` on 2026-04-20</span>
