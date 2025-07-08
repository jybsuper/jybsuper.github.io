---
title: "Efficient Forward Pass for Agent RL: Solving Multi-Turn Context Consistency (Part 2)"
date: 2025-07-07 09:00:00 -0700
categories: [LLM]
tags: [flash-attention, flex-attention, sdpa-attention, kv-cache, verl, agent-rl, llm-infra, multi-turn]
author: yanbin_jiang
toc: true
pin: true
---

In [Part 1](https://jybsuper.github.io/posts/multiturn_forward_pass/), I explored the fundamental challenge of training-inference context mismatch in reasoning models and prototyped three solutions. While those initial experiments on a single conversation demonstrated promising accuracy improvements (300-700× better KL divergence), the real question remained: **how do these approaches perform at scale?** 

This post benchmarks the three approaches from Part 1 on a real multi-turn reasoning dataset. I'll share performance numbers, memory usage, and accuracy metrics to help us decide which approach to implement in VeRL.

## More Comprehensive Benchmarks

### Data

To properly evaluate these approaches, I needed a dataset that captures the complexity of real agent RL scenarios:
1. **Multi-turn conversations** - Essential for testing context handling across turns
2. **Reasoning traces** - Must include thinking/reasoning tokens to test removal logic  
3. **Tool calling** - Adds multi-step complexity typical in agent workflows

After searching HuggingFace's data hub, I found that [Jofthomas/hermes-function-calling-thinking-V1](https://huggingface.co/datasets/Jofthomas/hermes-function-calling-thinking-V1) meets all these requirements. I reformatted it to OpenAI message format and filtered out samples without reasoning traces, publishing the cleaned benchmark dataset[^1].

### Benchmark Code

The benchmark implementation builds on the prototypes from Part 1 with several key improvements. While this is a simplified setup for rapid testing (single GPU, batch size 1, no FSDP/DeepSpeed), it provides valuable insights into the relative performance characteristics of each approach.

The full benchmark code is available at [`/assets/codes/multiturn-forward-pass/benchmark.py`](https://github.com/jybsuper/jybsuper.github.io/blob/main/assets/codes/multiturn-forward-pass/benchmark.py).

#### Key Optimization: Message Batching

A critical optimization differentiates this benchmark from the Part 1 prototypes: **intelligent message grouping**. Instead of processing each assistant message independently, I group consecutive messages that maintain the same representation when used as context.

This batching strategy works because reasoning token removal follows specific patterns. In Qwen3, for example:
- Reasoning tokens are removed from assistant messages that appear **before** the last user query
- Assistant messages **after** a user query keep their reasoning tokens when becoming context

This allows us to:
1. **Batch consecutive messages** that share the same context representation
2. **Process them together** in a single forward pass
3. **Match naive single-pass performance** for models that don't modify messages when they become context

The implementation details for each method (KV cache batching, 2D custom mask construction) are extensively documented in the benchmark code.

### Benchmark Results

#### Performance Comparison

| Method | Avg Time (s) | P99 Time (s) | Speed vs Naive | Max Memory (GB) | Memory Savings |
|:------:|:-----------:|:------------:|:--------------:|:-------------:|:--------------:|
| **Naive** | 0.1623 | 0.2433 | 1.0× (baseline) | 41.64 | - |
| **KV Cache** | 0.1804 | 0.3463 | 0.9× | 24.15 | 42% |
| **SDPA** | 0.0509 | 0.0692 | **3.2×** | 23.55 | 43% |
| **Flex** | 0.0616 | 0.0672 | **2.6×** | 24.72 | 41% |
| **Full Reasoning** | 0.0465 | 0.0515 | **3.5×** | 22.35 | 46% |

*Note: The "Full Reasoning" method includes all reasoning tokens without removal, serving as an upper bound for performance but sacrificing correctness.*

#### Time Breakdown Analysis

| Method | Model Forward | Other Operations | Forward % of Total |
|:------:|:------------:|:----------------:|:------------------:|
| **Naive** | 0.1504s | 0.0119s | 92.7% |
| **KV Cache** | 0.1653s | 0.0151s | 91.6% |
| **SDPA** | 0.0403s | 0.0107s | 79.1% |
| **Flex** | 0.0497s | 0.0119s | 80.7% |

#### Numerical Accuracy vs Naive Baseline

| Method | RMSE ↓ | Symmetric KL Divergence ↓ | Top-1 Match ↑ | Top-8 Match ↑ |
|:------:|:------:|:-------------------------:|:-------------:|:-------------:|
| **KV Cache** | 0.0630 | 0.208 | 99.70% | 99.29% |
| **SDPA** | 0.0836 | 0.316 | 99.55% | 98.89% |
| **Flex** | 0.0835 | 0.314 | 99.55% | 98.89% |
| **Full Reasoning** | 0.9837 | **47.92** | 96.38% | 92.76% |

*↓ Lower is better, ↑ Higher is better*

### Key Observations

The benchmark results show some unexpected findings:

#### 1. KV Cache: Memory Efficient but Surprisingly Slow

Despite being the standard optimization for inference, **KV cache performed worse than the naive approach** (0.9× speed). This counterintuitive result stems from:
- **Sequential dependency**: Each turn must wait for the previous turn's cache update
- **Cache manipulation overhead**: Cropping and rebuilding the cache for reasoning removal adds non-trivial cost
- **Kernel switching**: Alternating between cached and non-cached forward passes prevents kernel optimization

While KV cache achieved 42% memory savings, the performance penalty makes it unsuitable for training workloads where both throughput and memory efficiency are important.

#### 2. Custom Attention Masks: The Performance Winners

Both SDPA and FlexAttention with custom 2D masks delivered **3× speedups** over the naive approach:
- **Single forward pass**: All turns processed together, maximizing GPU utilization
- **Optimized kernels**: Even with custom masks, these implementations leverage efficient attention kernels
- **Minimal overhead**: The mask construction cost is negligible compared to the forward pass savings

The similar performance between SDPA (3.2×) and Flex (2.6×) suggests that for moderate sequence lengths, the choice between them may depend more on implementation constraints than raw performance.

#### 3. Numerical Accuracy: All Methods Within Acceptable Bounds

The accuracy comparison shows that all correct methods maintain high fidelity:
- **KV Cache**: Best numerical accuracy with KL divergence of 0.208, reflecting near-identical computation paths
- **SDPA/Flex**: Slightly higher KL divergence (~0.31) but still excellent accuracy
- **Full Reasoning**: Dramatic accuracy drop with **48× higher KL divergence** (47.92 vs ~0.3) confirms that training with reasoning tokens fundamentally changes model behavior

The small differences between KV cache and custom mask approaches are primarily due to different CUDA kernels rather than algorithmic differences.

#### 4. Scaling Characteristics (From Extended Testing)

Additional experiments with varying conversation lengths revealed important scaling properties:

**Short conversations**:
- SDPA marginally outperforms FlexAttention
- All methods fit comfortably in memory

**Long conversations**:
- FlexAttention shows better performance
- KV cache memory usage grows faster than 2D custom attention masks as conversations get longer

### Implementation Strategy for VeRL

Based on these benchmarks, here's our implementation roadmap:

#### Phase 1: SDPA Custom Masks (Immediate Priority)

We'll start with SDPA custom mask support for several reasons:
- **Native compatibility**: SDPA is the default attention implementation for many modern models (including Qwen3)
- **Proven performance**: 3.2× speedup with minimal code changes

#### Phase 2: FlexAttention for Long Contexts (Future Enhancement)

While SDPA performs well for typical conversation lengths, agent RL often involves extremely long trajectories. FlexAttention becomes compelling for:
- **Ultra-long conversations**: Better memory access patterns at scale
- **Sparse attention patterns**: Can skip computation for masked regions entirely

Since FlexAttention shares much of the mask construction logic with SDPA, adding support will be straightforward once the SDPA implementation is stable.

<br>

---

[^1]: The benchmark dataset is available at [jybsuper/hermes-function-calling-thinking-V1-openai-format](https://huggingface.co/datasets/jybsuper/hermes-function-calling-thinking-V1-openai-format/blob/main/reasoning-func-call-valid.json)

