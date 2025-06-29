---
title: "Efficient Forward Pass for Agent RL: Solving Multi-Turn Context Consistency"
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

## Solution 1: Turn-by-Turn Forward Passes

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

While this guarantees training-inference consistency, it comes with significant costs:
- **Computational overhead**: Redundant processing of shared chat history across turns
- **Implementation complexity**: Requires a different training loop pattern compared to standard single-turn or packed multi-turn training

<br>

---
[^1]: See my previous post on [When Reasoning Models Break Tokenization: The Hidden Complexity of Multiturn Training](https://jybsuper.github.io/posts/multiturn_tokenization/)
