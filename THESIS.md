# OpenMythos — Founder Thesis

## The One-Sentence Thesis

> **Recurrent depth is the only architecture that scales reasoning at constant parameter count — the same 3B model that answers "2+2" in 1 loop answers a PhD-level proof in 64 loops, paying only for the compute it uses.**

---

## Why This Is the Right Bet

Every LLM you can name today — GPT-4, Claude, Gemini, Llama, DeepSeek — is a **static-width model**. The number of FLOPs burned per token is determined at architecture design time. You cannot make them think harder on a hard problem without loading a bigger model all the time.

This is not how cognition works. The same human brain that recognizes a face also proves theorems. It just runs longer on the harder task.

OpenMythos separates two axes that every competitor fuses together:

| Axis | Competitor | OpenMythos |
|------|-----------|------------|
| Parameter count | Determines both storage AND compute cost | Determines storage cost only |
| Compute depth | Fixed at architecture time | Variable per query via ACT halting |

We call this property **Compute Elasticity**.

---

## The 10× Advantage

| Query type | Competitor cost | OpenMythos cost | Ratio |
|-----------|----------------|-----------------|-------|
| "What is 2+2?" | 1× (full model always) | 0.06× (1 loop / 16 max) | **16× cheaper** |
| "Explain RLHF" | 1× | 0.25× (4 loops) | **4× cheaper** |
| PhD-level proof | 1× (but wrong) | 1× (64 loops, correct) | **Same cost, better answer** |

The business model implication: **we bill per compute unit used, not per token**. Easy queries generate large margins. Hard queries generate fair margins. No competitor can offer this.

---

## The Killer Metric

**Reasoning-FLOP Efficiency (RFE)** = benchmark_accuracy / mean_FLOPs_per_token

Measured automatically by `InferenceProfiler.profile_dataset()`.

Current target: **RFE of 2× GPT-4o at equal MMLU-Pro accuracy**.

This is the single number we optimize for and report to investors, customers, and in benchmarks. It is not vanity — it directly predicts our gross margin advantage over API competitors.

---

## Defensibility

1. **Architecture moat**: ACT + LTI recurrence + DeepSeek-MoE is a specific combination with non-trivial training dynamics. Competitors cannot copy our checkpoint — they must spend equal training compute to match.

2. **Data flywheel**: Elastic compute traces (loop count per token per query) are a new category of training signal. Every inference generates labeled difficulty data we use to improve our routing. Competitors running static models cannot generate this data.

3. **Open-weights trust**: Enterprise customers moving off OpenAI/Anthropic for data privacy reasons need an auditable, self-hostable model that can still do hard reasoning. We are the only option.

4. **Alignment advantage**: Our per-token compute elasticity makes chain-of-thought and constitutional AI methods strictly more powerful — the model can allocate more loops to uncertainty resolution steps automatically.

---

## The Wedge Market

**Enterprise code review.**

- A 3B model with elastic compute does deep security analysis on complex diffs (64 loops) while answering trivial style nits in 1 loop — paying proportionally.
- No competitor offers this pricing model.
- Security engineers understand and value it immediately.
- First 50 design-partner contracts win the narrative and generate the training data.

Adjacent wedges after code review: legal document analysis, scientific literature synthesis, math tutoring.

---

## Why Users Switch

1. **Cost**: Easy-query savings of 4–16× are immediately measurable in billing dashboards.
2. **Hard-query quality**: First open model that matches o1 on math/code without distillation.
3. **Privacy**: Self-hostable; no data leaves the customer's VPC.
4. **Transparency**: Loop-count traces expose model reasoning per token — enterprise compliance teams love auditable AI decisions.
5. **Latency control**: Customers can cap `max_loops` to bound worst-case latency — no other model offers this.

---

## Why Competitors Cannot Copy This Quickly

1. Requires re-training from scratch (not fine-tuning) to learn elastic depth.
2. The LTI injection + ACT halting combination has subtle stability requirements during training that are not obvious from the paper.
3. Our difficulty-labeled inference traces create a compounding advantage — the more we serve, the better our routing.
4. Open-weights release creates a community moat before any closed competitor ships a response.

---

## Architecture Summary

```
Input
  │
  ▼ Prelude Layers (dense, warm up representations)
  │
  ▼ RecurrentBlock (elastic depth: 1–64 iterations)
  │    ├── TransformerBlock (MLA attention + MoEFFN)
  │    ├── LTI Injection    (stable recurrent state update)
  │    ├── ACT Halting      (per-token early exit)
  │    └── LoRA Adapter     (loop-index conditioning)
  │
  ▼ Coda Layers (dense, refine output)
  │
  ▼ LM Head (tied embeddings)
```

Key properties:
- **MLA attention** (DeepSeek-style): 4–8× KV cache compression vs standard GQA
- **DeepSeek MoE**: 64 experts, 4 active per token → 4× parameter count at same active compute
- **ACT halting**: Adaptive Computation Time — halts early on easy tokens
- **LTI injection**: Stable linear time-invariant recurrent update (no exploding gradients)
- **LoRA conditioning**: Loop index embedding → model knows which iteration it is on

---

## Roadmap to Category Leadership

| Milestone | Target | Metric |
|-----------|--------|--------|
| v1.0 (now) | 3B model, architecture proven | MMLU 72%, GSM8K 78% |
| v1.5 (Q2) | RFE benchmark published | 2× GPT-4o RFE |
| v2.0 (Q3) | 50 design partners, code review product | $500k ARR |
| v3.0 (Q4) | 7B model, first open o1-competitor | AMC-12 50%+ |
| v4.0 (Q1+1) | Elastic billing API, 1000 customers | $5M ARR |
