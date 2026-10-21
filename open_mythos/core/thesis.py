"""
OpenMythos Founder Thesis — The Central Insight
================================================

THESIS (one sentence):
    Recurrent depth is the only architecture that scales reasoning at
    constant parameter count — the same 3B model that answers "2+2" in
    1 loop answers a PhD-level proof in 64 loops, paying only for the
    compute it uses.

WHY THIS IS NOVEL:
    Every other LLM you can name (GPT-4, Claude, Gemini, Llama) is a
    "static width" model: the number of FLOPs per token is fixed by the
    architecture at training time. You cannot make them think harder on
    a hard problem without paying for a bigger model ALL THE TIME.

    OpenMythos separates two axes that are fused in every competitor:
        • Parameter count   → amortised over all queries (storage cost)
        • Compute depth     → paid per query (inference cost)

    This mirrors how human cognition works. The same brain that
    pattern-matches a face also proves theorems — it just runs for
    different durations. We call this property COMPUTE ELASTICITY.

THE 10X ADVANTAGE:
    • On easy queries  → 1–4 loops  → 4–16× cheaper than same-size static model
    • On hard queries  → 32–64 loops → first open model to reach o1-level
                         reasoning without distillation
    • Cost profile     → per-token billing maps directly to compute used,
                         not to model size — lowest cost at easy, peer at hard

DEFENSIBILITY:
    1. Architecture moat: ACT + LTI recurrence + MoE is a specific
       combination with non-trivial training dynamics. Competitors
       cannot copy the checkpoint, only the recipe — and must spend
       the same training compute to match.
    2. Data flywheel: elastic compute traces (loop counts per token)
       are a new kind of training signal. Every inference generates
       labeled difficulty data we use to improve routing.
    3. Open-weights trust: enterprises moving off OpenAI/Anthropic
       due to data privacy need an auditable, self-hostable model that
       can still do hard reasoning. OpenMythos is the only option.

KILLER METRIC:
    Reasoning-FLOP Efficiency = (MMLU-Pro accuracy) / (average FLOPs per token at inference)
    Target: 2× GPT-4o at equal accuracy on MMLU-Pro.
    Measurement: logged automatically by BenchmarkEvaluator.run_all() +
    InferenceProfiler.flop_profile().

WEDGE:
    Enterprise code review. A 3B model with elastic compute can do
    deep security analysis on complex diffs (64 loops) while answering
    trivial style nits in 1 loop — paying proportionally. No competitor
    offers this. First 50 design-partner contracts win the narrative.

WHY USERS SWITCH:
    1. API cost: easy-query savings of 4–16× are immediately measurable.
    2. Hard-query quality: one open model that matches o1 on math/code.
    3. Privacy: self-hostable; no data leaves the customer's VPC.
    4. Transparency: loop traces expose model reasoning — enterprise
       compliance teams love auditable AI decisions.
"""

__thesis__ = (
    "Recurrent depth is the only architecture that scales reasoning at constant "
    "parameter count — the same model thinks shallow on easy queries and deep on hard ones, "
    "paying only for the compute it uses."
)

__killer_metric__ = "reasoning_flop_efficiency"  # accuracy / flops_per_token
__wedge__ = "enterprise_code_review"
__moat__ = ["architecture", "elastic_compute_traces", "open_weights_trust"]
