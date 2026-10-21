#!/usr/bin/env python3
"""
Run full eval suite on a checkpoint. Exits 1 on regression (for CI).
v4: also measures Reasoning-FLOP Efficiency (RFE) — the killer metric.
"""
import argparse, sys
sys.path.insert(0, ".")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--tasks", nargs="+", default=["mmlu", "arc_challenge", "gsm8k"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    p.add_argument("--fail-on-regression", action="store_true")
    p.add_argument("--output-html", default=None)
    p.add_argument("--rfe-profile", action="store_true",
                   help="Also measure Reasoning-FLOP Efficiency (killer metric). Requires --limit for speed.")
    p.add_argument("--rfe-output", default="rfe_report.json",
                   help="Path to write RFE JSON report")
    p.add_argument("--max-loops", type=int, default=16,
                   help="Max ACT loops for RFE measurement")
    args = p.parse_args()

    from open_mythos.main import OpenMythos
    print(f"[eval] Loading checkpoint: {args.checkpoint}")
    model = OpenMythos.from_pretrained(args.checkpoint, device=args.device)
    model.eval()

    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            class _Tok:
                eos_token_id = enc.eot_token
                def encode(self, t): return enc.encode(t)
                def decode(self, ids): return enc.decode(ids)
            tokenizer = _Tok()
        except ImportError:
            raise RuntimeError("Install tiktoken or pass --tokenizer path")

    # Standard benchmark suite
    from open_mythos.eval.benchmarks import BenchmarkEvaluator
    evaluator = BenchmarkEvaluator(model, tokenizer, device=args.device, batch_size=8)
    results = evaluator.run_all(
        tasks=args.tasks, limit=args.limit,
        fail_on_regression=args.fail_on_regression,
    )

    if args.output_html:
        evaluator.export_html_report(args.output_html)

    print("\nBenchmark Results:")
    for name, r in results.items():
        print(f"  {name:20s}: {r.ci_str}")

    # Reasoning-FLOP Efficiency measurement (the killer metric)
    if args.rfe_profile:
        print("\n" + "=" * 60)
        print("  Measuring Reasoning-FLOP Efficiency (RFE — killer metric)")
        print("=" * 60)
        from open_mythos.inference.profiler import InferenceProfiler

        # Use MMLU subset as the benchmark for RFE
        try:
            from datasets import load_dataset
            ds = load_dataset("cais/mmlu", "abstract_algebra", split="test")
            n_rfe = min(args.limit or 50, len(ds))
            prompts, truths = [], []
            choices_labels = ["A", "B", "C", "D"]
            for ex in list(ds)[:n_rfe]:
                ctx = f"Question: {ex['question']}\nChoices:\n"
                ctx += "\n".join(f"({c}) {o}" for c, o in zip(choices_labels, ex["choices"]))
                ctx += "\nAnswer: ("
                prompts.append(ctx)
                truths.append(choices_labels[ex["answer"]] + ")")

            profiler = InferenceProfiler(model, tokenizer, device=args.device, max_loops=args.max_loops)

            def _eval(generated, truth):
                return truth.strip() in generated[:10]

            report = profiler.profile_dataset(
                prompts=prompts,
                ground_truths=truths,
                eval_fn=_eval,
                max_new_tokens=8,
                n_loops=args.max_loops,
            )
            report.save_json(args.rfe_output)
            print(f"\n  RFE = {report.reasoning_flop_efficiency:.4e}")
            print(f"  Mean loops/token = {report.mean_loop_count:.2f} / {args.max_loops} max")
            print(f"  Compute utilization = {report.mean_loop_count/args.max_loops*100:.1f}%")
            print(f"\n  NOTE: Compare this RFE against GPT-4o (static model, fixed FLOPs/token).")
            print(f"  A higher RFE means more accuracy per FLOP — OpenMythos's core advantage.")

        except ImportError:
            print("  [RFE] Install 'datasets' package to run RFE profiling: pip install datasets")
        except Exception as e:
            print(f"  [RFE] Profile failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
