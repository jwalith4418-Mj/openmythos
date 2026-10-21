#!/usr/bin/env python3
"""Generate HTML + Markdown benchmark report from SQLite eval database."""

import argparse
import sys

sys.path.insert(0, ".")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="eval_results.db")
    p.add_argument("--output", default="eval_report.html")
    p.add_argument("--markdown", default="eval_report.md")
    args = p.parse_args()

    from open_mythos.eval.benchmarks import BenchmarkEvaluator

    # Dummy model/tokenizer — we only need the DB access and report methods
    class _Dummy:
        pass

    ev = BenchmarkEvaluator.__new__(BenchmarkEvaluator)
    ev.db_path = args.db
    ev.export_html_report(args.output)

    import sqlite3

    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        "SELECT step, benchmark, accuracy, ci_lower, ci_upper, timestamp FROM results ORDER BY step DESC"
    ).fetchall()
    conn.close()
    md = "# OpenMythos Benchmark Report\n\n| Step | Benchmark | Accuracy | 95% CI | Timestamp |\n|------|-----------|----------|--------|----------|\n"
    for step, bm, acc, ci_lo, ci_hi, ts in rows:
        md += f"| {step} | {bm} | {acc*100:.1f}% | [{ci_lo*100:.1f}, {ci_hi*100:.1f}] | {ts} |\n"
    with open(args.markdown, "w") as f:
        f.write(md)
    print(f"Reports written: {args.output}, {args.markdown}")


if __name__ == "__main__":
    main()
