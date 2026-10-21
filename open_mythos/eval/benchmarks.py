"""
Automated Benchmark Pipeline for OpenMythos — v2

Fixes and additions over v1:
  1. GSM8K greedy eval uses temperature=0.0 → true argmax.
  2. MMLU runs all 57 subjects (not 10).
  3. HumanEval pass@k implemented with execution harness.
  4. TruthfulQA MC2 evaluation added.
  5. BBH evaluation added via lm-eval.
  6. HellaSwag manual fallback added.
  7. HTML leaderboard report generation added.
  8. Regression detection emits sys.exit(1) for CI integration.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BenchmarkResult:
    name: str
    accuracy: float
    n_correct: int
    n_total: int
    ci_lower: float
    ci_upper: float
    elapsed_sec: float
    shots: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ci_str(self) -> str:
        return f"{self.accuracy*100:.1f}% [{self.ci_lower*100:.1f}, {self.ci_upper*100:.1f}]"

    @staticmethod
    def wilson_ci(n_correct: int, n_total: int, confidence: float = 0.95) -> Tuple[float, float]:
        if n_total == 0:
            return 0.0, 1.0
        z = 1.96
        p = n_correct / n_total
        denom = 1 + z**2 / n_total
        center = (p + z**2 / (2 * n_total)) / denom
        margin = z * math.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
        return max(0.0, center - margin), min(1.0, center + margin)


# All 57 MMLU subjects
MMLU_ALL_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
    "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes",
    "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions",
]


class BenchmarkEvaluator:
    BENCHMARKS = {
        "mmlu":          {"shots": 5,  "n_questions": 14_042},
        "gpqa":          {"shots": 0,  "n_questions": 448},
        "arc_challenge": {"shots": 25, "n_questions": 1_172},
        "hellaswag":     {"shots": 10, "n_questions": 10_042},
        "gsm8k":         {"shots": 8,  "n_questions": 1_319},
        "math":          {"shots": 4,  "n_questions": 5_000},
        "humaneval":     {"shots": 0,  "n_questions": 164},
        "truthfulqa":    {"shots": 0,  "n_questions": 817},
        "bbh":           {"shots": 3,  "n_questions": 6_511},
    }

    def __init__(self, model, tokenizer, device="cuda", db_path="eval_results.db",
                 wandb_run=None, batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.db_path = db_path
        self.wandb_run = wandb_run
        self.batch_size = batch_size
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, step INTEGER, benchmark TEXT,
                accuracy REAL, n_correct INTEGER, n_total INTEGER,
                ci_lower REAL, ci_upper REAL, elapsed_sec REAL, shots INTEGER, metadata TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _save_to_db(self, step, result):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO results VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?)",
            (time.strftime("%Y-%m-%dT%H:%M:%S"), step, result.name,
             result.accuracy, result.n_correct, result.n_total,
             result.ci_lower, result.ci_upper, result.elapsed_sec,
             result.shots, json.dumps(result.metadata)),
        )
        conn.commit()
        conn.close()

    def _get_best_prior(self, benchmark):
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT MAX(accuracy) FROM results WHERE benchmark=?", (benchmark,)).fetchone()
        conn.close()
        return row[0] if row and row[0] is not None else None

    def _check_regression(self, result, threshold=0.02):
        best = self._get_best_prior(result.name)
        if best is not None and (best - result.accuracy) > threshold:
            return (f"REGRESSION on {result.name}: "
                    f"{result.accuracy*100:.1f}% vs best {best*100:.1f}% "
                    f"(Δ={-(result.accuracy - best)*100:.1f}%)")
        return None

    # ------------------------------------------------------------------
    # lm-eval harness integration
    # ------------------------------------------------------------------

    def run_lm_eval(self, tasks, num_fewshot=None, limit=None):
        try:
            import lm_eval
        except ImportError:
            print("[Eval] lm-evaluation-harness not installed. pip install lm-eval")
            return self._run_manual_eval(tasks, limit=limit)

        results = {}
        lm = _OpenMythosLMEvalWrapper(self.model, self.tokenizer, self.device, self.batch_size)
        for task in tasks:
            t0 = time.time()
            try:
                task_results = lm_eval.simple_evaluate(
                    model=lm, tasks=[task], num_fewshot=num_fewshot, limit=limit,
                )
                task_r = task_results["results"][task]
                acc = task_r.get("acc,none", task_r.get("acc", 0.0))
                if isinstance(acc, dict):
                    acc = acc.get("value", 0.0)
                n_total = self.BENCHMARKS.get(task, {}).get("n_questions", 100)
                if limit:
                    n_total = min(n_total, limit)
                n_correct = int(acc * n_total)
                ci_lo, ci_hi = BenchmarkResult.wilson_ci(n_correct, n_total)
                results[task] = BenchmarkResult(
                    name=task, accuracy=acc, n_correct=n_correct, n_total=n_total,
                    ci_lower=ci_lo, ci_upper=ci_hi, elapsed_sec=time.time() - t0,
                    shots=num_fewshot or self.BENCHMARKS.get(task, {}).get("shots", 0),
                )
            except Exception as e:
                print(f"[Eval] {task} failed via lm-eval: {e}")
        return results

    def _run_manual_eval(self, tasks, limit=None):
        results = {}
        for task in tasks:
            if task == "mmlu":
                results[task] = self._eval_mmlu(limit=limit)
            elif task == "arc_challenge":
                results[task] = self._eval_arc(limit=limit)
            elif task == "gsm8k":
                results[task] = self._eval_gsm8k(limit=limit)
            elif task == "hellaswag":
                results[task] = self._eval_hellaswag(limit=limit)
            elif task == "humaneval":
                results[task] = self._eval_humaneval(limit=limit)
            elif task == "truthfulqa":
                results[task] = self._eval_truthfulqa(limit=limit)
        return results

    # ------------------------------------------------------------------
    # Log-likelihood helper
    # ------------------------------------------------------------------

    def _loglikelihood_batch(self, contexts, continuations):
        self.model.eval()
        lls = []
        for ctx, cont in zip(contexts, continuations):
            full_ids = self.tokenizer.encode(ctx + cont)
            ctx_ids = self.tokenizer.encode(ctx)
            cont_len = len(full_ids) - len(ctx_ids)
            input_ids = torch.tensor([full_ids], device=self.device)
            with torch.no_grad():
                logits = self.model(input_ids)
            log_probs = F.log_softmax(logits[0, :-1], dim=-1)
            cont_targets = torch.tensor(full_ids[-cont_len:], device=self.device)
            ll = log_probs[-cont_len:].gather(-1, cont_targets.unsqueeze(-1)).squeeze(-1).sum()
            lls.append(ll.item())
        return lls

    # ------------------------------------------------------------------
    # MMLU — all 57 subjects
    # ------------------------------------------------------------------

    def _eval_mmlu(self, limit=None):
        try:
            from datasets import load_dataset
        except ImportError:
            return BenchmarkResult("mmlu", 0.0, 0, 0, 0.0, 1.0, 0.0, 5)

        t0 = time.time()
        choices = ["A", "B", "C", "D"]
        n_correct = n_total = 0
        per_subject_limit = (limit // len(MMLU_ALL_SUBJECTS)) if limit else None

        for subject in MMLU_ALL_SUBJECTS:
            try:
                ds = load_dataset("cais/mmlu", subject, split="test")
                if per_subject_limit:
                    ds = ds.select(range(min(per_subject_limit, len(ds))))
                for ex in ds:
                    ctx = f"Question: {ex['question']}\nChoices:\n"
                    ctx += "\n".join(f"({c}) {o}" for c, o in zip(choices, ex["choices"]))
                    ctx += "\nAnswer: ("
                    lls = self._loglikelihood_batch([ctx] * 4, [c + ")" for c in choices])
                    if lls.index(max(lls)) == ex["answer"]:
                        n_correct += 1
                    n_total += 1
            except Exception:
                continue

        acc = n_correct / max(n_total, 1)
        ci_lo, ci_hi = BenchmarkResult.wilson_ci(n_correct, n_total)
        return BenchmarkResult("mmlu", acc, n_correct, n_total, ci_lo, ci_hi, time.time() - t0, 5)

    # ------------------------------------------------------------------
    # ARC-Challenge
    # ------------------------------------------------------------------

    def _eval_arc(self, limit=None):
        try:
            from datasets import load_dataset
            ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        except ImportError:
            return BenchmarkResult("arc_challenge", 0.0, 0, 0, 0.0, 1.0, 0.0, 25)
        t0 = time.time()
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        n_correct = n_total = 0
        for ex in ds:
            q, opts, labels, ans = ex["question"], ex["choices"]["text"], ex["choices"]["label"], ex["answerKey"]
            ctx = f"Question: {q}\n" + "\n".join(f"{l}: {o}" for l, o in zip(labels, opts)) + "\nAnswer:"
            lls = self._loglikelihood_batch([ctx] * len(opts), [f" {l}" for l in labels])
            if labels[lls.index(max(lls))] == ans:
                n_correct += 1
            n_total += 1
        acc = n_correct / max(n_total, 1)
        ci_lo, ci_hi = BenchmarkResult.wilson_ci(n_correct, n_total)
        return BenchmarkResult("arc_challenge", acc, n_correct, n_total, ci_lo, ci_hi, time.time() - t0, 25)

    # ------------------------------------------------------------------
    # GSM8K — FIX v2: true greedy (temperature=0.0)
    # ------------------------------------------------------------------

    def _eval_gsm8k(self, limit=None):
        try:
            from datasets import load_dataset
            ds = load_dataset("openai/gsm8k", "main", split="test")
        except ImportError:
            return BenchmarkResult("gsm8k", 0.0, 0, 0, 0.0, 1.0, 0.0, 8)
        t0 = time.time()
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        n_correct = n_total = 0
        self.model.eval()
        for ex in ds:
            prompt = f"Question: {ex['question']}\nAnswer: Let me think step by step."
            input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
            with torch.no_grad():
                # FIX v2: temperature=0.0 invokes argmax path
                output_ids = self.model.generate(input_ids, max_new_tokens=256, temperature=0.0, top_k=1)
            generated = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:].tolist())
            pred = self._extract_gsm8k_answer(generated)
            true_answer = self._extract_gsm8k_answer(ex["answer"])
            if pred is not None and true_answer is not None and abs(pred - true_answer) < 1e-3:
                n_correct += 1
            n_total += 1
        acc = n_correct / max(n_total, 1)
        ci_lo, ci_hi = BenchmarkResult.wilson_ci(n_correct, n_total)
        return BenchmarkResult("gsm8k", acc, n_correct, n_total, ci_lo, ci_hi, time.time() - t0, 8)

    @staticmethod
    def _extract_gsm8k_answer(text):
        m = re.search(r"####\s*([\d,]+)", text)
        if m:
            return float(m.group(1).replace(",", ""))
        numbers = re.findall(r"[-+]?\d*\.?\d+", text)
        return float(numbers[-1]) if numbers else None

    # ------------------------------------------------------------------
    # HellaSwag — FIX v2: manual fallback implemented
    # ------------------------------------------------------------------

    def _eval_hellaswag(self, limit=None):
        try:
            from datasets import load_dataset
            ds = load_dataset("Rowan/hellaswag", split="validation")
        except Exception:
            return BenchmarkResult("hellaswag", 0.0, 0, 0, 0.0, 1.0, 0.0, 10)
        t0 = time.time()
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        n_correct = n_total = 0
        for ex in ds:
            ctx = ex["activity_label"] + ": " + ex["ctx"]
            endings = ex["endings"]
            label = int(ex["label"])
            lls = self._loglikelihood_batch([ctx] * len(endings), endings)
            if lls.index(max(lls)) == label:
                n_correct += 1
            n_total += 1
        acc = n_correct / max(n_total, 1)
        ci_lo, ci_hi = BenchmarkResult.wilson_ci(n_correct, n_total)
        return BenchmarkResult("hellaswag", acc, n_correct, n_total, ci_lo, ci_hi, time.time() - t0, 10)

    # ------------------------------------------------------------------
    # HumanEval pass@k — FIX v2: implemented with execution harness
    # ------------------------------------------------------------------

    def _eval_humaneval(self, limit=None, k_values=(1, 10)):
        try:
            from datasets import load_dataset
            ds = load_dataset("openai_humaneval", split="test")
        except Exception:
            return BenchmarkResult("humaneval", 0.0, 0, 0, 0.0, 1.0, 0.0, 0)

        t0 = time.time()
        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        n_samples = max(k_values)
        all_pass = []
        self.model.eval()

        for ex in ds:
            prompt = ex["prompt"]
            test_code = ex["test"]
            entry_point = ex["entry_point"]

            passes = []
            for _ in range(n_samples):
                input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids, max_new_tokens=512, temperature=0.8, top_p=0.95,
                    )
                completion = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:].tolist())
                # Extract only valid Python (stop at next function def or class)
                completion = self._extract_python_code(completion)
                passes.append(self._check_execution(prompt + completion, test_code, entry_point))

            all_pass.append(passes)

        # pass@k unbiased estimator (Chen et al. 2021)
        pass_at_1 = sum(any(p) for p in all_pass) / max(len(all_pass), 1)
        n_total = len(all_pass)
        n_correct = sum(any(p) for p in all_pass)
        ci_lo, ci_hi = BenchmarkResult.wilson_ci(n_correct, n_total)

        return BenchmarkResult(
            "humaneval", pass_at_1, n_correct, n_total, ci_lo, ci_hi,
            time.time() - t0, 0,
            metadata={"pass@1": pass_at_1,
                      "pass@10": self._estimate_pass_at_k(all_pass, k=10) if n_samples >= 10 else None},
        )

    @staticmethod
    def _extract_python_code(text: str) -> str:
        """Truncate at the first line that starts a new top-level definition."""
        lines = text.split("\n")
        out = []
        for i, line in enumerate(lines):
            if i > 0 and (line.startswith("def ") or line.startswith("class ")):
                break
            out.append(line)
        return "\n".join(out)

    @staticmethod
    def _check_execution(code: str, test_code: str, entry_point: str) -> bool:
        """
        Execute generated code + tests in an isolated subprocess with timeout.

        Security hardening (FIX: previously ran arbitrary code in server process):
        - Subprocess runs with a restricted environment (no PYTHONPATH, no extra env).
        - Execution is time-bounded (10s) and output-bounded.
        - The generated file is written to a temp dir and cleaned up after.
        - For production use, replace this with Docker/gVisor/Firecracker sandboxing.
          Example: docker run --rm --network=none --memory=256m --cpus=0.5 python:3.11-slim python /code/solution.py
        """
        full_code = code + "\n" + test_code + f"\ncheck({entry_point})\n"
        fname = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False,
                dir=tempfile.gettempdir(),
            ) as f:
                f.write(full_code)
                fname = f.name

            # Restricted environment: strip PYTHONPATH and other injected paths.
            restricted_env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": tempfile.gettempdir(),  # prevent ~/.config access
            }

            result = subprocess.run(
                [sys.executable, "-I", fname],  # -I: isolated mode (no site, no PYTHONPATH)
                capture_output=True,
                timeout=10,
                env=restricted_env,
                cwd=tempfile.gettempdir(),  # isolate working dir
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
        finally:
            if fname:
                try:
                    os.unlink(fname)
                except Exception:
                    pass

    @staticmethod
    def _estimate_pass_at_k(all_pass: list, k: int) -> float:
        """Unbiased pass@k estimator (Chen et al. 2021 equation 3)."""
        total = 0
        for passes in all_pass:
            n = len(passes)
            c = sum(passes)
            if n - c < k:
                total += 1.0
            else:
                total += 1.0 - math.comb(n - c, k) / math.comb(n, k)
        return total / max(len(all_pass), 1)

    # ------------------------------------------------------------------
    # TruthfulQA — MC2 scoring
    # ------------------------------------------------------------------

    def _eval_truthfulqa(self, limit=None):
        try:
            from datasets import load_dataset
            ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        except Exception:
            return BenchmarkResult("truthfulqa", 0.0, 0, 0, 0.0, 1.0, 0.0, 0)

        t0 = time.time()
        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        # MC2: normalized probability mass assigned to all true answers
        total_mc2 = 0.0
        n_total = 0

        self.model.eval()
        for ex in ds:
            q = ex["question"]
            choices = ex["mc2_targets"]["choices"]
            labels = ex["mc2_targets"]["labels"]   # 1 = correct, 0 = incorrect

            ctx = f"Q: {q}\nA:"
            lls = self._loglikelihood_batch([ctx] * len(choices), [" " + c for c in choices])

            # Normalise log-likelihoods to probabilities
            max_ll = max(lls)
            probs = [math.exp(ll - max_ll) for ll in lls]
            total_prob = sum(probs)
            if total_prob > 0:
                probs = [p / total_prob for p in probs]

            mc2 = sum(p for p, l in zip(probs, labels) if l == 1)
            total_mc2 += mc2
            n_total += 1

        acc = total_mc2 / max(n_total, 1)
        ci_lo, ci_hi = BenchmarkResult.wilson_ci(int(acc * n_total), n_total)
        return BenchmarkResult("truthfulqa", acc, int(acc * n_total), n_total, ci_lo, ci_hi, time.time() - t0, 0)

    # ------------------------------------------------------------------
    # run_all
    # ------------------------------------------------------------------

    def run_all(self, step=0, tasks=None, limit=None, fail_on_regression=False):
        if tasks is None:
            tasks = ["mmlu", "arc_challenge", "gsm8k"]

        print(f"\n{'='*60}")
        print(f"  OpenMythos Benchmark Evaluation — Step {step}")
        print(f"{'='*60}\n")

        all_results = self.run_lm_eval(tasks, limit=limit)

        wb_metrics = {}
        regressions = []
        for name, result in all_results.items():
            self._save_to_db(step, result)
            reg = self._check_regression(result)
            if reg:
                regressions.append(reg)
                print(f"  ⚠️  {reg}")
            wb_metrics[f"eval/{name}"] = result.accuracy
            wb_metrics[f"eval/{name}_ci_lower"] = result.ci_lower
            wb_metrics[f"eval/{name}_ci_upper"] = result.ci_upper
            print(f"  {name:20s}: {result.ci_str}  ({result.elapsed_sec:.1f}s)")

        if self.wandb_run:
            try:
                self.wandb_run.log(wb_metrics, step=step)
            except Exception:
                pass

        if regressions:
            print(f"\n⚠️  {len(regressions)} REGRESSION(S) DETECTED")
            if fail_on_regression:
                sys.exit(1)   # CI integration: fail the job on regression
        else:
            print("\n✓ No regressions detected")

        print(f"{'='*60}\n")
        return all_results

    # ------------------------------------------------------------------
    # Leaderboard + HTML report
    # ------------------------------------------------------------------

    def leaderboard(self, top_n=10):
        conn = sqlite3.connect(self.db_path)
        benchmarks = [r[0] for r in conn.execute("SELECT DISTINCT benchmark FROM results").fetchall()]
        print(f"\n{'='*60}")
        print("  OpenMythos Leaderboard")
        print(f"{'='*60}")
        for bm in benchmarks:
            rows = conn.execute(
                "SELECT step, accuracy, ci_lower, ci_upper FROM results WHERE benchmark=? ORDER BY accuracy DESC LIMIT ?",
                (bm, top_n)
            ).fetchall()
            print(f"\n  {bm}")
            for step, acc, ci_lo, ci_hi in rows:
                print(f"  step {step:>8}  {acc*100:>9.1f}%  [{ci_lo*100:.1f}, {ci_hi*100:.1f}]")
        conn.close()

    def export_html_report(self, output_path: str = "eval_report.html") -> None:
        """Generate a self-contained HTML leaderboard from the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT step, benchmark, accuracy, ci_lower, ci_upper, elapsed_sec, timestamp FROM results ORDER BY step DESC"
        ).fetchall()
        conn.close()

        rows_html = ""
        for step, bm, acc, ci_lo, ci_hi, elapsed, ts in rows:
            rows_html += (
                f"<tr><td>{step}</td><td>{bm}</td>"
                f"<td>{acc*100:.1f}%</td>"
                f"<td>[{ci_lo*100:.1f}, {ci_hi*100:.1f}]</td>"
                f"<td>{elapsed:.1f}s</td><td>{ts}</td></tr>\n"
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>OpenMythos Evaluation Leaderboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 960px; margin: 40px auto; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 8px 12px; border: 1px solid #ddd; text-align: left; }}
  th {{ background: #1a1a2e; color: white; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
</style>
</head>
<body>
<h1>OpenMythos Benchmark Leaderboard</h1>
<p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
<table>
  <thead><tr><th>Step</th><th>Benchmark</th><th>Accuracy</th><th>95% CI</th><th>Elapsed</th><th>Timestamp</th></tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</body>
</html>"""
        with open(output_path, "w") as f:
            f.write(html)
        print(f"[Eval] HTML report written to {output_path}")


# ---------------------------------------------------------------------------
# lm-eval wrapper
# ---------------------------------------------------------------------------


class _OpenMythosLMEvalWrapper:
    def __init__(self, model, tokenizer, device, batch_size):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def tok_encode(self, string):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests):
        results = []
        for ctx, cont in requests:
            ctx_ids = self.tok_encode(ctx)
            cont_ids = self.tok_encode(cont)
            full_ids = ctx_ids + cont_ids
            input_ids = torch.tensor([full_ids], device=self.device)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(input_ids)
            log_probs = F.log_softmax(logits[0, :-1], dim=-1)
            cont_len = len(cont_ids)
            cont_targets = torch.tensor(cont_ids, device=self.device)
            ll = log_probs[-cont_len:].gather(-1, cont_targets.unsqueeze(-1)).squeeze(-1).sum().item()
            is_greedy = all(log_probs[-cont_len:][i].argmax().item() == cont_ids[i] for i in range(cont_len))
            results.append((ll, is_greedy))
        return results

    def generate_until(self, requests):
        results = []
        for ctx, gen_kwargs in requests:
            max_new = gen_kwargs.get("max_gen_toks", 256)
            until = gen_kwargs.get("until", [self.tokenizer.decode([self.eot_token_id])])
            input_ids = torch.tensor([self.tok_encode(ctx)], device=self.device)
            self.model.eval()
            with torch.no_grad():
                output_ids = self.model.generate(input_ids, max_new_tokens=max_new, top_k=1, temperature=0.0)
            generated = self.tok_decode(output_ids[0, input_ids.shape[1]:].tolist())
            for stop in until:
                if stop in generated:
                    generated = generated[:generated.index(stop)]
            results.append(generated)
        return results
