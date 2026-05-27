#!/usr/bin/env python3
"""Publish finished GDsuite Delphi eval results to the HuggingFace Hub.

Scans the eval output tree, keeps only models with a complete set of
result JSONs, computes a tidy per-task metrics summary, and uploads both
the summary and the raw JSONs to a HF dataset repo.

Needs a HF token with write access to the target namespace — run
`huggingface-cli login` as the WillHeld account first (or set HF_TOKEN).

Usage:
    python push_results_to_hf.py                         # push, private repo
    python push_results_to_hf.py --public
    python push_results_to_hf.py --dry-run               # build, don't upload
    python push_results_to_hf.py --repo WillHeld/my-name \
        --results-dir /sphinx/u/salt-checkpoints/delphi-outputs
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile

DONE_JSON_COUNT = 34
PERSONA_FAMILY = "multihop_persona_qa"
DEFAULT_REPO = "WillHeld/gdsuite-delphi-result"
DEFAULT_RESULTS = os.environ.get(
    "OUTPUT_ROOT", "/sphinx/u/salt-checkpoints/delphi-outputs")


def _read_json(path: str):
    with open(path) as f:
        return json.load(f)


def _json_files(root: str) -> list[str]:
    out = []
    for dirpath, _, files in os.walk(root):
        out += [os.path.join(dirpath, f) for f in files if f.endswith(".json")]
    return out


def _hard_acc(rows: list[dict]) -> float | None:
    """Fraction of items with P(correct) > P(incorrect) — mirrors
    run_eval.py._hard_acc. Higher ⇒ the model resists the misleading
    pattern."""
    if not rows:
        return None
    hits = sum(1 for r in rows
               if r.get("correct_avg_prob", 0.0)
               > r.get("incorrect_avg_prob", 0.0))
    return hits / len(rows)


def _mean_prob_margin(rows: list[dict]) -> float | None:
    """Mean P(correct) - P(incorrect). Higher ⇒ a softer preference for
    the generalizing answer, retaining confidence information that
    `hard_acc` discards."""
    if not rows:
        return None
    return sum(
        r.get("correct_avg_prob", 0.0) - r.get("incorrect_avg_prob", 0.0)
        for r in rows
    ) / len(rows)


def _mean_log_prob_margin(rows: list[dict]) -> float | None:
    """Mean log P(correct) - log P(incorrect). This is usually the most
    scaling-law-friendly soft metric because it is an unbounded margin."""
    if not rows:
        return None
    return sum(
        r.get("correct_log_prob", float("-inf"))
        - r.get("incorrect_log_prob", float("-inf"))
        for r in rows
    ) / len(rows)


def _mean_correct_log_prob(rows: list[dict]) -> float | None:
    """Mean teacher-forced log probability of the correct answer."""
    if not rows:
        return None
    return sum(r.get("correct_log_prob", float("-inf")) for r in rows) / len(rows)


def _logsumexp(a: float, b: float) -> float:
    m = max(a, b)
    if math.isinf(m):
        return m
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _mean_log_normalized_correct_prob(rows: list[dict]) -> float | None:
    """Mean log normalized probability of the correct answer among the
    two scored choices: log P(correct) - logsumexp(log P(correct),
    log P(incorrect))."""
    if not rows:
        return None
    vals = []
    for r in rows:
        correct = r.get("correct_log_prob", float("-inf"))
        incorrect = r.get("incorrect_log_prob", float("-inf"))
        vals.append(correct - _logsumexp(correct, incorrect))
    return sum(vals) / len(vals)


def _mean_normalized_correct_prob(rows: list[dict]) -> float | None:
    """Mean normalized probability of the correct answer among the two
    scored choices."""
    if not rows:
        return None
    vals = []
    for r in rows:
        correct = r.get("correct_log_prob", float("-inf"))
        incorrect = r.get("incorrect_log_prob", float("-inf"))
        vals.append(math.exp(correct - _logsumexp(correct, incorrect)))
    return sum(vals) / len(vals)


def summarize_model(model: str, model_dir: str) -> list[dict]:
    """Tidy rows for one model: one per (family, task). The 5 logprob
    families score `hard_acc`; multihop_persona_qa scores `match_rate`
    per (persona, question), averaged across seeds."""
    rows: list[dict] = []
    for family in sorted(os.listdir(model_dir)):
        fam_dir = os.path.join(model_dir, family)
        if not os.path.isdir(fam_dir):
            continue
        if family == PERSONA_FAMILY:
            for fn in sorted(os.listdir(fam_dir)):
                if not fn.endswith("_summary.json"):
                    continue
                persona = fn[:-len("_summary.json")]
                # summary file: {seed: {question_id: match_rate}}
                per_q: dict[str, list[float]] = {}
                for seed_map in _read_json(os.path.join(fam_dir, fn)).values():
                    for qid, rate in seed_map.items():
                        per_q.setdefault(qid, []).append(rate)
                for qid, vals in sorted(per_q.items()):
                    rows.append({
                        "model": model, "family": family,
                        "task": f"{persona}/{qid}",
                        "metric": "match_rate",
                        "score": sum(vals) / len(vals),
                        "n": len(vals),
                    })
        else:
            for path in sorted(_json_files(fam_dir)):
                data = _read_json(path)
                if not isinstance(data, list):
                    continue
                task = os.path.relpath(path, fam_dir)[:-len(".json")]
                for metric, score in (
                    ("hard_acc", _hard_acc(data)),
                    ("correct_log_prob", _mean_correct_log_prob(data)),
                    ("prob_margin", _mean_prob_margin(data)),
                    ("log_prob_margin", _mean_log_prob_margin(data)),
                    ("log_normalized_correct_prob", _mean_log_normalized_correct_prob(data)),
                    ("normalized_correct_prob", _mean_normalized_correct_prob(data)),
                ):
                    rows.append({
                        "model": model, "family": family, "task": task,
                        "metric": metric,
                        "score": score,
                        "n": len(data),
                    })
    return rows


def _readme(finished: list[tuple[str, int]], n_rows: int) -> str:
    models_md = "\n".join(f"- `{m}`" for m, _ in finished)
    return f"""---
license: apache-2.0
tags:
- generalization-dynamics
- evaluation
pretty_name: GDsuite results — Delphi collection
---

# GDsuite results — Delphi model collection

[GDsuite](https://github.com/Jiaxin-Wen/GDsuite) evaluation results for
the [marin-community/delphi](https://huggingface.co/collections/marin-community/delphi)
model collection.

## Contents

- `summary.jsonl` — tidy per-task metrics ({n_rows} rows). One row per
  (model, family, task, metric):
  - `metric = hard_acc` (5 logprob families) — fraction of items where
    P(correct) > P(incorrect); higher ⇒ resists the misleading pattern.
  - `metric = correct_log_prob` (5 logprob families) — mean
    teacher-forced log probability of the correct answer.
  - `metric = prob_margin` (5 logprob families) — mean
    P(correct) - P(incorrect); a bounded soft preference score.
  - `metric = log_prob_margin` (5 logprob families) — mean
    log P(correct) - log P(incorrect); an unbounded soft margin.
  - `metric = log_normalized_correct_prob` (5 logprob families) — mean
    log P(correct) - logsumexp(log P(correct), log P(incorrect)); a
    two-choice normalized log probability.
  - `metric = normalized_correct_prob` (5 logprob families) — exp of the
    previous per item, averaged across items.
  - `metric = match_rate` (`multihop_persona_qa`) — mean regex-match rate
    across seeds, per (persona, question).
  - `score` is the value; `n` is the item / seed count behind it.
- `raw/<model>/<family>/<task>.json` — full per-item eval outputs.

## Models ({len(finished)})

{models_md}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS,
                    help=f"eval output tree (default: {DEFAULT_RESULTS})")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help=f"HF dataset repo id (default: {DEFAULT_REPO})")
    ap.add_argument("--public", action="store_true",
                    help="create a public repo (default: private)")
    ap.add_argument("--done-count", type=int, default=DONE_JSON_COUNT,
                    help="result JSONs that mark a model finished")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the summary, print, but do not upload")
    args = ap.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    if not os.path.isdir(results_dir):
        raise SystemExit(f"results dir not found: {results_dir}")

    finished, partial = [], []
    for name in sorted(os.listdir(results_dir)):
        d = os.path.join(results_dir, name)
        if not os.path.isdir(d):
            continue
        n = len(_json_files(d))
        (finished if n >= args.done_count else partial).append((name, n))

    print(f"results dir:          {results_dir}")
    print(f"finished models:      {len(finished)}")
    print(f"partial / unfinished: {len(partial)}")
    if not finished:
        raise SystemExit("nothing finished to push.")

    summary: list[dict] = []
    for name, _ in finished:
        summary += summarize_model(name, os.path.join(results_dir, name))
    print(f"summary rows:         {len(summary)}")

    staging = tempfile.mkdtemp(prefix="gdsuite_hf_")
    with open(os.path.join(staging, "summary.jsonl"), "w") as f:
        for r in summary:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(staging, "README.md"), "w") as f:
        f.write(_readme(finished, len(summary)))

    if args.dry_run:
        print(f"[dry-run] staged summary.jsonl + README.md at {staging}")
        print(f"[dry-run] would push raw results for {len(finished)} models "
              f"to {args.repo}")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset",
                    private=not args.public, exist_ok=True)
    # summary + card at the repo root
    api.upload_folder(folder_path=staging, repo_id=args.repo,
                      repo_type="dataset",
                      commit_message=f"Summary: {len(finished)} models, "
                                     f"{len(summary)} rows")
    # raw JSONs under raw/<model>/... — only the finished models
    api.upload_folder(folder_path=results_dir, path_in_repo="raw",
                      repo_id=args.repo, repo_type="dataset",
                      allow_patterns=[f"{m}/**" for m, _ in finished],
                      commit_message=f"Raw results: {len(finished)} models")
    print(f"pushed → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
