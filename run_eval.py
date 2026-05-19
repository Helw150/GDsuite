"""Unified entrypoint for the 6 main evaluation families from
"Generalization dynamics across fine-tuning"
(https://jiaxin-wen.github.io/blog/generalization-dynamics.html).

All 5 logprob families share ONE handler — they differ only in demo
sampling strategy and the joiner used to concatenate demo blocks. The
per-item correct/incorrect answers are precomputed in the dataset (see
huggingface.co/datasets/jiaxin-wen/generalization-dynamics-evals), so the
runner never computes patterns at runtime. Multi-hop persona QA is
generative and has its own handler.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Callable

import yaml


DEFAULT_HF_DATASET = "jiaxin-wen/generalization-dynamics-evals"


# ════════════════════════════════════════════════════════════════════
# vLLM helpers
# ════════════════════════════════════════════════════════════════════

def create_llm(model_name: str, revision: str | None, vllm_cfg: dict):
    """Build a vLLM LLM instance from the config.yaml `vllm:` block."""
    from vllm import LLM
    kwargs = dict(
        model=model_name, revision=revision,
        tensor_parallel_size=vllm_cfg.get("tensor_parallel_size", 1),
        gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.9),
        dtype=vllm_cfg.get("dtype", "bfloat16"),
        enforce_eager=vllm_cfg.get("enforce_eager", True),
        enable_prefix_caching=vllm_cfg.get("enable_prefix_caching", True),
    )
    for k in ("max_model_len", "max_num_seqs"):
        if vllm_cfg.get(k):
            kwargs[k] = vllm_cfg[k]
    return LLM(**kwargs)


def cleanup_llm(llm) -> None:
    import torch
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def score_logprobs(llm, tokenizer, prompts: list[str],
                   answers: list[str]) -> list[dict]:
    """Teacher-forced P(answer | prompt) for each (prompt, answer) pair.

    Multi-token answers: tokenize prompt + ' ' + answer, find the answer
    boundary via the tokenization of prompt + ' ', and read per-token
    logprobs at every position from the boundary onward. Returns dicts
    with `avg_prob` (mean per-token probability) and `log_prob` (sum of
    per-token log probabilities) per pair.
    """
    from vllm import SamplingParams

    full_texts, answer_starts = [], []
    for prompt, answer in zip(prompts, answers):
        full_text = prompt + " " + answer
        prompt_ids = tokenizer.encode(prompt + " ", add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        ans_start = len(prompt_ids)
        if full_ids[:ans_start] != prompt_ids:
            # Fallback: trailing space changed prompt tokenization.
            ans_start = len(tokenizer.encode(prompt, add_special_tokens=False))
        full_texts.append(full_text)
        answer_starts.append(ans_start)

    outputs = llm.generate(
        full_texts,
        SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0),
        use_tqdm=True)

    results: list[dict] = []
    for out, ans_start in zip(outputs, answer_starts):
        logp_seq, token_ids = out.prompt_logprobs, out.prompt_token_ids
        if logp_seq is None:
            results.append({"avg_prob": 0.0, "log_prob": float("-inf")})
            continue
        probs, logs = [], []
        for pos in range(ans_start, len(logp_seq)):
            cell = logp_seq[pos]
            if cell is None or pos >= len(token_ids):
                continue
            tid = token_ids[pos]
            if tid in cell:
                lp = cell[tid].logprob
                probs.append(math.exp(lp))
                logs.append(lp)
        results.append({
            "avg_prob": sum(probs) / len(probs) if probs else 0.0,
            "log_prob": sum(logs) if logs else float("-inf"),
        })
    return results


# ════════════════════════════════════════════════════════════════════
# Demo sampling + ICL prompt assembly
# ════════════════════════════════════════════════════════════════════

def _balanced(demos: list[dict], k: int, rng: random.Random) -> list[dict]:
    """K/2 per class via the `label` field (used by Flipped Answer)."""
    c0 = [d for d in demos if d["label"] == 0]
    c1 = [d for d in demos if d["label"] == 1]
    rng.shuffle(c0); rng.shuffle(c1)
    out = c0[:k // 2] + c1[:k // 2]
    rng.shuffle(out)
    return out


DEMO_STRATEGIES: dict[str, Callable[[list[dict], int, random.Random], list[dict]]] = {
    "none":     lambda demos, k, rng: [],
    "random":   lambda demos, k, rng: rng.sample(demos, min(k, len(demos))),
    "ordered":  lambda demos, k, rng: demos[:k],
    "balanced": _balanced,
}


def _select_demo_set(demos: list[dict], seed: int) -> list[dict]:
    """For multi-demo-set tasks (successive), `seed` picks one demo_set."""
    if not demos or "demo_set" not in demos[0]:
        return demos
    n_sets = max(d.get("demo_set", 0) for d in demos) + 1
    return [d for d in demos if d.get("demo_set", 0) == seed % n_sets]


def build_icl_prompt(blocks: list[str], test_block: str,
                     joiner: str = "\n\n") -> str:
    """Concatenate pre-formatted demo blocks + a test block.
    Each demo block is `{prompt} {answer}`; the test block is the test
    item's `prompt` (which ends with an open answer marker)."""
    return joiner.join(blocks + [test_block])


# ════════════════════════════════════════════════════════════════════
# Run context + IO helpers
# ════════════════════════════════════════════════════════════════════

@dataclass
class RunContext:
    """Cross-family runtime parameters. Each handler receives one."""
    family: str
    out_dir: str
    n_seeds: int | None = None
    max_eval: int | None = None
    skip_existing: bool = True

    @property
    def family_dir(self) -> str:
        d = os.path.join(self.out_dir, self.family)
        os.makedirs(d, exist_ok=True)
        return d


def _seeds(cfg: dict, ctx: RunContext) -> int:
    return ctx.n_seeds if ctx.n_seeds is not None else cfg.get("n_seeds", 1)


def _skip(out_path: str, ctx: RunContext) -> bool:
    if ctx.skip_existing and os.path.exists(out_path):
        print(f"  [skip] {out_path}")
        return True
    return False


def _write(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _read_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path)]


def _hard_acc(rows: list[dict]) -> float:
    """Fraction of rows where P(correct) > P(incorrect). Higher ⇒ the
    model resists the misleading pattern."""
    if not rows:
        return 0.0
    return sum(1 for r in rows
               if r["correct_avg_prob"] > r["incorrect_avg_prob"]) / len(rows)


def _load_chatml_demos(path: str) -> list[dict]:
    """ChatML messages JSONL → [{prompt: 'Q: <user>\\nA:', answer: '<ai>'}, …].
    Lets persona reuse the unified `{prompt, answer}` demo schema and
    `build_icl_prompt`."""
    out = []
    for r in _read_jsonl(path):
        msgs = r["messages"]
        u = next(m["content"] for m in msgs if m["role"] == "user")
        a = next(m["content"] for m in msgs if m["role"] == "assistant")
        out.append({"prompt": f"Q: {u}\nA:", "answer": a})
    return out


# ════════════════════════════════════════════════════════════════════
# Generic logprob family runner — covers families 1-5
# ════════════════════════════════════════════════════════════════════

def run_logprob_family(llm, tokenizer, cfg: dict, ctx: RunContext) -> None:
    """Iterate `cfg["tasks"]`, scoring each task's items. Task `name` may
    contain `/` (nested subdir) — both the source JSONL and the output
    JSON path follow that subdir. A task with no demo rows in its file
    (e.g. repetitive_answer.algebra/*) is automatically zero-shot."""
    strategy = cfg.get("demo_strategy", "random")
    joiner = cfg.get("joiner", "\n")
    sample = DEMO_STRATEGIES[strategy]
    n_seeds_cfg = _seeds(cfg, ctx)

    for task in cfg["tasks"]:
        name = task["name"]
        k = task.get("k", 0)
        out_path = os.path.join(ctx.family_dir, f"{name}.json")
        if _skip(out_path, ctx):
            continue

        rows = _read_jsonl(os.path.join(cfg["data_dir"], f"{name}.jsonl"))
        demos_all = [r for r in rows if r["split"] == "demo"]
        tests     = [r for r in rows if r["split"] == "test"]
        if ctx.max_eval:
            tests = tests[:ctx.max_eval]
        # Zero-shot tasks (no demos) → single seed.
        n_seeds = 1 if strategy == "none" or not demos_all else n_seeds_cfg
        # `random` strategy resamples demos PER test item (rng advances
        # inside the inner loop, matching the legacy PvC protocol). The
        # other strategies produce demos that don't depend on which item
        # is being scored — build them once per seed.
        resample_per_item = strategy == "random"

        prompts: list[str] = []
        metas:   list[dict] = []
        for seed in range(n_seeds):
            rng = random.Random(seed)
            pool = _select_demo_set(demos_all, seed)
            if not resample_per_item:
                blocks = [f"{d['prompt']} {d['answer']}"
                          for d in sample(pool, k, rng)]
            for t in tests:
                if resample_per_item:
                    blocks = [f"{d['prompt']} {d['answer']}"
                              for d in sample(pool, k, rng)]
                prompts.append(build_icl_prompt(blocks, t["prompt"], joiner))
                # Pass every test field through (correct_answer,
                # incorrect_answer, plus any meta like `template` /
                # `round`) — only `split` and `prompt` are structural.
                metas.append({**{k_: v for k_, v in t.items()
                                 if k_ not in ("split", "prompt")},
                              "seed": seed})
        if not prompts:
            continue

        print(f"  {name}: {n_seeds} seeds × {len(tests)} tests "
              f"= {len(prompts)} jobs (K={k})")
        # Two scoring passes share the same prompts → vLLM prefix cache
        # makes the second pass ~free.
        c_scores = score_logprobs(llm, tokenizer, prompts,
                                  [m["correct_answer"]   for m in metas])
        i_scores = score_logprobs(llm, tokenizer, prompts,
                                  [m["incorrect_answer"] for m in metas])

        results = [{**m,
                    "correct_avg_prob":   c["avg_prob"],
                    "correct_log_prob":   c["log_prob"],
                    "incorrect_avg_prob": i["avg_prob"],
                    "incorrect_log_prob": i["log_prob"]}
                   for m, c, i in zip(metas, c_scores, i_scores)]
        _write(out_path, results)
        print(f"    hard-acc (P(correct) > P(incorrect)): "
              f"{_hard_acc(results):.3f}")


# ════════════════════════════════════════════════════════════════════
# Family 6 — Multi-hop Persona QA (generative + regex match)
# ════════════════════════════════════════════════════════════════════

def run_multihop_persona_qa(llm, tokenizer, cfg: dict,
                            ctx: RunContext) -> None:
    from vllm import SamplingParams
    n_seeds = _seeds(cfg, ctx)
    n_samples = cfg["n_samples_per_question"]
    sp = SamplingParams(
        temperature=cfg.get("temperature", 1.0),
        max_tokens=cfg.get("max_new_tokens", 300),
        stop=cfg.get("stop", ["\nQ:", "\n\nQ:"]))

    for persona in cfg["personas"]:
        sum_path  = os.path.join(ctx.family_dir, f"{persona}_summary.json")
        full_path = os.path.join(ctx.family_dir, f"{persona}_full.json")
        if _skip(sum_path, ctx):
            continue

        facts = _load_chatml_demos(
            os.path.join(cfg["data_dir"], persona, "facts.jsonl"))
        test_qs = json.load(open(
            os.path.join(cfg["data_dir"], persona, "test_questions.json")))

        # Build (seed × question × sample) prompts. Demos are reshuffled
        # per seed (full pool fed as ICL) and shared across all items.
        prompts: list[str] = []
        metas:   list[dict] = []
        for seed in range(n_seeds):
            rng = random.Random(seed)
            demos = list(facts); rng.shuffle(demos)
            blocks = [f"{d['prompt']} {d['answer']}" for d in demos]
            for tq in test_qs:
                p = build_icl_prompt(blocks, f"Q: {tq['question']}\nA:")
                for s in range(n_samples):
                    prompts.append(p)
                    metas.append({"seed": seed, "test_id": tq["id"],
                                  "category": tq.get("category", "fact"),
                                  "expected_regex": tq["expected_answer_regex"],
                                  "sample_idx": s})

        print(f"  {persona}: {n_seeds} seeds × {len(test_qs)} Qs × "
              f"{n_samples} samples = {len(prompts)} generations")
        responses = [o.outputs[0].text.strip()
                     for o in llm.generate(prompts, sp)]
        full = [{**m, "response": r[:400],
                 "match": bool(re.search(m["expected_regex"], r,
                                         re.IGNORECASE))}
                for m, r in zip(metas, responses)]

        # summary: {seed: {question_id: match_rate}}
        summary: dict[str, dict[str, float]] = {}
        for seed in range(n_seeds):
            seed_rows = [x for x in full if x["seed"] == seed]
            summary[str(seed)] = {
                tq["id"]: sum(1 for r in seed_rows
                              if r["test_id"] == tq["id"] and r["match"])
                          / max(1, sum(1 for r in seed_rows
                                       if r["test_id"] == tq["id"]))
                for tq in test_qs}
        _write(full_path, full)
        _write(sum_path, summary)

        # Print mean-across-seeds match rate per question.
        for tq in test_qs:
            mean = (sum(summary[str(s)][tq["id"]] for s in range(n_seeds))
                    / max(1, n_seeds))
            print(f"    {tq['id']:<20} {mean:.0%}")


# ════════════════════════════════════════════════════════════════════
# Dispatch — only TWO handler functions for the entire suite
# ════════════════════════════════════════════════════════════════════

FAMILY_HANDLERS: dict[str, Callable[[Any, Any, dict, RunContext], None]] = {
    "flipped_answer":      run_logprob_family,
    "repetitive_answer":   run_logprob_family,
    "successive_answer":   run_logprob_family,
    "truthy_answer":       run_logprob_family,
    "intuitive_answer":    run_logprob_family,
    "multihop_persona_qa": run_multihop_persona_qa,
}
ALL_FAMILIES = list(FAMILY_HANDLERS)


# ════════════════════════════════════════════════════════════════════
# Data root + config resolution
# ════════════════════════════════════════════════════════════════════

def resolve_data_root(hf_dataset: str | None, local_data_dir: str | None,
                      release_dir: str) -> str:
    """Where to look for `<family>/...` data files. Precedence:
      1. --local_data_dir         (explicit user override)
      2. release/data/            (in-tree dev copy, if it exists)
      3. snapshot_download(--hf_dataset)
    """
    if local_data_dir:
        return os.path.abspath(local_data_dir)
    if hf_dataset:
        from huggingface_hub import snapshot_download
        root = snapshot_download(repo_id=hf_dataset, repo_type="dataset")
        print(f"  using HF dataset snapshot: {root}")
        return root
    return release_dir


def _absolutize_paths(cfg: dict, data_root: str) -> None:
    """Recursively rewrite `data_dir` / `data_path` to absolute paths
    under `data_root`. Used after the data root is resolved."""
    if not isinstance(cfg, dict):
        return
    for key, val in cfg.items():
        if key in ("data_dir", "data_path") and isinstance(val, str) \
                and not os.path.isabs(val):
            cfg[key] = os.path.join(data_root, val)
        elif isinstance(val, dict):
            _absolutize_paths(val, data_root)


def _resolve_config(here: str, args: argparse.Namespace) -> dict:
    """Load config.yaml, apply CLI overrides, resolve data paths."""
    cfg_path = args.config or os.path.join(here, "config.yaml")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    for key in ("gpu_memory_utilization", "tensor_parallel_size",
                "max_model_len", "max_num_seqs"):
        v = getattr(args, key)
        if v is not None:
            config["vllm"][key] = v

    hf_arg = args.hf_dataset or None
    local = args.local_data_dir
    if not local and os.path.isdir(os.path.join(here, "data")):
        local = os.path.join(here, "data")
        hf_arg = None
        print(f"  using local data tree: {local}")
    data_root = resolve_data_root(hf_arg, local, here)

    for fam_cfg in config.values():
        _absolutize_paths(fam_cfg, data_root)
    return config


# ════════════════════════════════════════════════════════════════════
# CLI + main
# ════════════════════════════════════════════════════════════════════

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", required=True,
                   help="HF hub id, e.g. allenai/Olmo-3-1025-7B")
    p.add_argument("--revision", default=None,
                   help="HF revision (branch/tag/commit). Defaults to main.")
    p.add_argument("--config", default=None,
                   help="Path to config.yaml (default: ./config.yaml)")
    p.add_argument("--output_dir", default=None,
                   help="Where to write outputs. "
                        "Default: outputs/<model>_<revision>/")
    p.add_argument("--families", nargs="+", default=None,
                   choices=ALL_FAMILIES,
                   help="Subset of families to run (default: all 6).")
    p.add_argument("--n_seeds", type=int, default=None,
                   help="Override n_seeds for every family (smoke test).")
    p.add_argument("--max_eval", type=int, default=None,
                   help="Cap eval items per task (smoke test).")
    p.add_argument("--force", action="store_true",
                   help="Re-run tasks even if their output JSON exists.")
    p.add_argument("--gpu_memory_utilization", type=float, default=None,
                   help="Override vllm.gpu_memory_utilization from config.")
    p.add_argument("--tensor_parallel_size", type=int, default=None,
                   help="Override vllm.tensor_parallel_size from config.")
    p.add_argument("--max_model_len", type=int, default=None,
                   help="Override vllm.max_model_len from config.")
    p.add_argument("--max_num_seqs", type=int, default=None,
                   help="Override vllm.max_num_seqs (cap concurrent reqs).")
    p.add_argument("--hf_dataset", default=DEFAULT_HF_DATASET,
                   help=f"HF dataset repo for the prepared eval data "
                        f"(default: {DEFAULT_HF_DATASET}). "
                        "Pass an empty string to disable.")
    p.add_argument("--local_data_dir", default=None,
                   help="Local clone of the eval-data tree (skips HF "
                        "download). Must contain the same <family>/... "
                        "subdirs as the HF dataset.")
    return p.parse_args(argv)


def _run_name(model_name: str, revision: str | None) -> str:
    short = model_name.split("/")[-1]
    return f"{short}_{revision}" if revision else short


def main(argv=None) -> None:
    args = _parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    config = _resolve_config(here, args)

    families = args.families or ALL_FAMILIES
    out_dir = args.output_dir or os.path.join(
        here, "outputs", _run_name(args.model_name, args.revision))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Model:    {args.model_name} (revision={args.revision})")
    print(f"Output:   {out_dir}")
    print(f"Families: {families}")

    llm = create_llm(args.model_name, args.revision, config["vllm"])
    tokenizer = llm.get_tokenizer()
    try:
        for family in families:
            print(f"\n=== {family} ===")
            ctx = RunContext(
                family=family, out_dir=out_dir,
                n_seeds=args.n_seeds, max_eval=args.max_eval,
                skip_existing=not args.force)
            FAMILY_HANDLERS[family](llm, tokenizer, config[family], ctx)
    finally:
        cleanup_llm(llm)
    print("\nDone.")


if __name__ == "__main__":
    main()
