# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Delphi 1e22 vs 1e23 loss regression — vLLM version of the bf16 repro.

Parity with reproduce_delphi_25b_regression_fp32.py, but uses vLLM instead
of HF transformers to score the same fixed text. The point is to isolate
the bf16 regression: if vLLM bf16 shows the same 25B-worse-than-9.7B
result, the issue is precision, not an HF runtime bug.

Memory: vLLM bf16 25B fits comfortably on an 80 GB card; fp32 25B needs
141 GB. The DTYPES dict below is bf16-only by default — flip the fp32 line
on if you also want to verify recovery under vLLM fp32.
"""
import gc
import math

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

BASELINE_MODEL = "marin-community/delphi-1e22-9.7Bparams-160Btokens"
REGRESSED_MODEL = "marin-community/delphi-1e23-25Bparams-628Btokens"

FALLBACK_TEXT = "The unanimous Declaration of the thirteen united States of America, When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation."

DTYPES = {
    "bf16": "bfloat16",
    # "fp32": "float32",   # uncomment to also test vLLM fp32 (needs 141 GB card)
}


def score(model_name, dtype, prompt_token_ids):
    """Mean per-token cross-entropy of `prompt_token_ids` under `model_name`
    loaded by vLLM at `dtype`. Mirrors what F.cross_entropy computes in the
    HF script (averaged over the N-1 scored positions)."""
    llm = LLM(
        model=model_name,
        dtype=dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        enable_prefix_caching=False,
        trust_remote_code=True,
        max_model_len=max(2048, len(prompt_token_ids) + 8),
    )
    out = llm.generate(
        prompts=None,
        sampling_params=SamplingParams(max_tokens=1, prompt_logprobs=1,
                                       temperature=0.0),
        prompt_token_ids=[prompt_token_ids],
        use_tqdm=False,
    )[0]

    # prompt_logprobs[i] is the distribution at position i predicting token
    # at position i. Position 0 has no prediction (its cell is None), so the
    # scored positions are 1..N-1 — the same N-1 used by the HF script.
    logp_seq, token_ids = out.prompt_logprobs, out.prompt_token_ids
    logs = []
    for pos in range(1, len(logp_seq)):
        cell = logp_seq[pos]
        if cell is None:
            continue
        tid = token_ids[pos]
        if tid in cell:
            logs.append(cell[tid].logprob)
    loss = -sum(logs) / len(logs) if logs else float("inf")

    # Tear vLLM all the way down — it grabs the whole GPU otherwise and the
    # next model fails to allocate.
    destroy_model_parallel()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return loss


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL, trust_remote_code=True)
    token_ids = tokenizer.encode(FALLBACK_TEXT, add_special_tokens=True, truncation=True)
    print(f"tokens: {len(token_ids)} input, {len(token_ids) - 1} scored\n")

    results = {}
    for model_name in (BASELINE_MODEL, REGRESSED_MODEL):
        for tag, dtype in DTYPES.items():
            loss = score(model_name, dtype, token_ids)
            results[(model_name, tag)] = loss
            print(f"{model_name}  [{tag}]: loss={loss:.6f}, ppl={math.exp(loss):.3f}")
        print()

    # Summary table + the headline delta.
    header = f"{'model':<55}" + "".join(f"{tag:>18}" for tag in DTYPES)
    print(header)
    print("-" * len(header))
    for model_name in (BASELINE_MODEL, REGRESSED_MODEL):
        row = f"{model_name:<55}"
        for tag in DTYPES:
            v = results[(model_name, tag)]
            row += f"  {v:>8.4f}/ppl{math.exp(v):<5.2f}"
        print(row)

    base_bf16 = results[(BASELINE_MODEL, "bf16")]
    reg_bf16 = results[(REGRESSED_MODEL, "bf16")]
    print(f"\n[vLLM bf16]  25B vs 9.7B: {reg_bf16:.4f} vs {base_bf16:.4f} "
          f"({'25B better' if reg_bf16 < base_bf16 else '25B still worse'})")


if __name__ == "__main__":
    main()
