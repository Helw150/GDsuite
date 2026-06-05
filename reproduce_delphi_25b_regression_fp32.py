# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Delphi 1e22 vs 1e23 loss regression — fp32 vs bf16 comparison.

The original repro loaded weights in bf16 and found the 1e23 (25B) model scoring
~2x worse loss than the 1e22 (9.7B) on the same text — backwards for scaling.

Training used jmp "p=f32,c=bfloat16": f32 *master weights*, bf16 compute. The
Hub checkpoints are f32, but `dtype=torch.bfloat16` rounds the weights to bf16 at
load time, and the deeper 25B (51 layers vs 37) is more sensitive to that. This
script scores each model in both bf16 and fp32 so we can see whether fp32 (which
matches the training weights) recovers the 25B.

Memory/disk: fp32 25B is ~100 GB of weights (and ~100 GB to download). Needs a
141 GB card (H200) and enough disk. For smaller/multi-GPU setups, uncomment the
device_map="auto" line to shard/offload. Models are loaded one at a time.
"""
import gc
import math

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

BASELINE_MODEL = "marin-community/delphi-1e22-9.7Bparams-160Btokens"
REGRESSED_MODEL = "marin-community/delphi-1e23-25Bparams-628Btokens"

FALLBACK_TEXT = "The unanimous Declaration of the thirteen united States of America, When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation."

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def score(model_name, dtype, input_ids, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=True,
        # device_map="auto",   # uncomment to shard/offload if it won't fit one GPU
    ).to(device)
    model.eval()

    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
        ).logits

    loss = F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    ).item()

    model.to("cpu")
    del model, logits
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL, trust_remote_code=True)
    token_ids = tokenizer.encode(FALLBACK_TEXT, add_special_tokens=True, truncation=True)
    input_ids = torch.tensor([token_ids], device=device)

    print(f"device: {device}")
    print(f"tokens: {input_ids.shape[1]} input, {input_ids.shape[1] - 1} scored\n")

    results = {}
    for model_name in (BASELINE_MODEL, REGRESSED_MODEL):
        for tag, dtype in DTYPES.items():
            loss = score(model_name, dtype, input_ids, device)
            results[(model_name, tag)] = loss
            print(f"{model_name}  [{tag}]: loss={loss:.6f}, ppl={math.exp(loss):.3f}")
        print()

    # Summary table + the headline delta we care about.
    print(f"{'model':<55} {'bf16':>18} {'fp32':>18}")
    print("-" * 93)
    for model_name in (BASELINE_MODEL, REGRESSED_MODEL):
        b = results[(model_name, "bf16")]
        f = results[(model_name, "fp32")]
        print(f"{model_name:<55} {b:>10.4f}/ppl{math.exp(b):<5.2f} {f:>10.4f}/ppl{math.exp(f):<5.2f}")

    reg_bf16 = results[(REGRESSED_MODEL, "bf16")]
    reg_fp32 = results[(REGRESSED_MODEL, "fp32")]
    base_fp32 = results[(BASELINE_MODEL, "fp32")]
    print(
        f"\n25B: bf16 loss {reg_bf16:.4f} -> fp32 loss {reg_fp32:.4f} "
        f"(delta {reg_bf16 - reg_fp32:+.4f})"
    )
    print(f"25B fp32 vs 9.7B fp32: {reg_fp32:.4f} vs {base_fp32:.4f} "
          f"({'25B better' if reg_fp32 < base_fp32 else '25B still worse'})")


if __name__ == "__main__":
    main()
