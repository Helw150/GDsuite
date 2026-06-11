# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-layer residual-stream profile of Delphi-1e23 (25B) loaded via HF.

Companion to the Levanter f32 profile so we can line up absmax / norm at
every residual-stream depth and find the layer where HF and Levanter
disagree. Same model, same fp32, same fallback text.

Reports per residual-stream tap:
  - absmax            : max |x| over all elements
  - norm_max          : max over token positions of ||x_t||_2
  - norm_mean         : mean over token positions of ||x_t||_2
  - norm_total        : ||x||_2 of the whole (seq, hidden) tensor
  - mean              : signed mean over all elements  (DC offset)
  - median            : signed median over all elements
  - sigma             : population std over all elements

Levanter's "norm" column is per-position; norm_max is the cleanest match.
The mean/median/sigma columns are the across-layer summary stats — sigma
tracks the typical magnitude while absmax tracks the worst-case outlier.

Output rows:
  idx 0          → embedding output
  idx 1..L       → residual stream after layer 0..L-1
  final_norm     → after the model's final pre-lm_head norm
  logits         → after lm_head (absmax only)
  CE             → mean per-token cross-entropy (sanity: Levanter got 0.341)
"""
import math

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "marin-community/delphi-1e23-25Bparams-628Btokens"
DTYPE = torch.float32

FALLBACK_TEXT = "The unanimous Declaration of the thirteen united States of America, When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation."


def _stats(x: torch.Tensor) -> dict:
    """absmax + per-token L2 norms + signed mean/median/sigma for a
    (1, seq, hidden) residual stream tap. Computes in fp32."""
    x = x.detach().to(torch.float32)
    flat = x.reshape(-1, x.shape[-1])           # (seq, hidden)
    per_tok = flat.norm(dim=-1)                  # (seq,)
    elems = flat.reshape(-1)
    return {
        "absmax": flat.abs().max().item(),
        "norm_max": per_tok.max().item(),
        "norm_mean": per_tok.mean().item(),
        "norm_total": flat.norm().item(),
        "mean": elems.mean().item(),
        "median": elems.median().item(),
        "sigma": elems.std(unbiased=False).item(),
    }


def _row(idx: str, s: dict) -> str:
    return (f"{idx:<12}  absmax={s['absmax']:>12.4f}   "
            f"norm_max={s['norm_max']:>12.4f}   "
            f"norm_mean={s['norm_mean']:>12.4f}   "
            f"norm_total={s['norm_total']:>12.4f}   "
            f"mean={s['mean']:>+9.4f}   "
            f"median={s['median']:>+9.4f}   "
            f"sigma={s['sigma']:>9.4f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    token_ids = tokenizer.encode(FALLBACK_TEXT, add_special_tokens=True, truncation=True)
    input_ids = torch.tensor([token_ids], device=device)
    print(f"model:  {MODEL}")
    print(f"dtype:  {DTYPE}")
    print(f"device: {device}")
    print(f"tokens: {input_ids.shape[1]} input, {input_ids.shape[1] - 1} scored\n")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=DTYPE, trust_remote_code=True,
    ).to(device)
    model.eval()

    # Capture input to lm_head — that IS the post-final-norm residual stream
    # without us having to find the norm submodule by name.
    pre_lmhead = {}
    def hook(_mod, inp, _out):
        pre_lmhead["x"] = inp[0]
    handle = model.lm_head.register_forward_hook(hook)

    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            output_hidden_states=True,
        )

    handle.remove()
    hidden_states = out.hidden_states     # tuple len = n_layers + 1
    logits = out.logits

    # Residual-stream taps: embedding output + after each layer.
    print(_row("0 (embed)", _stats(hidden_states[0])))
    for i in range(1, len(hidden_states)):
        print(_row(f"{i:<2} (L{i - 1})", _stats(hidden_states[i])))

    # Post-final-norm row.
    print(_row("final_norm", _stats(pre_lmhead["x"])))

    # Logits absmax.
    logits_absmax = logits.detach().float().abs().max().item()
    print(f"{'logits':<12}  absmax={logits_absmax:>12.4f}")

    # CE sanity check.
    loss = F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    ).item()
    print(f"\nCE: loss={loss:.6f}  ppl={math.exp(loss):.3f}  "
          f"(Levanter f32 baseline: 0.341)")


if __name__ == "__main__":
    main()
