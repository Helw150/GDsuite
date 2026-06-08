# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Layer-0 sub-op residual profile of Delphi-1e23 (25B), HF fp32.

Mirror of the planned Levanter v4-32 dump: captures every intermediate
tensor inside layer 0's forward so we can diff op-by-op and find the
first sub-op whose absmax/norm diverges. For each capture we report
absmax, per-token L2 norms (max / mean / total), and the (head, channel)
index of the absmax — so a misplaced-channel bug shows up as a channel
mismatch even when magnitudes look close.

Captured rows (in this order):
  layer_0_input           residual coming in (= embed for layer 0)
  input_layernorm         RMSNorm before attention
  q_proj, k_proj, v_proj  post-projection, pre-norm  (3D, B,T,F)
  q_norm, k_norm          Qwen3-style QK-norm        (4D, B,H,T,D)
  post_rope_q, post_rope_k  after apply_rotary_pos_emb
  o_proj_in               attention output pre-o_proj (weighted sum of V)
  o_proj_out              o_proj output (attention block out)
  self_attn_out           self_attn module return (sanity ~ o_proj_out)
  residual_after_attn     layer_0_input + self_attn_out
  post_attn_layernorm     RMSNorm before MLP
  mlp                     mlp output
  residual_after_mlp      residual_after_attn + mlp           (= layer_0_output)
  layer_0_output          layer module return                 (= idx 1 of full profile)

The post-RoPE rows depend on the trust_remote_code module exposing
apply_rotary_pos_emb at module scope — we probe a few common names; if
none match, those rows print [missing] and you can fall back to
comparing q_norm/k_norm (RoPE is parameter-free, so if pre-RoPE matches
and RoPE conventions match, post-RoPE matches too).
"""
import math
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "marin-community/delphi-1e23-25Bparams-628Btokens"
DTYPE = torch.float32

FALLBACK_TEXT = "The unanimous Declaration of the thirteen united States of America, When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation."

captures: dict = {}


def _stats(t: torch.Tensor) -> dict:
    """absmax + per-token L2 norms + (head, channel) of the absmax.

    Handles (B, T, F) and (B, H, T, D) — for the latter we permute heads
    next to the feature dim so per-token norms combine heads, matching the
    natural residual-stream view.
    """
    x = t.detach().to(torch.float32)
    orig_shape = tuple(x.shape)
    if x.dim() == 4:
        H = x.shape[1]
        x = x.permute(0, 2, 1, 3).contiguous().reshape(x.shape[0], x.shape[2], -1)
    elif x.dim() == 2:
        x = x.unsqueeze(0)  # treat as (1, T, F)
        H = 1
    else:
        H = 1
    flat = x.reshape(-1, x.shape[-1])
    per_tok = flat.norm(dim=-1)
    chan_absmax = flat.abs().max(dim=0).values  # (F,)
    abs_idx = int(chan_absmax.argmax().item())
    chan_per_head = x.shape[-1] // H
    return {
        "shape": orig_shape,
        "absmax": flat.abs().max().item(),
        "norm_max": per_tok.max().item(),
        "norm_mean": per_tok.mean().item(),
        "norm_total": flat.norm().item(),
        "head": (abs_idx // chan_per_head) if H > 1 else None,
        "channel": (abs_idx % chan_per_head) if H > 1 else abs_idx,
    }


def _row(name: str, s: dict) -> str:
    head_str = f"  h={s['head']:>2}" if s["head"] is not None else ""
    return (f"{name:<24} absmax={s['absmax']:>11.4f}  "
            f"nrm_max={s['norm_max']:>11.4f}  "
            f"nrm_mean={s['norm_mean']:>10.4f}  "
            f"nrm_tot={s['norm_total']:>11.4f}  "
            f"ch={s['channel']:>4}{head_str}  shape={s['shape']}")


def install_rope_capture(layer0) -> str | None:
    """Monkey-patch the rotary-embedding function at layer-0's attention
    module scope so the first call's outputs land in `captures`. Tries the
    common names; returns the one it patched, or None."""
    mod_name = type(layer0.self_attn).__module__
    mod = sys.modules.get(mod_name)
    if mod is None:
        return None
    for fname in ("apply_rotary_pos_emb", "apply_rotary_emb", "apply_rope"):
        if hasattr(mod, fname):
            orig = getattr(mod, fname)

            def wrapped(*args, **kwargs):
                out = orig(*args, **kwargs)
                if (isinstance(out, tuple) and len(out) == 2
                        and "post_rope_q" not in captures):
                    captures["post_rope_q"] = out[0].detach()
                    captures["post_rope_k"] = out[1].detach()
                return out
            setattr(mod, fname, wrapped)
            return fname
    return None


def main() -> None:
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

    cfg = model.config
    print(f"config: hidden_size={getattr(cfg, 'hidden_size', '?')}, "
          f"num_layers={getattr(cfg, 'num_hidden_layers', '?')}, "
          f"num_heads={getattr(cfg, 'num_attention_heads', '?')}, "
          f"num_kv_heads={getattr(cfg, 'num_key_value_heads', '?')}, "
          f"head_dim={getattr(cfg, 'head_dim', '?')}")
    layer0 = model.model.layers[0]
    print(f"layer-0 type:   {type(layer0).__name__}")
    print(f"self_attn type: {type(layer0.self_attn).__name__}")
    print(f"self_attn subs: {sorted(n for n, _ in layer0.self_attn.named_children())}\n")

    def out_hook(name):
        def h(_m, _i, o):
            captures[name] = (o[0] if isinstance(o, tuple) else o).detach()
        return h

    def in_hook(name):
        def h(_m, i, _o):
            captures[name] = i[0].detach()
        return h

    handles = []
    handles.append(layer0.register_forward_pre_hook(
        lambda _m, args: captures.__setitem__("layer_0_input", args[0].detach())))
    handles.append(layer0.input_layernorm.register_forward_hook(out_hook("input_layernorm")))
    handles.append(layer0.self_attn.q_proj.register_forward_hook(out_hook("q_proj")))
    handles.append(layer0.self_attn.k_proj.register_forward_hook(out_hook("k_proj")))
    handles.append(layer0.self_attn.v_proj.register_forward_hook(out_hook("v_proj")))
    if hasattr(layer0.self_attn, "q_norm"):
        handles.append(layer0.self_attn.q_norm.register_forward_hook(out_hook("q_norm")))
    if hasattr(layer0.self_attn, "k_norm"):
        handles.append(layer0.self_attn.k_norm.register_forward_hook(out_hook("k_norm")))
    handles.append(layer0.self_attn.o_proj.register_forward_hook(in_hook("o_proj_in")))
    handles.append(layer0.self_attn.o_proj.register_forward_hook(out_hook("o_proj_out")))
    handles.append(layer0.self_attn.register_forward_hook(out_hook("self_attn_out")))
    handles.append(layer0.post_attention_layernorm.register_forward_hook(out_hook("post_attn_layernorm")))
    handles.append(layer0.mlp.register_forward_hook(out_hook("mlp")))
    handles.append(layer0.register_forward_hook(out_hook("layer_0_output")))

    rope_name = install_rope_capture(layer0)
    print(f"RoPE capture: {rope_name or 'NOT FOUND — post_rope_{q,k} unavailable'}\n")

    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
        )
    for h in handles:
        h.remove()

    # Derived rows the user's plan asks for but aren't a single module call.
    if "layer_0_input" in captures and "self_attn_out" in captures:
        captures["residual_after_attn"] = captures["layer_0_input"] + captures["self_attn_out"]
    if "residual_after_attn" in captures and "mlp" in captures:
        captures["residual_after_mlp"] = captures["residual_after_attn"] + captures["mlp"]

    order = [
        "layer_0_input",
        "input_layernorm",
        "q_proj", "k_proj", "v_proj",
        "q_norm", "k_norm",
        "post_rope_q", "post_rope_k",
        "o_proj_in", "o_proj_out", "self_attn_out",
        "residual_after_attn",
        "post_attn_layernorm",
        "mlp",
        "residual_after_mlp",
        "layer_0_output",
    ]
    for name in order:
        if name not in captures:
            print(f"{name:<24} [missing]")
        else:
            print(_row(name, _stats(captures[name])))

    loss = F.cross_entropy(
        out.logits[:, :-1, :].float().reshape(-1, out.logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    ).item()
    print(f"\nCE: loss={loss:.6f}  ppl={math.exp(loss):.3f}  "
          f"(Levanter f32 baseline: 0.341)")


if __name__ == "__main__":
    main()
