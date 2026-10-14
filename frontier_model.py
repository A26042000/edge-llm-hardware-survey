#!/usr/bin/env python3
"""
frontier_model.py -- applies the per-component model (Eq. model, validated in Sec. validation)
to the workload shifts of 2025-2026: reasoning-length contexts, KV compression and latent
attention, hybrid linear-attention / SSM models, speculative decoding, diffusion LLMs, and
on-device mixture-of-experts.  Writes figs/frontier.pdf and frontier_numbers.tex.

All workload parameters are public model configurations or stated assumptions (marked ASSUME);
platform parameters are the nominal rows of Table tab:modelparams.  Change and re-run.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from edge_token_model import MODELS, OPS, PLATFORMS, per_token

plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"Weights": "#0C447C", "KV cache": "#1F6E56", "Flash (experts)": "#7FA7C9",
     "Extra compute": "#6B5B95", "Non-GEMM": "#993C1D", "State": "#8FB8A8"}

M = MODELS["Llama-3.2-1B"]
NPU, CIM = PLATFORMS[1], PLATFORMS[3]
FLASH_BW = 4e9          # ASSUME: UFS 4.0-class sequential read
L, H, Hkv, dh = M["L"], M["H"], M["Hkv"], M["dh"]

def kv_per_pos(kind, bits=8):
    """bytes of cached state per context position for the 1B configuration"""
    if kind == "gqa":    return L * 2 * Hkv * dh * bits / 8
    if kind == "mla":    return L * (512 + 64) * bits / 8          # latent d_c=512 + decoupled RoPE key 64
    if kind == "hybrid": return (L // 4) * 2 * Hkv * dh * bits / 8   # 1 attention layer in 4
    if kind == "ssm":    return 0.0
    raise KeyError(kind)

def ng_ops(c, mult=1.0, sample_mult=1.0, softmax_frac=1.0):
    return mult * (c["softmax"] * softmax_frac + c["norm"] + c["rope"] + c["act"]) + sample_mult * c["sample"]

def token(p, n=2048, kvbits=8, kind="gqa", amort=1.0, flop_mult=1.0, ng_mult=1.0,
          sample_mult=1.0, extra_w=0.0, w_bytes=None, w_flop=None, flash=0.0, rerotate=False):
    """time components (s) of one generated token.
    amort     : generated tokens sharing one weight/KV fetch (speculative, diffusion)
    flop_mult : matrix operations per generated token relative to plain decode
    extra_w   : additional weight bytes per fetch (draft model)"""
    c = per_token(M, n, p["wbits"], kvbits)
    Bw = c["w_bytes"] if w_bytes is None else w_bytes
    Fw = c["gemv_flop"] if w_flop is None else w_flop
    if p["resident"]: Bw_mem = 0.0
    else: Bw_mem = (Bw + extra_w) / amort
    kvb = kv_per_pos(kind, kvbits) * n / amort
    att_frac = {"gqa": 1, "mla": 1, "hybrid": 0.25, "ssm": 0}[kind]
    Fa = c["att_flop"] * att_frac * (8.5 if kind == "mla" else 1)       # absorbed MLA: ~8.5x attention ops
    state = 0.0
    if kind in ("hybrid", "ssm"):                                         # delta-rule / SSM state: read + write
        nlin = L - L // 4 if kind == "hybrid" else L
        state = nlin * H * dh * dh * 2 * 2
    t = {}
    t_wmem, t_wcomp = Bw_mem / p["BW"], Fw * flop_mult / p["P"]
    t["Weights"] = t_wmem
    t["Extra compute"] = max(0.0, t_wcomp - t_wmem)
    t_kv, t_ac = kvb / p["BW"], Fa * flop_mult / p["P"]
    t["KV cache"] = t_kv
    t["Extra compute"] += max(0.0, t_ac - t_kv)
    t["State"] = state / p["BW"]
    t["Flash (experts)"] = flash / FLASH_BW
    sm_frac = att_frac
    t["Non-GEMM"] = ng_ops(c, ng_mult, sample_mult, sm_frac) / p["T_ng"]
    if rerotate:   # pre-RoPE quantized keys: every cached key is rotated again at every step
        t["Non-GEMM"] += L * Hkv * dh * OPS["rope"] * n / p["T_ng"]
    return t

# ---------------------------------------------------------------- scenarios
k, alpha = 4, 0.7                                   # ASSUME: 4 drafted tokens, 70% acceptance
E_acc = (1 - alpha ** (k + 1)) / (1 - alpha)        # expected tokens per verification pass
b, s = 32, 8                                         # ASSUME: diffusion block of 32 tokens, 8 denoising steps
# MoE: gpt-oss-20b-class (public: 20.9B total, 3.6B active; 24 layers, 32 experts, top-4, d=2880)
moe_active, moe_expert_active, moe_bits = 3.61e9, 4 * 24 * 3 * 2880 * 2880, 4.25
moe_bytes = moe_active * moe_bits / 8
hit = 0.90                                           # ASSUME: expert-cache hit rate in DRAM

def scen(p):
    w4 = per_token(M, 2048, p["wbits"], 8)["w_bytes"]
    return [
        ("Chat (2k context)", token(p, 2048)),
        ("Reasoning (32k)", token(p, 32768)),
        ("Reasoning, 2-bit KV", token(p, 32768, kvbits=2)),
        ("Reasoning, 2-bit pre-RoPE keys", token(p, 32768, kvbits=2, rerotate=True)),
        ("Reasoning, latent KV (MLA)", token(p, 32768, kind="mla")),
        ("Reasoning, hybrid 3:1", token(p, 32768, kind="hybrid")),
        ("Speculative (k=4, 2k)", token(p, 2048, amort=E_acc, flop_mult=(k + 1) / E_acc,
                                         ng_mult=(k + 1) / E_acc, sample_mult=(k + 1) / E_acc,
                                         extra_w=k * 0.1 * w4)),
        ("Diffusion (b=32, s=8, 2k)", token(p, 2048, amort=b / s, flop_mult=s, ng_mult=s, sample_mult=s)),
        ("MoE 21B/3.6B in DRAM", token(p, 2048, w_bytes=moe_bytes, w_flop=2 * moe_active)),
        ("MoE, experts in flash", token(p, 2048, w_bytes=moe_bytes, w_flop=2 * moe_active,
                                        flash=(1 - hit) * moe_expert_active * moe_bits / 8)),
    ]

npu = scen(NPU)
cim = scen(CIM)[:8]
ORDER = ["Weights", "KV cache", "State", "Flash (experts)", "Extra compute", "Non-GEMM"]

fig = plt.figure(figsize=(7.3, 3.1))
go = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.9], wspace=0.72)
gi = go[1].subgridspec(1, 2, width_ratios=[1.3, 0.7], wspace=0.08)
a1 = fig.add_subplot(go[0]); a2 = fig.add_subplot(gi[0]); a3 = fig.add_subplot(gi[1], sharey=a2)
# (a) bytes per token vs context
ns = np.logspace(np.log10(512), np.log10(131072), 120)
wb = per_token(M, 0, 4, 8)["w_bytes"]
a1.axhline(wb / 1e6, color=C["Weights"], lw=1.4)
a1.text(0.55, wb / 1e6 * 0.72, "W4 weights (618 MB)", color=C["Weights"], fontsize=6.3, va="top")
LINES = (("FP16 KV", "gqa", 16, "-"), ("INT8 KV", "gqa", 8, "--"), ("latent KV (MLA-style)", "mla", 8, "-."),
         ("2-bit KV, or hybrid 3:1", "gqa", 2, ":"))
for lab, kind, bits, ls in LINES:
    y = kv_per_pos(kind, bits) * ns / 1e6
    a1.loglog(ns / 1024, y, ls=ls, color=C["KV cache"], lw=1.1, label=lab)
    xc = wb / kv_per_pos(kind, bits) / 1024
    if xc < 128:
        a1.plot(xc, wb / 1e6, "o", ms=3, color=C["KV cache"])
a1.legend(fontsize=5.6, frameon=False, loc="lower right", title="KV cache per token", title_fontsize=5.8)
a1.axvspan(8, 32, color="#F0C2A8", alpha=0.4, lw=0)
a1.text(16, 2.5e3, "long\nreasoning\ntraces", fontsize=5.8, color="#7A2E14", ha="center")
a1.set_xlim(0.5, 128); a1.set_ylim(0.5, 1e4)
a1.set_xticks([0.5, 2, 8, 32, 128]); a1.set_xticklabels(["0.5", "2", "8", "32", "128"])
a1.set_xlabel("context (k tokens)"); a1.set_ylabel("MB read per token")
a1.set_title("(a) KV vs. weight bytes (1B model)", fontsize=8.2)
# (b) phone NPU per-token time
labs = [x[0] for x in npu]
for i, (lab, t) in enumerate(npu):
    left = 0
    for comp in ORDER:
        v = 1e3 * t[comp]
        a2.barh(i, v, left=left, color=C[comp], edgecolor="white", lw=0.3, label=comp if i == 0 else None)
        left += v
    a2.text(left + 0.8, i, f"{1 / sum(t.values()):.0f} tok/s", va="center", fontsize=5.9)
a2.set_yticks(range(len(labs))); a2.set_yticklabels(labs, fontsize=6.3); a2.invert_yaxis()
a2.set_xlabel("ms per token"); a2.set_xlim(0, 78)
a2.set_title("(b) phone NPU, 60 GB/s", fontsize=8.2)
a2.legend(fontsize=5.6, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.62, -0.17))
# (c) non-GEMM share on the weight-resident configuration
sh = [100 * t["Non-GEMM"] / sum(t.values()) for _, t in cim]
a3.barh(range(len(cim)), sh, color=C["Non-GEMM"], height=0.6)
for i, (v, (_, t)) in enumerate(zip(sh, cim)):
    a3.text(v + 2, i, f"{v:.0f}%, {1e3 * sum(t.values()):.2g} ms", va="center", fontsize=5.3)
for i in range(len(cim), len(labs)): a3.text(2, i, "n/a", va="center", fontsize=5.9, color="0.5")
plt.setp(a3.get_yticklabels(), visible=False); a3.tick_params(axis="y", length=0)
a3.set_xlim(0, 175); a3.set_xticks([0, 50, 100]); a3.set_xlabel("non-GEMM (%)")
a3.set_title("(c) resident CIM", fontsize=8.2)
fig.savefig("figs/frontier.pdf", bbox_inches="tight"); fig.savefig("figs/frontier.png", dpi=170, bbox_inches="tight")

out = {}
for (lab, t), (_, tc) in zip(npu, cim + [(None, None)] * 2):
    tot = sum(t.values())
    out[lab] = {"npu_ms": round(1e3 * tot, 2), "npu_tok_s": round(1 / tot, 1),
                "npu_share": {k_: round(100 * v / tot, 1) for k_, v in t.items()},
                "cim_ng_share": None if tc is None else round(100 * tc["Non-GEMM"] / sum(tc.values()), 1),
                "cim_ms": None if tc is None else round(1e3 * sum(tc.values()), 3)}
cross = {f"{k_}{b_}": round(wb / kv_per_pos(k_, b_) / 1024, 1) for k_, b_ in (("gqa", 16), ("gqa", 8), ("gqa", 2), ("mla", 8), ("hybrid", 8))}
out["crossover_k_tokens"] = cross
out["E_acc"] = round(E_acc, 2)
json.dump(out, open("frontier_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
