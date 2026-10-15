#!/usr/bin/env python3
"""
fitmap.py -- the deployability map: for each edge platform class, which model size,
weight precision, KV precision and context length fit in memory, and the memory-bound
decode ceiling that follows (Eq. ceiling).  Writes figs/fitmap.pdf and fitmap_numbers.tex.

Model configurations are the public Hugging Face config.json values.  Platform capacities
and bandwidths are nominal class values (ASSUME where marked), stated in the figure caption.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})

MODELS = [  # name, params, L, Hkv, dh
    ("Qwen2.5-0.5B", 0.49e9, 24, 2, 64),
    ("Llama-3.2-1B", 1.24e9, 16, 8, 64),
    ("Llama-3.2-3B", 3.21e9, 28, 8, 128),
    ("Llama-3.1-8B", 8.03e9, 32, 8, 128),
    ("Qwen2.5-14B", 14.8e9, 48, 8, 128),
]
# platform class: name, usable memory for the model (bytes), bandwidth (B/s), colour
PLAT = [
    ("Embedded FPGA (KV260, 4 GB DDR4)",      3.5e9,   19.2e9, "#1F6E56"),   # ~0.5 GB for OS/bitstream
    ("Laptop GPU (RTX 3050, 4 GB GDDR6)",     3.6e9,   187e9,  "#0C447C"),   # measured beta
    ("Jetson Orin Nano (8 GB shared)",        6.0e9,   68e9,   "#3E7CB1"),   # ASSUME: 6 GB usable
    ("Flagship phone (12 GB LPDDR5X)",        6.0e9,   60e9,   "#993C1D"),   # ASSUME: ~6 GB for one app
]
OVERHEAD = 1.10   # ASSUME: 10 % runtime/activation overhead on top of weights + KV

def kv_per_tok(L, Hkv, dh, bits): return L * 2 * Hkv * dh * bits / 8
def max_context(cap, params, L, Hkv, dh, wbits, kvbits):
    free = cap / OVERHEAD - params * wbits / 8
    return max(0.0, free / kv_per_tok(L, Hkv, dh, kvbits))
def ceiling(bw, params, L, Hkv, dh, wbits, kvbits, n):
    return bw / (params * wbits / 8 + kv_per_tok(L, Hkv, dh, kvbits) * n)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
x = np.arange(len(MODELS)); w = 0.19
for i, (pn, cap, bw, col) in enumerate(PLAT):
    mc8 = [max_context(cap, *m[1:], 4, 8) for m in MODELS]
    mc2 = [max_context(cap, *m[1:], 4, 2) for m in MODELS]
    a1.bar(x + (i - 1.5) * w, [max(v, 1) for v in mc8], w, color=col, label=pn.split(" (")[0])
    a1.scatter(x + (i - 1.5) * w, [max(v, 1) for v in mc2], marker="_", color=col, s=60, lw=1.4, zorder=3)
a1.set_yscale("log"); a1.set_ylim(1e2, 1e7)
a1.axhspan(8e3, 32e3, color="#F0C2A8", alpha=0.45, lw=0)
a1.text(4.45, 15e3, "reasoning traces (8k-32k)", fontsize=6, color="#7A2E14", va="center", ha="right")
a1.axhline(128e3, color="0.5", lw=0.6, ls=":"); a1.text(4.45, 150e3, "128k window", fontsize=6, color="0.4", ha="right")
a1.set_xticks(x); a1.set_xticklabels([m[0] for m in MODELS], fontsize=6.5, rotation=15)
a1.set_ylabel("longest context that fits (tokens)")
a1.set_title("(a) capacity: W4 weights, INT8 KV (bar) / 2-bit KV (tick)", fontsize=8)
a1.legend(fontsize=5.6, frameon=False, loc="upper right", ncol=1)
# (b) ceiling at 8k context
for i, (pn, cap, bw, col) in enumerate(PLAT):
    vals = []
    for m in MODELS:
        fits = max_context(cap, *m[1:], 4, 8) >= 8192
        vals.append(ceiling(bw, *m[1:], 4, 8, 8192) if fits else np.nan)
    a2.plot(x, vals, "o-", color=col, ms=4, lw=1.1)
a2.set_yscale("log"); a2.set_ylim(0.2, 1e4)
a2.axhline(10, color="0.5", lw=0.6, ls="--"); a2.text(len(MODELS) - 1.05, 12, "~reading speed (10 tok/s)", fontsize=6, color="0.4", ha="right")
a2.set_xticks(x); a2.set_xticklabels([m[0] for m in MODELS], fontsize=6.5, rotation=15)
a2.set_ylabel("decode ceiling at 8k context (tok/s)")
a2.set_title("(b) memory-bound ceiling, W4 / INT8 KV, 8k context", fontsize=8)
a2.text(0.02, 0.04, "missing point = does not fit at 8k", transform=a2.transAxes, fontsize=6, color="0.4")
fig.tight_layout(w_pad=1.2)
fig.savefig("figs/fitmap.pdf", bbox_inches="tight"); fig.savefig("figs/fitmap.png", dpi=170, bbox_inches="tight")

out = {}
for pn, cap, bw, _ in PLAT:
    out[pn] = {m[0]: {"max_ctx_int8kv": int(max_context(cap, *m[1:], 4, 8)), "max_ctx_2bitkv": int(max_context(cap, *m[1:], 4, 2)),
                      "ceiling_8k": round(ceiling(bw, *m[1:], 4, 8, 8192), 1)} for m in MODELS}
json.dump(out, open("fitmap_results.json", "w"), indent=1)
L1 = next(m for m in MODELS if m[0] == "Llama-3.2-1B"); L8 = next(m for m in MODELS if m[0] == "Llama-3.1-8B"); L3 = next(m for m in MODELS if m[0] == "Llama-3.2-3B")
ph = PLAT[3]; fp = PLAT[0]; MW_SRAM = 16e6
open("fitmap_numbers.tex", "w").write(
    "\\newcommand{\\fitPhoneEightB}{%d}\n\\newcommand{\\fitPhoneEightBTwo}{%d}\n\\newcommand{\\fitPhoneThreeB}{%d}\n"
    "\\newcommand{\\fitFpgaEightB}{%d}\n\\newcommand{\\fitFpgaOneB}{%d}\n\\newcommand{\\fitMwOneB}{%d}\n"
    "\\newcommand{\\ceilPhoneEightB}{%.1f}\n\\newcommand{\\ceilFpgaOneB}{%.0f}\n\\newcommand{\\ceilPhoneThreeB}{%.0f}\n"
    % (max_context(ph[1], *L8[1:], 4, 8) / 1e3, max_context(ph[1], *L8[1:], 4, 2) / 1e3, max_context(ph[1], *L3[1:], 4, 8) / 1e3,
       max_context(fp[1], *L8[1:], 4, 8) / 1e3, max_context(fp[1], *L1[1:], 4, 8) / 1e3, MW_SRAM / OVERHEAD / (1.58 / 8) / 1e6,
       ceiling(ph[2], *L8[1:], 4, 8, 8192), ceiling(fp[2], *L1[1:], 4, 8, 8192), ceiling(ph[2], *L3[1:], 4, 8, 8192)))
print(json.dumps(out, indent=1))
