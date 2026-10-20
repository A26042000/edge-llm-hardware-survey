#!/usr/bin/env python3
"""Graphical abstract (Elsevier: >= 1328 x 531 px, landscape).  Left: the bottleneck shift
(model) with the measured check; right: the four workload findings and the agenda."""
import importlib.util, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from census_data import CENSUS

spec = importlib.util.spec_from_file_location("m", "edge_token_model.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42})
GEMM, NG, MEM = "#0C447C", "#993C1D", "#1F6E56"
fit = json.load(open("fitmap_results.json"))
val = json.load(open("validation/validation_results.json"))["summary"]

model = m.MODELS["Llama-3.2-1B"]
plats = [m.PLATFORMS[0], m.PLATFORMS[1], m.PLATFORMS[3]]
names = ["Data center\n(batched)", "Phone NPU\n(weights streamed)", "Weight-resident\nedge silicon"]
fig = plt.figure(figsize=(13.28, 5.31), dpi=100)
ax = fig.add_axes([0.05, 0.21, 0.36, 0.58])
for i, p in enumerate(plats):
    t = m.anatomy(model, 2048, p); tot = sum(t.values())
    g = 100 * t["Weight GEMV"] / tot; k = 100 * t["Attention + KV"] / tot; n = 100 - g - k
    ax.bar(i, g, color=GEMM, width=0.6); ax.bar(i, k, bottom=g, color=MEM, width=0.6); ax.bar(i, n, bottom=g + k, color=NG, width=0.6)
ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=11.5)
ax.set_ylabel("share of one decode token (%)", fontsize=11.5); ax.set_ylim(0, 100)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.annotate("", xy=(2.35, 106), xytext=(-0.35, 106), annotation_clip=False, arrowprops=dict(arrowstyle="->", lw=2.2, color="0.25"))
ax.text(1.0, 109, "the binding cost moves as the power budget tightens", ha="center", fontsize=11.5, style="italic", color="0.25")
for c, l, x in ((GEMM, "weight streaming", 0.05), (MEM, "KV traffic", 0.33), (NG, "non-GEMM", 0.52)):
    fig.text(x + 0.03, 0.93, "■ " + l, color=c, fontsize=12.5, fontweight="bold")
fig.text(0.05, 0.02, f"first-order model, Llama-3.2-1B, 2k context; checked on a Ryzen 7 5800HS CPU and an RTX 3050 GPU:\n"
         f"weight and KV terms within {val['gpu_holdout_abs_err_pct_max']:.0f}% (GPU hold-out) and {max(val['cpu_n0_abs_err_pct']):.0f}% (CPU)",
         fontsize=9.5, color="0.35")

n = len(CENSUS); si = sum(1 for r in CENSUS if r[7] == "Si")
ph8 = fit["Flagship phone (12 GB LPDDR5X)"]["Llama-3.1-8B"]["max_ctx_int8kv"] // 1000
fig.text(0.46, 0.84, "Component-aware survey of edge LLM hardware", fontsize=15.5, fontweight="bold", color=GEMM)
fig.text(0.46, 0.76, f"PRISMA-reported selection; census of {n} edge designs ({si} in measured silicon)\nclassified on five axes; per-component model checked on measured hardware", fontsize=10.5, color="0.3", va="top", linespacing=1.3)
items = [
 ("Where the token goes", "KV traffic (cloud) → weight streaming (streamed edge) → non-GEMM work\n(resident); the measured non-GEMM share is 5–35%, set by the software stack"),
 ("Reasoning workloads", "the KV cache equals the weight read at 18–37k tokens (1B model);\nlong-context KV hardware becomes the default case"),
 ("Deployability map", f"an 8B model at 4-bit weights fits a flagship phone only to ~{ph8}k tokens;\ncapacity is the first wall"),
 ("Decoder × hardware", "speculative and diffusion decoding speed up streamed hardware 2–4×\nbut slow weight-resident hardware"),
 ("Energy follows bytes", "GPU power is flat across precision and context, so J/token tracks\nbytes moved, ~1000× above milliwatt silicon"),
 ("Agenda", "on-device KV at reasoning lengths; non-GEMM inside in-memory\nfabrics; an evaluation standard"),
]
y = 0.655
for head, body in items:
    fig.text(0.46, y, head, fontsize=11, fontweight="bold", color=NG, va="top")
    fig.text(0.608, y, body, fontsize=9.6, va="top", linespacing=1.25)
    y -= 0.105
fig.savefig("figs/graphical_abstract.png", dpi=150)
fig.savefig("figs/graphical_abstract.pdf")
print("ok", n, si, ph8)
