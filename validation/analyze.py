#!/usr/bin/env python3
"""
analyze.py -- compares the first-order model (Eq. model) with measured batch-1 decode on
the test laptop and draws figs/validation.pdf.

Measurements (llama.cpp b11501, llama-bench, Llama-3.2-1B-Instruct GGUF, n_gen = 64,
context depths 0/1k/4k/8k, f16 KV cache):
  CPU: AMD Ryzen 7 5800HS, 8 threads, dual-channel DDR4-3200 (measured read BW 28.7 GB/s)
  GPU: NVIDIA RTX 3050 Laptop (4 GB GDDR6), all layers offloaded
Model: t_tok = (B_W + B_KV(n)) / beta + tau
  CPU : beta = measured read bandwidth, tau = 0      (no fitted parameter)
  GPU : beta, tau fitted on two points only (F16 and Q4_0 at n = 0); the other ten points
        (Q8_0 and all non-zero depths) are held out.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(H, "..", "figs")
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
GEMM, NG, MEM = "#0C447C", "#993C1D", "#1F6E56"

SIZE = {"Q4_0": 773025920, "Q8_0": 1321083008, "f16": 2479595360}   # bytes read per token (tied LM head)
KV_PER_POS = 16 * 2 * 8 * 64 * 2                                       # bytes per cached position (f16)
BETA_CPU = 28.7e9

def load(dev, q):
    f = os.path.join(H, f"{dev}_{q}.json")
    return {r["n_depth"]: r["avg_ts"] for r in json.load(open(f))}
meas = {(d, q): load(d, q) for d in ("cpu", "gpu") for q in SIZE}
# the zero-depth CPU points were re-measured with 5 repetitions (thermal settling)
for q in ("Q4_0", "Q8_0"):
    meas[("cpu", q)][0] = json.load(open(os.path.join(H, f"cpu_{q}_d0.json")))[0]["avg_ts"]

# --- GPU fit on two points
t16 = 1 / meas[("gpu", "f16")][0]; t4 = 1 / meas[("gpu", "Q4_0")][0]
beta_gpu = (SIZE["f16"] - SIZE["Q4_0"]) / (t16 - t4)
tau_gpu = t4 - SIZE["Q4_0"] / beta_gpu

def pred(dev, q, n):
    b = SIZE[q] + KV_PER_POS * n
    return 1 / (b / BETA_CPU) if dev == "cpu" else 1 / (b / beta_gpu + tau_gpu)

rows = []
for (dev, q), m in meas.items():
    for n, ts in sorted(m.items()):
        p = pred(dev, q, n)
        rows.append(dict(dev=dev, q=q, n=n, meas=ts, pred=p, err=100 * (p - ts) / ts,
                         fit=(dev == "gpu" and n == 0 and q in ("f16", "Q4_0"))))
summary = {
    "beta_cpu_GBps": BETA_CPU / 1e9, "beta_gpu_GBps": beta_gpu / 1e9, "tau_gpu_ms": 1e3 * tau_gpu,
    "cpu_n0_abs_err_pct": [round(abs(r["err"]), 1) for r in rows if r["dev"] == "cpu" and r["n"] == 0],
    "gpu_holdout_abs_err_pct_max": round(max(abs(r["err"]) for r in rows if r["dev"] == "gpu" and not r["fit"]), 1),
    "gpu_holdout_abs_err_pct_mean": round(float(np.mean([abs(r["err"]) for r in rows if r["dev"] == "gpu" and not r["fit"]])), 1),
    "cpu_deep_err_pct": {f'{r["q"]}@{r["n"]}': round(r["err"], 0) for r in rows if r["dev"] == "cpu" and r["n"] > 0},
    "gpu_tau_share_pct": {q: round(100 * tau_gpu * meas[("gpu", q)][0], 0) for q in SIZE},
    "cpu_us_per_pos_meas": round(1e6 * (1 / meas[("cpu", "Q4_0")][8192] - 1 / meas[("cpu", "Q4_0")][0]) / 8192, 2),
    "gpu_us_per_pos_meas": round(1e6 * (1 / meas[("gpu", "Q4_0")][8192] - 1 / meas[("gpu", "Q4_0")][0]) / 8192, 3),
    "cpu_us_per_pos_bw": round(1e6 * KV_PER_POS / BETA_CPU, 2),
    "gpu_us_per_pos_bw": round(1e6 * KV_PER_POS / beta_gpu, 3),
}
json.dump({"summary": summary, "rows": rows}, open(os.path.join(H, "validation_results.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))

prof = {d: json.load(open(os.path.join(H, f"prof_{d}.json"))) for d in ("cpu", "cuda")}

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw={"width_ratios": [1, 1, 1.15]})
# (a) parity
mk = {"Q4_0": "o", "Q8_0": "s", "f16": "^"}
for r in rows:
    col = GEMM if r["dev"] == "gpu" else MEM
    face = "white" if r["fit"] else col
    a1.scatter(r["meas"], r["pred"], marker=mk[r["q"]], s=20, edgecolor=col, facecolor=face, lw=0.9, zorder=3)
lim = [4, 200]
a1.plot(lim, lim, color="0.5", lw=0.7, ls="--"); a1.set_xscale("log"); a1.set_yscale("log")
a1.set_xlim(lim); a1.set_ylim(lim)
a1.set_xlabel("measured decode (tok/s)"); a1.set_ylabel("model prediction (tok/s)")
a1.set_title("(a) model vs. measurement", fontsize=8.5)
h1 = [a1.scatter([], [], color=col, s=14, label=lab) for lab, col in (("RTX 3050 (GPU)", GEMM), ("Ryzen 7 (CPU)", MEM))]
h2 = [a1.scatter([], [], marker=m, facecolor="none", edgecolor="0.3", s=14, label=q.replace("f16", "FP16")) for q, m in mk.items()]
h3 = [a1.scatter([], [], marker="o", facecolor="white", edgecolor="0.3", s=14, label="fit point")]
lg = a1.legend(handles=h1 + h3, fontsize=5.8, frameon=False, loc="upper left"); a1.add_artist(lg)
a1.legend(handles=h2, fontsize=5.8, frameon=False, loc="lower right")
# (b) context sweep, Q4_0
ns = np.array([0, 1024, 4096, 8192])
for dev, col, lab in (("gpu", GEMM, "RTX 3050"), ("cpu", MEM, "Ryzen 7")):
    m = meas[(dev, "Q4_0")]
    a2.plot(ns / 1024, [m[n] for n in ns], "o-", color=col, ms=3.5, label=f"{lab} measured")
    a2.plot(ns / 1024, [pred(dev, "Q4_0", n) for n in ns], "--", color=col, lw=1, label=f"{lab} model")
a2.set_xticks([0, 1, 4, 8]); a2.set_yscale("log"); a2.set_xlabel("context length (k tokens)"); a2.set_ylabel("decode tok/s (Q4_0)")
a2.set_title("(b) KV term vs. context", fontsize=8.5)
a2.legend(fontsize=5.3, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.44), columnspacing=0.6, handlelength=1.4)
# (c) eager-PyTorch anatomy
cats = ["GEMV", "Attn+KV", "Softmax", "Norm", "RoPE", "Act", "Sampling", "Other"]
cols = [GEMM, MEM, "#7A2E14", "#C2643F", "#DE936F", "#F0C2A8", "#5a1f0d", "#BBBBBB"]
bars = [("Ryzen 7, 512", prof["cpu"]["runs"][0]), ("Ryzen 7, 4k", prof["cpu"]["runs"][2]),
        ("RTX 3050, 512", prof["cuda"]["runs"][0]), ("RTX 3050, 8k", prof["cuda"]["runs"][2])]
for i, (lab, run) in enumerate(bars):
    left = 0
    for c, col in zip(cats, cols):
        v = run["share_pct"].get(c, 0)
        a3.barh(i, v, left=left, color=col, edgecolor="white", lw=0.3, label=c if i == 0 else None)
        left += v
a3.set_yticks(range(len(bars))); a3.set_yticklabels([b[0] for b in bars], fontsize=6.2); a3.invert_yaxis()
a3.set_xlim(0, 100); a3.set_xlabel("share of token time, eager PyTorch (%)")
a3.set_title("(c) measured operator anatomy", fontsize=8.5)
a3.legend(fontsize=5.4, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.50))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "validation.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "validation.png"), dpi=160, bbox_inches="tight")
