#!/usr/bin/env python3
"""
edge_token_model.py -- first-order, per-component decode-token model.

Reproduces the two analytical figures of the survey:
  figs/comp_roofline.pdf : arithmetic intensity of each Transformer component
                           (batch-1 decode and 1k-token prefill) against the
                           rooflines of three representative platforms.
  figs/token_anatomy.pdf : per-component share of one decode token's latency in
                           four deployment configurations, and the sensitivity of
                           the non-GEMM share to non-GEMM throughput and context.

Everything is analytical: operation and byte counts follow directly from the
public model configurations below; platform parameters are nominal, rounded,
representative values (Table "tab:modelparams" in the paper), NOT measurements
of any specific product.  Components are summed serially (no overlap), which
is an upper bound on exposed non-GEMM latency: perfect fusion would hide part
of it behind matrix work.  Change any parameter and re-run.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
C_GEMM, C_ATT, C_NG, C_KV = "#0C447C", "#4F86C6", "#993C1D", "#1F6E56"
C_NG2 = ["#993C1D", "#C2643F", "#DE936F", "#F0C2A8", "#7A2E14"]

# --------------------------------------------------------------------------
# Public model configurations (Hugging Face config.json of each model)
# --------------------------------------------------------------------------
MODELS = {
    "Llama-3.2-1B": dict(d=2048, L=16, H=32, Hkv=8, dh=64, dff=8192, V=128256),
    "Llama-3.1-8B": dict(d=4096, L=32, H=32, Hkv=8, dh=128, dff=14336, V=128256),
}

# cost, in simple vector operations, of one element of each non-GEMM operator
# (exp, reciprocal, rsqrt, sin/cos each counted as 4 simple ops)
OPS = dict(softmax=8,      # max-compare, subtract, exp(4), accumulate, scale
           norm=4,         # square, accumulate, scale by gamma, x*rsqrt
           rope=3,         # 2 mul + 1 add per output element (cos/sin precomputed)
           act=10,         # SiLU: exp(4), add, reciprocal(4), gate multiply
           sample=8)       # vocabulary softmax / top-k over the LM-head logits


def per_token(m, n, wbits, kvbits):
    """Operation and byte counts for ONE decode token at context length n."""
    d, L, H, Hkv, dh, dff, V = (m[k] for k in ("d", "L", "H", "Hkv", "dh", "dff", "V"))
    w_layer = d * H * dh + 2 * d * Hkv * dh + H * dh * d + 3 * d * dff
    w_tot = L * w_layer + V * d                     # + LM head (tied embedding)
    c = {}
    c["gemv_flop"] = 2 * w_tot
    c["w_bytes"] = w_tot * wbits / 8
    c["att_flop"] = L * 4 * H * n * dh              # QK^T and AV
    c["kv_bytes"] = L * 2 * n * Hkv * dh * kvbits / 8
    c["softmax"] = L * H * n * OPS["softmax"]
    c["norm"] = (2 * L + 1) * d * OPS["norm"]
    c["rope"] = L * (H + Hkv) * dh * OPS["rope"]
    c["act"] = L * dff * OPS["act"]
    c["sample"] = V * OPS["sample"]
    return c


# --------------------------------------------------------------------------
# Representative platform configurations (nominal, rounded)
#   P   : matrix-engine peak (op/s)      BW : off-chip bandwidth (B/s)
#   T_ng: non-GEMM (vector/scalar) throughput (simple op/s)
#   B   : batch size sharing each weight fetch
#   resident: weights (and KV) held on chip; BW_on = on-chip bandwidth
# --------------------------------------------------------------------------
PLATFORMS = [
    dict(name="Data-center GPU\nFP16, batch 64", P=312e12, BW=2.0e12, T_ng=19.5e12,
         B=64, wbits=16, kvbits=16, resident=False),
    dict(name="Phone SoC NPU\nW4, batch 1", P=30e12, BW=60e9, T_ng=1e12,
         B=1, wbits=4, kvbits=8, resident=False),
    dict(name="Embedded FPGA\nW4, batch 1", P=1.4e12, BW=19.2e9, T_ng=19.2e9,
         B=1, wbits=4, kvbits=8, resident=False),
    dict(name="Weight-resident\nternary CIM", P=20e12, BW=1.0e12, T_ng=32e9,
         B=1, wbits=1.58, kvbits=8, resident=True),
]


def anatomy(m, n, p, T_ng=None):
    c = per_token(m, n, p["wbits"], p["kvbits"])
    T = p["T_ng"] if T_ng is None else T_ng
    t = {}
    w_mem = 0.0 if p["resident"] else c["w_bytes"] / p["B"]
    t["Weight GEMV"] = max(c["gemv_flop"] / p["P"], w_mem / p["BW"])
    t["Attention + KV"] = max(c["att_flop"] / p["P"], c["kv_bytes"] / p["BW"])
    for k, lab in (("softmax", "Softmax"), ("norm", "Norm"), ("rope", "RoPE"),
                   ("act", "Activation"), ("sample", "Sampling")):
        t[lab] = c[k] / T
    return t


def fig_anatomy():
    m = MODELS["Llama-3.2-1B"]
    n = 2048
    labels = ["Weight GEMV", "Attention + KV", "Softmax", "Norm", "RoPE",
              "Activation", "Sampling"]
    cols = [C_GEMM, C_KV] + C_NG2
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.9),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    rows = []
    for i, p in enumerate(PLATFORMS):
        t = anatomy(m, n, p)
        tot = sum(t.values())
        left = 0.0
        for lab, col in zip(labels, cols):
            share = 100 * t[lab] / tot
            ax.barh(i, share, left=left, color=col, edgecolor="white", lw=0.4,
                    label=lab if i == 0 else None)
            left += share
        ng = 100 * sum(t[k] for k in labels[2:]) / tot
        ngs = "<1" if ng < 1 else f"{ng:.0f}"
        ax.text(101, i, f"{tot*1e3:.3g} ms/token\nnon-GEMM {ngs}%", va="center",
                fontsize=7)
        rows.append((p["name"].replace("\n", ", "), tot * 1e3, ng))
    ax.set_yticks(range(len(PLATFORMS)))
    ax.set_yticklabels([p["name"] for p in PLATFORMS], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of one decode token's latency (%)")
    ax.set_title("(a) Llama-3.2-1B, context 2048", fontsize=8.5)
    ax.legend(ncol=4, fontsize=6.5, loc="upper center",
              bbox_to_anchor=(0.55, -0.2), frameon=False)

    # (b) sensitivity: non-GEMM share vs non-GEMM throughput, resident config
    p = PLATFORMS[3]
    Ts = np.logspace(9, 13, 60)
    for n_, ls in ((512, ":"), (2048, "-"), (8192, "--")):
        sh = []
        for T in Ts:
            t = anatomy(m, n_, p, T_ng=T)
            sh.append(100 * sum(t[k] for k in labels[2:]) / sum(t.values()))
        bx.semilogx(Ts, sh, ls, color=C_NG, label=f"context {n_}")
    for p2, mk, off in ((PLATFORMS[2], "s", (-34, 16)), (PLATFORMS[1], "^", (4, 9))):
        t = anatomy(m, 2048, p2)
        sh = 100 * sum(t[k] for k in labels[2:]) / sum(t.values())
        bx.plot(p2["T_ng"], sh, mk, color="k", ms=4)
        bx.annotate(p2["name"].split("\n")[0] + "\n(weights streamed)",
                    (p2["T_ng"], sh), xytext=off, textcoords="offset points",
                    fontsize=6, arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4"))
    bx.axhline(50, color="0.6", lw=0.6)
    bx.set_xlabel("non-GEMM throughput (op/s)")
    bx.set_ylabel("non-GEMM share (%)")
    bx.set_ylim(0, 100)
    bx.set_title("(b) weight-resident configuration", fontsize=8.5)
    bx.legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig("figs/token_anatomy.pdf", bbox_inches="tight")
    fig.savefig("figs/token_anatomy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return rows


def fig_roofline():
    """Arithmetic intensity of each component vs three platform rooflines."""
    m = MODELS["Llama-3.1-8B"]
    n_ctx, n_pre = 2048, 1024
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    I = np.logspace(-1, 4, 200)
    for p, ls in ((PLATFORMS[0], "-"), (PLATFORMS[1], "--"), (PLATFORMS[2], ":")):
        ax.loglog(I, np.minimum(p["P"], I * p["BW"]), ls, color="0.35", lw=1)
        ax.text(1.2e4, p["P"] * 1.15, p["name"].split("\n")[0], fontsize=6,
                ha="right", va="bottom", color="0.3")
    c = per_token(m, n_ctx, 4, 8)
    pts = [
        ("GEMV, W4 (decode)", c["gemv_flop"] / c["w_bytes"], C_GEMM, "o"),
        ("GEMV, FP16 (decode)", 2 / 2, C_GEMM, "s"),
        ("Attention, INT8 KV (decode)", c["att_flop"] / c["kv_bytes"], C_KV, "D"),
        ("GEMM, W4 (prefill, 1k tok.)", 2 * n_pre / 0.5, C_GEMM, "*"),
        ("GEMV, W4, batch 64", 64 * 2 / 0.5, C_ATT, "v"),
    ]
    for lab, ai, col, mk in pts:
        y = min(PLATFORMS[1]["P"], ai * PLATFORMS[1]["BW"])
        ax.plot(ai, y, mk, color=col, ms=5, label=f"{lab}: {ai:.0f} op/B")
    ax.axvspan(0.1, PLATFORMS[1]["P"] / PLATFORMS[1]["BW"], color=C_KV, alpha=0.06)
    ax.set_xlabel("arithmetic intensity (op/byte)")
    ax.set_ylabel("throughput (op/s)")
    ax.set_xlim(0.1, 1.5e4)
    ax.set_ylim(1e9, 1e15)
    ax.legend(fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2, frameon=False, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig("figs/comp_roofline.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig("figs/comp_roofline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pts


if __name__ == "__main__":
    rows = fig_anatomy()
    for r in rows:
        print(f"{r[0]:45s} {r[1]:9.3f} ms/token  non-GEMM {r[2]:5.1f}%")
    for lab, ai, *_ in fig_roofline():
        print(f"{lab:32s} AI = {ai:8.1f} op/B")
    m = MODELS["Llama-3.2-1B"]
    for n in (512, 2048, 8192, 32768):
        c = per_token(m, n, 4, 8)
        print(f"1B, n={n:6d}: weights {c['w_bytes']/1e6:7.1f} MB  KV {c['kv_bytes']/1e6:7.1f} MB  "
              f"non-GEMM ops {(c['softmax']+c['norm']+c['rope']+c['act']+c['sample'])/1e6:6.2f} M")
