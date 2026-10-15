#!/usr/bin/env python3
"""
make_figs.py -- regenerates every data-bearing figure and table of the survey
from census_data.py, the bibliography, and the reported figures listed below.

Outputs: figs/gapheat.pdf, figs/timeline.pdf, figs/compare.pdf, figs/priority.pdf,
         figs/corpus.pdf, tab_census.tex, census_numbers.tex
Every reported number below is taken from the primary source cited in the paper.
"""
import os, re, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from census_data import CENSUS
import bibtools
BIB = bibtools.load()

os.makedirs("figs", exist_ok=True)
GEMM, NG, MEM = "#0C447C", "#993C1D", "#1F6E56"
REG_COL = {"fw": "#8FB8D8", "1w": "#3E7CB1", "mw": "#0C2F5C"}
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})

N = len(CENSUS)
COMP_ROWS = [("GEMM", "GEMM / projections"), ("ATT", "Attention products"),
             ("KV", "KV-cache handling"), ("SM", "Softmax"), ("NORM", "Normalization"),
             ("ROPE", "RoPE"), ("ACT", "Activation"), ("MOE", "MoE routing")]
REGIMES = [("fw", "Several-watt"), ("1w", "Single-watt"), ("mw", "Milliwatt")]

# ------------------------------------------------------------------ gap matrix
counts = {(c, r): 0 for c, _ in COMP_ROWS for r, _ in REGIMES}
for row in CENSUS:
    for c in row[6]:
        counts[(c, row[4])] += 1
mat = np.array([[counts[(c, r)] for r, _ in REGIMES] for c, _ in COMP_ROWS], float)
fig, ax = plt.subplots(figsize=(3.45, 3.05))
cmap = LinearSegmentedColormap.from_list("g", ["#f4f4f2", "#cfe0d9", "#8fb8a8", "#3d8068", MEM])
ax.imshow(mat, cmap=cmap, vmin=0, vmax=mat.max(), aspect="auto")
for i, (ck, _) in enumerate(COMP_ROWS):
    for j, (rk, _) in enumerate(REGIMES):
        v = counts[(ck, rk)]
        col = "white" if v > mat.max() * 0.55 else ("#888888" if v == 0 else "#1a3d33")
        ax.text(j, i, f"{v}", ha="center", va="center", fontsize=9, color=col,
                fontweight="bold" if v else "normal")
ax.set_xticks(range(3)); ax.set_xticklabels(["Several-W", "Single-W", "mW"], fontsize=7.5, fontweight="bold")
ax.set_yticks(range(len(COMP_ROWS))); ax.set_yticklabels([l for _, l in COMP_ROWS], fontsize=7.5)
for t, (ck, _) in zip(ax.get_yticklabels(), COMP_ROWS):
    if ck == "ROPE": t.set_color(NG); t.set_fontweight("bold")
ax.xaxis.set_ticks_position("top"); ax.tick_params(length=0)
ax.set_title(r"tighter power budget $\longrightarrow$", fontsize=7.5, color="0.45", style="italic", pad=16)
r = [k for k, _ in COMP_ROWS].index("ROPE")
ax.add_patch(mpatches.Rectangle((-0.5, r - 0.48), 3.0, 0.96, fill=False, ec=NG, lw=1.6))
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig("figs/gapheat.pdf", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ timeline / evolution
years = list(range(2020, 2027))
IMPL_ORDER = [("Si", "measured silicon"), ("SoC", "commercial SoC"), ("FPGA", "FPGA prototype"),
              ("PLSyn", "post-layout / synthesis"), ("Sim", "simulation"), ("NR", "not stated")]
IMPL_COL = {"Si": "#0C2F5C", "SoC": "#3E7CB1", "FPGA": "#1F6E56", "PLSyn": "#8FB8A8",
            "Sim": "#D9C3A5", "NR": "#CCCCCC"}
def impl_key(s):
    s = s.split("/")[0]
    return "PLSyn" if s in ("PL", "Syn") else s
by_impl = {k: [0] * len(years) for k, _ in IMPL_ORDER}
by_reg = {k: [0] * len(years) for k, _ in REGIMES}
for row in CENSUS:
    y = max(row[2], 2020)
    by_impl[impl_key(row[7])][years.index(y)] += 1
    by_reg[row[4]][years.index(y)] += 1
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.35), sharey=True)
bot = np.zeros(len(years))
for k, lab in IMPL_ORDER:
    v = np.array(by_impl[k]); a1.bar(years, v, bottom=bot, color=IMPL_COL[k], label=lab, width=0.72); bot += v
a1.set_title("(a) census designs by year and evidence level", fontsize=8.5)
a1.legend(fontsize=6.5, frameon=False, loc="upper left")
a1.set_ylabel("designs in census")
bot = np.zeros(len(years))
for k, lab in REGIMES:
    v = np.array(by_reg[k]); a2.bar(years, v, bottom=bot, color=REG_COL[k], label=lab, width=0.72); bot += v
a2.set_title("(b) census designs by year and power class", fontsize=8.5)
a2.legend(fontsize=6.5, frameon=False, loc="upper left")
for a in (a1, a2):
    a.set_xticks(years); a.set_xticklabels([str(y) if y > 2020 else "$\\leq$2020" for y in years], fontsize=7)
fig.tight_layout(); fig.savefig("figs/timeline.pdf", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ compare (scatter + ceiling)
DESIGNS = [  # (label, params [B], reported decode tok/s, resident?)
    ("TerEffic", 0.37, 16300, True),
    ("TeLLMe v2", 0.73, 25, False),
    ("TeLLMe v1", 0.73, 9.51, False),
    ("LightMamba", 2.7, 7.2, False),
    ("Hummingbird (KV260)", 8.0, 4.8, False),
    ("LlamaF", 1.1, 1.5, False),
    ("FlightLLM", 7.0, 55, False),
    ("EdgeLLM", 6.0, 67, False),
]
import importlib.util
spec = importlib.util.spec_from_file_location("ebc", "edge_bottleneck_calc.py")
ebc = importlib.util.module_from_spec(spec); spec.loader.exec_module(ebc)
CEIL = []
m = ebc.ModelSpec("LLaMA3-8B", 8e9, 4096, 32, 4.0)
h = ebc.HardwareSpec("KV260", 19.2, 4.0, 8.0, nongemm_floor_us=8.0, bw_util=0.94)
CEIL.append(("Hummingbird\n(KV260)", ebc.estimate(m, h, ebc.Workload(512)).tok_per_s, 4.8))
m = ebc.ModelSpec("ternary 0.73B", 0.73e9, 1536, 24, 1.58)
h = ebc.HardwareSpec("edge FPGA", 17.1, 4.0, 4.8, nongemm_floor_us=6.0, bw_util=0.85)
CEIL.append(("TeLLMe v2", ebc.estimate(m, h, ebc.Workload(1024)).tok_per_s, 25))
m = ebc.ModelSpec("LLaMA2-7B", 7e9, 4096, 32, 4.0)
h = ebc.HardwareSpec("U280", 460.0, 8.0, 45.0, nongemm_floor_us=8.0, bw_util=0.85)
CEIL.append(("FlightLLM", ebc.estimate(m, h, ebc.Workload(512)).tok_per_s, 55))
m = ebc.ModelSpec("370M ternary", 370e6, 1024, 24, 1.58)
h = ebc.HardwareSpec("on-chip", 19.2, 128.0, 27.0, nongemm_floor_us=2.0, bw_util=0.85)
CEIL.append(("TerEffic\n(resident)", ebc.estimate(m, h, ebc.Workload(512)).tok_per_s, 16300))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
OFF = {"FlightLLM": (-40, 6), "EdgeLLM": (5, -10), "TeLLMe v1": (5, -9), "PD-Swap": (-30, 6),
       "TeLLMe v2": (5, 1), "Hummingbird (KV260)": (-62, -10), "LightMamba": (4, 4), "LlamaF": (4, 3)}
for lab, p, t, res in DESIGNS:
    a1.scatter(p, t, s=34, c=MEM if res else GEMM, marker="s" if res else "o", zorder=3,
               edgecolor="white", lw=0.6)
    a1.annotate(lab, (p, t), textcoords="offset points", xytext=OFF.get(lab, (4, 4)), fontsize=6.3)
a1.axhspan(2000, 60000, color=MEM, alpha=0.08)
a1.text(1.6, 5000, "weights resident on chip", color=MEM, fontsize=6.5, style="italic")
a1.axhspan(0.8, 150, color=GEMM, alpha=0.05)
a1.text(0.3, 100, "weights streamed off chip", color=GEMM, fontsize=6.5, style="italic")
a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlim(0.25, 12); a1.set_ylim(0.8, 6e4)
a1.set_xlabel("model size (billion parameters)"); a1.set_ylabel("reported decode tok/s")
a1.set_title("(a) reported throughput vs. model size", fontsize=8.5)
labs = [c[0] for c in CEIL]; ratio = [c[2] / c[1] for c in CEIL]
cols = [MEM if "resident" in l else GEMM for l in labs]
a2.barh(range(len(CEIL)), ratio, color=cols, height=0.55)
a2.axvline(1.0, color=NG, lw=1, ls="--"); a2.text(1.08, 2.45, "memory-bound\nceiling", color=NG, fontsize=6.2)
for i, (l, c, rp) in enumerate(CEIL):
    a2.text(ratio[i] * 1.12, i, f"{rp:g} / {c:.3g} tok/s", va="center", fontsize=6.3)
a2.set_xscale("log"); a2.set_xlim(0.1, 400)
a2.set_yticks(range(len(CEIL))); a2.set_yticklabels(labs, fontsize=6.8); a2.invert_yaxis()
a2.set_xlabel("reported / memory-bound ceiling")
a2.set_title("(b) reported throughput relative to ceiling", fontsize=8.5)
fig.tight_layout(); fig.savefig("figs/compare.pdf", bbox_inches="tight"); plt.close(fig)
print("ceilings:", [(c[0].split()[0], round(c[1], 2), c[2], round(c[2] / c[1], 2)) for c in CEIL])

# ------------------------------------------------------------------ priority map
GAPS = [  # label, difficulty(1-3), headroom(1-3), importance(1-3), verdict, horizon, label offset
    ("Non-GEMM CIM", 3, 3, 3, "Priority 1", "mid-term", (0, -20)),
    ("On-device KV", 2, 3, 3, "Priority 1", "near-term", (0, 12)),
    ("Evaluation\nstandard", 1, 3, 3, "Priority 1", "immediate", (0, -24)),
    ("RoPE datapath", 2, 2, 2, "Conditional", "conditional", (0, -19)),
    ("Edge MoE routing", 3, 2, 2, "Conditional", "after software caching", (0, -19)),
    ("Norm. at mW", 1, 1.2, 2, "Deprioritized", "", (40, -3)),
    ("Ternary LUT GEMM", 1, 0.55, 3, "Largely solved", "", (50, -3)),
]
VCOL = {"Priority 1": NG, "Conditional": "#D08A2E", "Deprioritized": "#8A8A8A", "Largely solved": MEM}
fig, ax = plt.subplots(figsize=(3.45, 3.0))
for lab, d, hd, imp, v, hz, off in GAPS:
    ax.scatter(d, hd, s=55 * imp ** 1.6, color=VCOL[v], alpha=0.85, edgecolor="white", lw=0.8, zorder=3)
    txt = lab + (f"\n({hz})" if hz else "")
    ax.annotate(txt, (d, hd), textcoords="offset points", xytext=off, ha="center", va="center",
                fontsize=5.8)
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["low", "medium", "high"])
ax.set_yticks([0.55, 1.2, 2, 3]); ax.set_yticklabels(["solved", "low", "medium", "high"])
ax.set_xlabel("implementation difficulty / risk"); ax.set_ylabel("recoverable headroom")
ax.set_xlim(0.45, 3.6); ax.set_ylim(0.25, 3.5); ax.grid(alpha=0.25)
for v, c in VCOL.items():
    ax.scatter([], [], color=c, s=25, label=v)
ax.legend(fontsize=5.8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
          columnspacing=0.6, handletextpad=0.2)
fig.tight_layout(); fig.savefig("figs/priority.pdf", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ census table (two page-wide tables)
PH = {"P": "P", "D": "D", "P+D": "P+D", "op": "op."}
RG = {"fw": "several-W", "1w": "single-W", "mw": "mW"}
CN = {"GEMM": "GEMM", "ATT": "Att", "KV": "KV\\$", "SM": "SM", "NORM": "Norm", "ROPE": "RoPE",
      "ACT": "Act", "MOE": "MoE"}
FL = {"t": "$^{t}$", "r": "$^{a}$", "e": "$^{b}$"}
CAPTION = ("The edge-regime design census (cut-off September 2026; $n=%d$). Regime classes are "
     "defined in \\cref{sec:scope}; operator-level units (phase op.) are classified by the regime of their "
     "intended integration target. Phase: P = prefill, D = decode. Components: Att = attention products, "
     "KV\\$ = KV-cache handling, SM = Softmax, Norm = normalization, Act = activation. Impl.: Si = measured "
     "silicon; SoC = measured on commercial mobile silicon; FPGA = FPGA prototype; PL = post-layout; "
     "Syn = post-synthesis; Sim = simulation; NR = not stated. Rows were checked against the primary "
     "source, in full text wherever it was accessible. $^{a}$Regime inferred: power not stated in the text (only in figures) "
     "or derived from reported throughput and efficiency; $^{b}$evaluated on encoder or "
     "short-sequence Transformer workloads; $^{p}$preprint (arXiv only at the cut-off). The classification rule is given in \\cref{sec:census_rule}.") % N
HEAD = "\\toprule Design & Year & Platform & Regime & Phase & Components & Impl.\\\\ \\midrule"
order = {"fw": 0, "1w": 1, "mw": 2}
rows = []
for name, key, yr, plat, reg, ph, comps, impl, fl in sorted(CENSUS, key=lambda r: (order[r[4]], -r[2], r[0])):
    marks = "".join(FL[f] for f in fl) + ("$^{p}$" if bibtools.is_preprint(BIB[key]) else "")
    rows.append(f"{name}~\\cite{{{key}}}{marks} & {yr} & {plat} & {RG[reg]} & {PH[ph]} & "
                f"{', '.join(CN[c] for c in comps)} & {impl} \\\\")
L = ["% ---- AUTO-GENERATED by make_figs.py from census_data.py ----",
     "\\begin{table*}[p]", "\\centering\\fontsize{6.2}{7.2}\\selectfont\\setlength{\\tabcolsep}{3.5pt}",
     "\\caption{%s}\\label{tab:census}" % CAPTION,
     "\\begin{tabular}{@{}p{3.4cm}lp{4.3cm}llp{3.9cm}l@{}}", HEAD] + rows + ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
open("tab_census.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

# numbers used in the text
silicon = sum(1 for r in CENSUS if r[7] == "Si")
recent = sum(1 for r in CENSUS if r[2] >= 2025)
mw = sum(1 for r in CENSUS if r[4] == "mw")
open("census_numbers.tex", "w").write(
    "\\renewcommand{\\ncensus}{%d}\n\\newcommand{\\nsilicon}{%d}\n\\newcommand{\\nrecent}{%d}\n"
    "\\newcommand{\\nmw}{%d}\n" % (N, silicon, recent, mw))
print(f"census n={N}, silicon={silicon}, 2025-26={recent}, mW={mw}")
print("gap counts:", {f"{c}-{r}": v for (c, r), v in counts.items()})
