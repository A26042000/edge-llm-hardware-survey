#!/usr/bin/env python3
"""
make_corpus.py -- corpus statistics (Fig. corpus) from the references actually cited
in the compiled manuscript (manuscript.aux) and their years in edge_llm_refs.bib.
Run after a first LaTeX pass.
"""
import re, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from census_data import CENSUS

plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})

aux = open("manuscript.aux", encoding="utf-8").read()
cited = set()
for m in re.findall(r"\\citation\{([^}]*)\}", aux):
    cited.update(k.strip() for k in m.split(","))

bib = open("edge_llm_refs.bib", encoding="utf-8").read()
year = {}
for ent in re.split(r"\n(?=@)", bib):
    m = re.match(r"\s*@\w+\s*\{\s*([^,\s]+)\s*,", ent)
    if not m:
        continue
    y = re.search(r"\byear\s*=\s*[{\"]?(\d{4})", ent)
    if y:
        year[m.group(1)] = int(y.group(1))

SURVEYS = set("""bai_resource blankestijn2026fpga cai_edge_inf chen_edge_inference chitty_transformer
edge_eff_slm friha_edge guo_codesign jli_hwperspective kachris kachris2026algorithmics kang2024fpgaasic
ke_component kim_converging kim_fullstack kim_giants koilia kvcache_survey li2026posenc liu2026moesurvey
luo2024embeddedinfra mahbub_softmax navardi_genai patel2026memorywall qu_mei song2025fpgallm wolters2024cim
xu2025hwnnsurvey xu_ondevice xu_resource zheng_edge zhou2024efficientinf""".split())
ALGSYS = set("""alibi awq bitnet bitnet158 bitnet_a48 bitnetcpp cachegen commvq daop deepseekr1 devlin_bert
distserve eliseev_moe flashattention flashdecoding flexgen gear gptq gta_gla h2o hybrimoe keyformer kivi kvquant
medusa minicache minicpm4 mistral7b moeinfinity moelens qwen25 roformer sarathi smoothquant snapkv speculative
splitwise vaswani2017 vllm specee melt edgebench bench360 mlperf sustainable_edge nongemm_bench llmcompass
bambhaniya2024genz yuan2024llmviewer atmer2025prefill ma2026challenges roofline kitchenham_slr prisma2020
chen2026batch1 murthy2026phase operator_fusion wei2025agentxpu mla_hw zhao2026cosm lee2026pimsherpa llmflash
pittet2026ssr tmac yu2026nli llamacpp wolf2020transformers grattafiori2024llama3 lee2025ropefast
baek2025fireq wu2025polarquant deepseekv2 dao2024mamba2 yang2025gdn lieber2024jamba nie2025llada arriola2025bd3lm
wu2025fastdllm li2024eagle snell2024ttc openai2025gptoss""".split())
CENSUSK = {r[1] for r in CENSUS}

def cat(k):
    if k in CENSUSK: return "edge-regime designs (census)"
    if k in SURVEYS: return "surveys"
    if k in ALGSYS: return "models, algorithms, systems, benchmarks"
    return "other hardware (baselines, out-of-regime)"

CATS = ["edge-regime designs (census)", "other hardware (baselines, out-of-regime)",
        "models, algorithms, systems, benchmarks", "surveys"]
COLS = ["#0C447C", "#8FB8D8", "#C9B48F", "#993C1D"]
bins = ["$\\leq$2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"]
def b(y):
    return 0 if y <= 2019 else min(y, 2026) - 2019
cnt = {c: [0] * len(bins) for c in CATS}
missing = []
for k in cited:
    if k not in year:
        missing.append(k); continue
    cnt[cat(k)][b(year[k])] += 1
tot = {c: sum(v) for c, v in cnt.items()}
fig, ax = plt.subplots(figsize=(3.45, 2.5))
bot = np.zeros(len(bins))
for c, col in zip(CATS, COLS):
    v = np.array(cnt[c]); ax.bar(range(len(bins)), v, bottom=bot, color=col, width=0.75,
                                 label=f"{c} ({tot[c]})"); bot += v
ax.set_xticks(range(len(bins))); ax.set_xticklabels(bins, fontsize=6.8)
ax.set_ylabel("cited works")
ax.set_ylim(0, 112); ax.legend(fontsize=6, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("figs/corpus.pdf", bbox_inches="tight"); plt.close(fig)
n = len(cited)
recent = sum(1 for k in cited if year.get(k, 0) >= 2024)
import bibtools
BIB = bibtools.load()
pre = [k for k in cited if k in BIB and bibtools.is_preprint(BIB[k])]
pre26 = [k for k in pre if year.get(k, 0) >= 2026]
n26 = sum(1 for k in cited if year.get(k, 0) >= 2026)
open("corpus_numbers.tex", "w").write(
    "\\newcommand{\\ncited}{%d}\n\\newcommand{\\nrecentrefs}{%d}\n\\newcommand{\\npctrecent}{%d}\n"
    "\\newcommand{\\ncitedpre}{%d}\n\\newcommand{\\npctpre}{%d}\n\\newcommand{\\ncitedtwentysix}{%d}\n"
    "\\newcommand{\\npretwentysix}{%d}\n"
    % (n, recent, round(100 * recent / n), len(pre), round(100 * len(pre) / n), n26, len(pre26)))
print("preprints", len(pre), "of", n, "; 2026:", len(pre26), "of", n26)
print("cited", n, "2024+", recent, tot, "missing year:", missing)
