#!/usr/bin/env python3
"""
screen.py -- deduplication, rule-based title/abstract screening, and matching against the
included corpus (the works cited in the survey) and the design census.

Writes screened.jsonl (every unique record with its screening decision), prisma_counts.json,
and candidates.csv (records that pass every rule, look like edge-regime hardware designs, and
are not in the census -- the list assessed by hand, see assessed.csv).
"""
import json, os, re, glob, sys, collections, csv
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, ".."))
import bibtools
from census_data import CENSUS

def norm_title(t):
    t = re.sub(r"\\math(rm|bf|it)|[{}\\$^]", "", t or "").lower()
    return re.sub(r"[^a-z0-9]+", "", t)[:90]

def arxiv_id(rec):
    m = re.search(r"(\d{4}\.\d{4,5})", (rec.get("doi") or "") + " " + (rec.get("id") or "") if rec["src"] == "arxiv"
                  or "arxiv" in (rec.get("doi") or "") else "")
    return m.group(1) if m else None

# ------------------------------------------------------------------ load + dedup
recs = []
for f in sorted(glob.glob(os.path.join(H, "raw", "*.jsonl"))):
    q = os.path.basename(f)[:-6]
    for line in open(f, encoding="utf-8"):
        r = json.loads(line); r["query"] = q; recs.append(r)
identified = collections.Counter(r["src"] for r in recs)
uniq, seen = [], {}
for r in recs:
    keys = [k for k in (("doi", r["doi"]) if r["doi"] and "arxiv" not in r["doi"] else None,
                        ("ax", arxiv_id(r)), ("t", norm_title(r["title"])) if len(norm_title(r["title"])) > 25 else None) if k and k[1]]
    hit = next((seen[k] for k in keys if k in seen), None)
    if hit is None:
        r["queries"] = {r["query"]}; uniq.append(r); hit = r
    else:
        hit["queries"].add(r["query"])
        if not hit.get("abstract") and r.get("abstract"): hit["abstract"] = r["abstract"]
        if hit["src"] == "arxiv" and r["src"] == "openalex" and r.get("venue") not in (None, "arXiv", "arXiv (Cornell University)"):
            hit["venue"] = r["venue"]
    for k in keys: seen.setdefault(k, hit)

# ------------------------------------------------------------------ rules (title + abstract)
RX = lambda p: re.compile(p, re.I)
WORK = RX(r"\b(large language models?|LLMs?|language models?|transformers?|GPT|BERT|ViT|self-attention|attention)\b")
HW = RX(r"(accelerat|FPGA|ASIC|\bNPU|in-memory|\bPIM\b|\bCIM\b|chip|silicon|hardware|circuit|processor|\bSoC\b|systolic|datapath|micro-?architecture|\bnm\b|RTL)")
EXCL = [
    ("LLMs used for hardware design or EDA", RX(r"(verilog|\bRTL (code )?generation|HDL generation|hardware description language|LLM-aided|LLM-based (hardware|chip|circuit) design|"
                                               r"design automation|electronic design|\bEDA\b|code generation|analog circuit (design|sizing)|high-level synthesis code)")),
    ("not about Transformer/LLM inference workloads", RX(r"(power transformer|distribution transformer|transformer winding|substation|transformer oil|current transformer|dissolved gas)")),
    ("hardware security / reliability only", RX(r"(side-channel|fault injection|hardware trojan|adversarial attack|bit-?flip attack|soft error|radiation)")),
    ("training hardware only", RX(r"^(?!.*inference).*\btraining (accelerat|hardware|system)")),
]

def decide(r):
    text = (r["title"] or "") + " " + (r.get("abstract") or "")
    if r["queries"] & {"openalex_QC", "arxiv_QC"} and RX(r"\b(survey|review|tutorial)\b").search(r["title"] or ""):
        return "include-survey", None
    for why, rx in EXCL:
        if rx.search(r["title"] or "") or (why != "training hardware only" and rx.search(text) and not HW.search(r["title"] or "")):
            return "exclude", why
    if not WORK.search(text):
        return "exclude", "not about Transformer/LLM inference workloads"
    if not HW.search(text):
        return "exclude", "no hardware contribution (algorithm or software only)"
    return "pass", None

# ------------------------------------------------------------------ corpus + census
B = bibtools.load(os.path.join(H, "..", "edge_llm_refs.bib"))
cited = set()
for f in glob.glob(os.path.join(H, "..", "sec_*.tex")) + glob.glob(os.path.join(H, "..", "tab_*.tex")) + glob.glob(os.path.join(H, "..", "fig_*.tex")):
    for m in re.finditer(r"\\cite[pt]?\{([^}]*)\}", open(f, encoding="utf-8").read()):
        cited.update(k.strip() for k in m.group(1).split(","))
cited &= set(B)
census_keys = {r[1] for r in CENSUS}
def bib_keys(k):
    f = B[k][1]; out = []
    if f.get("doi"): out.append(("doi", f["doi"].lower().strip()))
    for src in (f.get("eprint", ""), f.get("journal", ""), f.get("note", ""), f.get("howpublished", ""), f.get("url", "")):
        m = re.search(r"(\d{4}\.\d{4,5})", src)
        if m: out.append(("ax", m.group(1)))
    out.append(("t", norm_title(f.get("title", ""))))
    return out
bib_index = {}
for k in cited:
    for kk in bib_keys(k): bib_index.setdefault(kk, k)

def match(r):
    keys = [("doi", r["doi"]) if r["doi"] else None, ("ax", arxiv_id(r)), ("t", norm_title(r["title"]))]
    for k in keys:
        if k and k[1] and k in bib_index: return bib_index[k]
    return None

reasons = collections.Counter(); out = []
for r in uniq:
    d, why = decide(r)
    r["decision"], r["reason"], r["bibkey"] = d, why, match(r)
    r["year"] = r.get("year") or 0
    if d == "exclude": reasons[why] += 1
    out.append(r)

EDGE = RX(r"(edge|mobile|embedded|on-device|low-power|low power|energy-efficient|energy efficient|smartphone|wearable|milliwatt|\bmW\b|IoT|battery)")
DESIGN = RX(r"(accelerator|processor|FPGA|ASIC|\bchip\b|in-memory|\bCIM\b|\bPIM\b|\bNPU\b|macro|silicon|\bnm\b)")
cand = [r for r in out if r["decision"] == "pass" and not r["bibkey"]
        and EDGE.search((r["title"] or "") + " " + (r.get("abstract") or "")) and DESIGN.search(r["title"] or "")
        and WORK.search(r["title"] or "")]

found_cited = {r["bibkey"] for r in out if r["bibkey"]}
counts = {
    "identified": dict(identified), "identified_total": sum(identified.values()),
    "duplicates_removed": len(recs) - len(uniq), "unique": len(uniq),
    "excluded_by_rule": dict(reasons), "excluded_total": sum(reasons.values()),
    "passed_screening": sum(r["decision"] != "exclude" for r in out),
    "cited_corpus": len(cited), "cited_found_by_search": len(found_cited),
    "cited_found_passed": len({r["bibkey"] for r in out if r["bibkey"] and r["decision"] != "exclude"}),
    "census": len(census_keys), "census_found_by_search": len(found_cited & census_keys),
    "census_cited_post2020": None,
    "candidates_for_assessment": len(cand),
}
post = [k for k in cited if int(re.sub(r"\D", "", B[k][1].get("year", "0"))[:4] or 0) >= 2020]
counts["cited_2020plus"] = len(post)
counts["cited_2020plus_found"] = len(found_cited & set(post))
json.dump(counts, open(os.path.join(H, "prisma_counts.json"), "w"), indent=1)
with open(os.path.join(H, "screened.jsonl"), "w", encoding="utf-8") as f:
    for r in out:
        r = dict(r, queries=sorted(r["queries"])); f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(os.path.join(H, "candidates.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["id", "year", "venue", "title"])
    for r in sorted(cand, key=lambda r: (-r["year"], r["title"])):
        w.writerow([r["doi"] or r["id"], r["year"], r.get("venue") or "", r["title"]])
print(json.dumps(counts, indent=1))
print("census not found:", sorted(census_keys - found_cited))
