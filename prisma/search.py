#!/usr/bin/env python3
"""
search.py -- logged, re-runnable database search for the PRISMA 2020 flow.

Sources : OpenAlex (indexes IEEE, ACM, Elsevier, Springer, MDPI, arXiv, ...) and the arXiv API.
Window  : 2020-01-01 .. 2026-09-30
Queries : QA (edge LLM/Transformer hardware), QB (operator-level hardware), QC (surveys)
Output  : raw/<source>_<query>.jsonl (one record per line) and search_log.json
          (query strings, run date, hit counts).
"""
import json, os, time, urllib.parse, urllib.request, datetime, re
import xml.etree.ElementTree as ET

H = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(H, "raw"); os.makedirs(RAW, exist_ok=True)
FROM, TO = "2020-01-01", "2026-09-30"

WORK = '("large language model" OR "large language models" OR LLM OR LLMs OR "language model" OR "language models" OR "transformer model" OR "transformer models" OR "transformer inference" OR "transformer accelerator" OR "transformer-based" OR "self-attention" OR "vision transformer")'
HW = '(accelerator OR accelerators OR FPGA OR ASIC OR NPU OR "neural processing unit" OR "processing-in-memory" OR "processing in memory" OR "compute-in-memory" OR "computing-in-memory" OR "in-memory computing" OR "hardware architecture" OR "hardware design" OR "hardware-software co-design" OR "hardware/software co-design")'
EDGE = '(edge OR mobile OR embedded OR "on-device" OR "low-power" OR "low power" OR "energy-efficient" OR "energy efficient" OR smartphone OR wearable)'
OPS = '(softmax OR "layer normalization" OR layernorm OR RMSNorm OR "rotary position" OR "rotary positional" OR RoPE OR "KV cache" OR "key-value cache" OR "mixture of experts" OR "nonlinear function" OR "non-linear function" OR "nonlinear operation" OR GELU OR SiLU OR "speculative decoding")'
HW2 = '(accelerator OR FPGA OR ASIC OR hardware OR circuit OR "processing-in-memory" OR "compute-in-memory" OR "in-memory computing")'
QUERIES = {
    "QA": f"{WORK} AND {HW} AND {EDGE}",
    "QB": f"{OPS} AND {HW2} AND {WORK}",
    "QC": '("large language model" OR "large language models" OR LLM OR LLMs OR transformer OR transformers) AND (hardware OR accelerator OR accelerators OR accelerating OR acceleration) AND (survey OR review OR tutorial)',
}
# arXiv: same logic on abstracts, restricted to the hardware-relevant categories
AX = {
    "QA": '(abs:"large language model" OR abs:LLM OR abs:transformer) AND (abs:accelerator OR abs:FPGA OR abs:ASIC OR abs:NPU OR abs:"in-memory" OR abs:"processing-in-memory" OR abs:"compute-in-memory" OR abs:chip) AND (abs:edge OR abs:mobile OR abs:embedded OR abs:"on-device" OR abs:"low-power" OR abs:"energy-efficient")',
    "QB": '(abs:softmax OR abs:"layer normalization" OR abs:RMSNorm OR abs:"rotary position" OR abs:RoPE OR abs:"KV cache" OR abs:"mixture of experts" OR abs:"nonlinear" OR abs:GELU OR abs:"speculative decoding") AND (abs:accelerator OR abs:FPGA OR abs:ASIC OR abs:hardware OR abs:"in-memory") AND (abs:transformer OR abs:LLM OR abs:"language model")',
    "QC": '(ti:"large language model" OR ti:LLM OR ti:transformer) AND (ti:hardware OR ti:accelerator OR ti:accelerators OR ti:accelerating) AND (ti:survey OR ti:review OR ti:tutorial)',
}
CATS = "(cat:cs.AR OR cat:cs.DC OR cat:cs.ET OR cat:cs.LG OR cat:eess.SP OR cat:eess.SY OR cat:cs.PF)"

def get(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "survey-prisma/1.0"}), timeout=90) as r:
                return r.read()
        except Exception as e:
            print("  retry", i, e); time.sleep(5 * (i + 1))
    raise RuntimeError(url)

def inv_abstract(ii):
    if not ii: return ""
    pos = [(p, w) for w, ps in ii.items() for p in ps]
    return " ".join(w for _, w in sorted(pos))

def openalex(qid, q):
    out, cursor = [], "*"
    field = "title.search" if qid == "QC" else "title_and_abstract.search"   # surveys: title only
    filt = f"{field}:{q},from_publication_date:{FROM},to_publication_date:{TO}"
    while cursor:
        url = ("https://api.openalex.org/works?" + urllib.parse.urlencode(
            {"filter": filt, "per-page": 200, "cursor": cursor,
             "select": "id,doi,title,publication_year,publication_date,type,primary_location,abstract_inverted_index,ids"}))
        d = json.loads(get(url))
        for w in d["results"]:
            loc = w.get("primary_location") or {}
            src = (loc.get("source") or {}).get("display_name")
            out.append({"src": "openalex", "id": w["id"], "doi": (w.get("doi") or "").lower().replace("https://doi.org/", ""),
                        "title": w.get("title") or "", "year": w.get("publication_year"), "type": w.get("type"),
                        "venue": src, "abstract": inv_abstract(w.get("abstract_inverted_index"))})
        cursor = d["meta"].get("next_cursor")
        if not d["results"]: break
        time.sleep(0.2)
    return out, d["meta"]["count"] if out or d else 0

def arxiv(qid, q):
    out, start, total = [], 0, None
    ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
    query = f"({q}) AND {CATS} AND submittedDate:[202001010000 TO 202609302359]"
    while total is None or start < total:
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"search_query": query, "start": start, "max_results": 500, "sortBy": "submittedDate", "sortOrder": "ascending"})
        root = ET.fromstring(get(url))
        total = int(root.find("o:totalResults", ns).text)
        ents = root.findall("a:entry", ns)
        for e in ents:
            aid = e.find("a:id", ns).text.rsplit("/abs/", 1)[-1]
            doi = e.find("{http://arxiv.org/schemas/atom}doi")
            out.append({"src": "arxiv", "id": aid, "doi": (doi.text.lower() if doi is not None else ""),
                        "title": " ".join(e.find("a:title", ns).text.split()),
                        "year": int(e.find("a:published", ns).text[:4]), "type": "preprint",
                        "venue": "arXiv", "abstract": " ".join(e.find("a:summary", ns).text.split())})
        if not ents: break
        start += len(ents); time.sleep(3.5)
    return out, total

if __name__ == "__main__":
    log = {"run_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"), "window": [FROM, TO],
           "openalex_queries": QUERIES, "arxiv_queries": AX, "arxiv_categories": CATS, "counts": {}}
    for qid in QUERIES:
        for name, fn, q in (("openalex", openalex, QUERIES[qid]), ("arxiv", arxiv, AX[qid])):
            print(name, qid, "...", flush=True)
            recs, total = fn(qid, q)
            with open(os.path.join(RAW, f"{name}_{qid}.jsonl"), "w", encoding="utf-8") as f:
                for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            log["counts"][f"{name}_{qid}"] = {"reported": total, "retrieved": len(recs)}
            print("  ", total, len(recs), flush=True)
    json.dump(log, open(os.path.join(H, "search_log.json"), "w"), indent=1)
