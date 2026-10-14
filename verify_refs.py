#!/usr/bin/env python3
"""verify_refs.py -- checks every reference cited in the compiled manuscript against an external
record: the DOI (via OpenAlex), the arXiv identifier (via the arXiv API), or, failing both, an
OpenAlex title search.  Writes ref_check.csv with a title-similarity score per entry."""
import re, json, time, csv, difflib, urllib.request, urllib.parse
import bibtools

B = bibtools.load()
aux = open("manuscript.aux", encoding="utf-8").read()
cited = sorted({k.strip() for m in re.findall(r"\\citation\{([^}]*)\}", aux) for k in m.split(",")} & set(B))

def norm(t):
    t = re.sub(r"\\[a-zA-Z]+|[{}$\\^_]", " ", t or "").lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()

def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()

def get(url):
    for i in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ref-check/1.0"}), timeout=40).read()
        except Exception as e:
            err = e; time.sleep(3 * (i + 1))
    raise err

def arxiv_id(f):
    for k in ("eprint", "journal", "note", "howpublished", "url"):
        m = re.search(r"(\d{4}\.\d{4,5})", f.get(k, ""))
        if m: return m.group(1)
    return None

rows = []
ax_needed = {}
for k in cited:
    typ, f = B[k]; title = f.get("title", ""); year = f.get("year", "")
    doi = f.get("doi", "").strip()
    rec = dict(key=k, bib_title=title, bib_year=year, method="", found_title="", found_year="", score=0.0)
    if doi:
        try:
            w = json.loads(get("https://api.openalex.org/works/doi:" + urllib.parse.quote(doi) + "?select=title,publication_year"))
            rec.update(method="doi", found_title=w.get("title") or "", found_year=w.get("publication_year"))
        except Exception as e:
            rec.update(method="doi-notfound")
        time.sleep(0.15)
    if not rec["found_title"] and arxiv_id(f):
        ax_needed[k] = arxiv_id(f); rec["method"] = "arxiv"
    rows.append(rec)

# arXiv batch
ids = list(ax_needed.items())
for i in range(0, len(ids), 40):
    chunk = ids[i:i + 40]
    x = get("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"id_list": ",".join(a for _, a in chunk), "max_results": 50})).decode("utf-8")
    found = {}
    for e in x.split("<entry>")[1:]:
        aid = re.search(r"<id>http://arxiv.org/abs/([^<v]+)", e).group(1)
        t = " ".join(re.search(r"<title>(.*?)</title>", e, re.S).group(1).split())
        y = re.search(r"<published>(\d{4})", e).group(1)
        found[aid] = (t, y)
    for k, a in chunk:
        r = next(r for r in rows if r["key"] == k)
        if a in found: r["found_title"], r["found_year"] = found[a]
        else: r["method"] = "arxiv-notfound"
    time.sleep(3.5)

# title search for the rest
for r in rows:
    if r["found_title"]: continue
    q = norm(r["bib_title"])[:200]
    if not q: continue
    try:
        d = json.loads(get("https://api.openalex.org/works?" + urllib.parse.urlencode(
            {"search": q, "per-page": 3, "select": "title,publication_year,doi"})))
        best = max(d["results"], key=lambda w: sim(r["bib_title"], w.get("title") or ""), default=None)
        if best: r.update(method=(r["method"] + "+title").strip("+"), found_title=best.get("title") or "", found_year=best.get("publication_year"))
    except Exception:
        pass
    time.sleep(0.2)

for r in rows:
    r["score"] = round(sim(r["bib_title"], r["found_title"]), 3) if r["found_title"] else 0.0
with open("ref_check.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
bad = [r for r in rows if r["score"] < 0.9]
print(f"cited {len(rows)}; matched >=0.9: {len(rows) - len(bad)}; to inspect: {len(bad)}")
for r in bad:
    print(f'{r["key"]:28s} {r["method"]:14s} {r["score"]:.2f} | {r["bib_title"][:70]} || {str(r["found_title"])[:70]} ({r["found_year"]})')
