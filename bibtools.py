"""bibtools.py -- minimal BibTeX reader used to label preprints (no external dependency)."""
import re

def load(path="edge_llm_refs.bib"):
    txt = open(path, encoding="utf-8").read()
    out = {}
    for chunk in re.split(r"\n(?=@)", "\n" + txt):
        m = re.match(r"\s*@(\w+)\s*\{\s*([^,\s]+)\s*,(.*)", chunk, re.S)
        if not m:
            continue
        typ, key, body = m.group(1).lower(), m.group(2), m.group(3)
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*", body):
            name, i, depth, j = fm.group(1).lower(), fm.end(), 0, fm.end()
            if i < len(body) and body[i] == "{":
                for j in range(i, len(body)):
                    depth += body[j] == "{"
                    depth -= body[j] == "}"
                    if depth == 0:
                        break
                fields.setdefault(name, body[i + 1:j])
            else:
                v = re.match(r'"([^"]*)"|([^,}\n]+)', body[i:])
                if v:
                    fields.setdefault(name, (v.group(1) or v.group(2) or "").strip())
        out[key] = (typ, fields)
    return out

def is_preprint(entry):
    """A work counts as a preprint when its only venue is arXiv (or another preprint server)."""
    typ, f = entry
    venue = " ".join(f.get(k, "") for k in ("journal", "booktitle")).strip()
    pre = ("arxiv" in (venue + f.get("archiveprefix", "") + f.get("eprint", "") +
                       f.get("howpublished", "") + f.get("note", "")).lower()
           or "openreview" in f.get("howpublished", "").lower())
    peer = venue and "arxiv" not in venue.lower()
    return bool(pre and not peer)
