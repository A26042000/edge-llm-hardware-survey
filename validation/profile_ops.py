#!/usr/bin/env python3
"""
profile_ops.py -- measured per-operator anatomy of batch-1 decode for Llama-3.2-1B
(eager PyTorch, Hugging Face transformers 4.46.3) on the CPU and the GPU of the test laptop.

Every decode step is timed with exclusive (non-nested) regions:
  GEMV       : all nn.Linear layers (Q/K/V/O, gate/up/down, LM head)
  Attn+KV    : remaining attention work (QK^T, AV, KV-cache append, reshapes)
  Softmax    : attention softmax
  Norm       : RMSNorm layers
  RoPE       : cos/sin generation + apply_rotary_pos_emb
  Act        : SiLU and the gate multiply of the SwiGLU FFN
  Sampling   : vocabulary softmax + argmax
  Other      : embedding lookup, residual adds, framework bookkeeping
The KV cache is pre-filled with random tensors of the requested length (timing does
not depend on the cache contents). GPU regions are timed with CUDA events (no host
synchronisation inside the step); CPU regions with perf_counter.
Usage:  python profile_ops.py cpu|cuda  > result.json
"""
import sys, os, json, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pylib"))
import torch
import torch.nn as nn
import transformers
from transformers.models.llama import modeling_llama as ML
from transformers import LlamaForCausalLM, DynamicCache

DEV = sys.argv[1] if len(sys.argv) > 1 else "cpu"
DTYPE = torch.float16 if DEV == "cuda" else torch.float32
CTX = [512, 2048, 8192] if DEV == "cuda" else [512, 2048, 4096]
STEPS, WARM = 24, 6
torch.set_num_threads(8)
HERE = os.path.dirname(os.path.abspath(__file__))
model = LlamaForCausalLM.from_pretrained(os.path.join(HERE, "hf_l1b"), torch_dtype=DTYPE,
                                         attn_implementation="eager").to(DEV).eval()
cfg = model.config

# ------------------------------------------------------------------ region timer
class Timer:
    def __init__(self):
        self.stack, self.records = [], []
    def now(self):
        if DEV == "cuda":
            e = torch.cuda.Event(enable_timing=True); e.record(); return e
        return time.perf_counter()
    def enter(self, cat):
        self.stack.append([cat, self.now(), []])
    def exit(self):
        cat, t0, kids = self.stack.pop()
        t1 = self.now()
        node = (cat, t0, t1, kids)
        if self.stack: self.stack[-1][2].append(node)
        else: self.records.append(node)
    def resolve(self):
        out = {}
        def el(a, b):
            return a.elapsed_time(b) / 1e3 if DEV == "cuda" else (b - a)
        def walk(n):
            cat, t0, t1, kids = n
            incl = el(t0, t1)
            child = 0.0
            for k in kids: child += walk(k)
            out[cat] = out.get(cat, 0.0) + max(incl - child, 0.0)
            return incl
        for r in self.records: walk(r)
        self.records = []
        return out
T = Timer()

def cat_of(m):
    if isinstance(m, nn.Linear): return "GEMV"
    if isinstance(m, ML.LlamaRMSNorm): return "Norm"
    if isinstance(m, ML.LlamaRotaryEmbedding): return "RoPE"
    if isinstance(m, ML.LlamaMLP): return "Act"          # exclusive part = gate*up multiply
    if isinstance(m, ML.LlamaAttention): return "Attn+KV"  # exclusive part = QK, AV, KV append
    if isinstance(m, (nn.SiLU,)) or type(m).__name__ in ("SiLUActivation",): return "Act"
    if isinstance(m, (ML.LlamaDecoderLayer, ML.LlamaModel, nn.Embedding)): return "Other"
    return None
NOHOOK = len(sys.argv) > 2 and sys.argv[2] == "nohooks"
for m in ([] if NOHOOK else model.modules()):
    c = cat_of(m)
    if c:
        m.register_forward_pre_hook(lambda mod, inp, c=c: T.enter(c))
        m.register_forward_hook(lambda mod, inp, out: T.exit())

_rope = ML.apply_rotary_pos_emb
def rope_wrapped(*a, **k):
    T.enter("RoPE"); r = _rope(*a, **k); T.exit(); return r
if not NOHOOK: ML.apply_rotary_pos_emb = rope_wrapped
_sm = nn.functional.softmax
def sm_wrapped(*a, **k):
    T.enter("Softmax"); r = _sm(*a, **k); T.exit(); return r
if not NOHOOK: nn.functional.softmax = sm_wrapped

def make_cache(n):
    c = DynamicCache()
    hd = cfg.hidden_size // cfg.num_attention_heads
    for l in range(cfg.num_hidden_layers):
        k = torch.randn(1, cfg.num_key_value_heads, n, hd, dtype=DTYPE, device=DEV)
        c.update(k, k.clone(), l)
    return c

def step(cache, pos, tok):
    T.enter("Other")
    out = model(input_ids=tok, past_key_values=cache, position_ids=torch.tensor([[pos]], device=DEV),
                use_cache=True, return_dict=True)
    T.exit()
    T.enter("Sampling")
    p = torch.softmax(out.logits[:, -1, :].float(), dim=-1)
    nxt = torch.argmax(p, dim=-1, keepdim=True)
    T.exit()
    return nxt

results = {"device": DEV, "dtype": str(DTYPE), "torch": torch.__version__,
           "transformers": transformers.__version__,
           "name": torch.cuda.get_device_name(0) if DEV == "cuda" else "AMD Ryzen 7 5800HS (8C/16T)",
           "runs": []}
with torch.inference_mode():
    for n in CTX:
        cache = make_cache(n)
        tok = torch.tensor([[1000]], device=DEV)
        shares, lat = [], []
        for s in range(WARM + STEPS):
            if DEV == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            tok = step(cache, n + s, tok)
            if DEV == "cuda": torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            parts = T.resolve()
            if s >= WARM:
                lat.append(dt); shares.append(parts)
        cats = sorted({k for p in shares for k in p})
        med = {c: statistics.median([p.get(c, 0.0) for p in shares]) for c in cats}
        tot = sum(med.values())
        results["runs"].append({"context": n, "latency_ms_median": 1e3 * statistics.median(lat),
                                "share_pct": {c: round(100 * v / tot, 2) for c, v in med.items()},
                                "time_ms": {c: round(1e3 * v, 3) for c, v in med.items()}})
        del cache
print(json.dumps(results, indent=1))
