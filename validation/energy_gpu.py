#!/usr/bin/env python3
"""
energy_gpu.py -- measured energy per generated token on the RTX 3050 (laptop variant).
Runs llama-bench (decode only, n_gen = 256, 3 repetitions) for each precision and context
depth while sampling board power with nvidia-smi every 100 ms; reports mean power during the
run, idle power before it, and J/token = P_mean / tok_s (total) and (P_mean - P_idle) / tok_s
(incremental).  Output: energy_gpu.json
"""
import subprocess, json, time, threading, statistics, os
H = os.path.dirname(os.path.abspath(__file__))
def sample(stop, out):
    while not stop.is_set():
        p = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        try: out.append(float(p.stdout.strip()))
        except ValueError: pass
        time.sleep(0.1)
def idle(sec=5):
    v = []
    for _ in range(int(sec / 0.1)):
        p = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        v.append(float(p.stdout.strip())); time.sleep(0.1)
    return statistics.mean(v)
res = []
for q in ("Q4_0", "Q8_0", "f16"):
    for d in (0, 8192):
        time.sleep(8); p_idle = idle()
        samples, stop = [], threading.Event()
        t = threading.Thread(target=sample, args=(stop, samples)); t.start()
        t0 = time.time()
        out = subprocess.run([os.path.join(H, "bin_cuda", "llama-bench.exe"), "-m", os.path.join(H, "models", f"L1B-{q}.gguf"),
                              "-ngl", "99", "-p", "0", "-n", "256", "-d", str(d), "-r", "3", "-o", "json"],
                             capture_output=True, text=True, cwd=H)
        stop.set(); t.join(); dt = time.time() - t0
        r = json.loads(out.stdout)[0]
        # discard the first 15 % of samples (model load) and the last 5 %
        s = samples[int(0.15 * len(samples)):int(0.95 * len(samples))] or samples
        p_mean = statistics.mean(s); p_max = max(s)
        rec = dict(precision=q, depth=d, tok_s=r["avg_ts"], tok_s_std=r["stddev_ts"], p_idle_W=round(p_idle, 2),
                   p_mean_W=round(p_mean, 2), p_max_W=round(p_max, 2), n_samples=len(s), wall_s=round(dt, 1),
                   J_per_tok_total=round(p_mean / r["avg_ts"], 4), J_per_tok_incremental=round((p_mean - p_idle) / r["avg_ts"], 4))
        print(rec, flush=True); res.append(rec)
json.dump({"device": "NVIDIA GeForce RTX 3050 Laptop GPU", "model": "Llama-3.2-1B-Instruct", "runs": res},
          open(os.path.join(H, "energy_gpu.json"), "w"), indent=1)
