#!/usr/bin/env python3
"""membw.py -- streaming-bandwidth microbenchmark used for beta_CPU in analyze.py.
Result on the test laptop (Oct 2026): CPU copy 12.8, triad 15.7, read(sum) 28.7 GB/s;
GPU copy 163.5 GB/s. The read figure is the one that matches decode (weights are only read)."""
import torch, time
torch.set_num_threads(8)
n = 256 * 1024 * 1024  # 1 GiB fp32
a = torch.empty(n); b = torch.ones(n); c = torch.ones(n)

def bw(f, bytes_moved, reps=6, sync=lambda: None):
    f(); sync(); best = 1e9
    for _ in range(reps):
        t = time.perf_counter(); f(); sync(); best = min(best, time.perf_counter() - t)
    return bytes_moved / best / 1e9

copy = bw(lambda: a.copy_(b), 2 * 4 * n)
triad = bw(lambda: torch.add(b, c, alpha=2.0, out=a), 3 * 4 * n)
read = bw(lambda: b.sum(), 4 * n)
print(f"CPU copy {copy:.1f} GB/s  triad {triad:.1f} GB/s  read(sum) {read:.1f} GB/s")
if torch.cuda.is_available():
    m = 128 * 1024 * 1024
    x = torch.ones(m, device="cuda"); y = torch.empty(m, device="cuda")
    g = bw(lambda: y.copy_(x), 2 * 4 * m, sync=torch.cuda.synchronize)
    print(f"GPU copy {g:.1f} GB/s  ({torch.cuda.get_device_name(0)})")
