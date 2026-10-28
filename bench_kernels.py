r"""
bench_kernels.py — сравнение вариантов ядра джекпота (v2, v3, ...) на одной и той же карте «вперемешку»,
чтобы фоновая нагрузка GPU одинаково влияла на все варианты. Печатает лучшее из серий.

  .venv\Scripts\python bench_kernels.py [m n k]
"""
import sys

import cupy as cp
import numpy as np

import pearl_gpu
import pearlhash_ref as ref


def bench(variants, m=4096, n=4096, k=2048, rank=128, reps=9, iters=20):
    cfg = ref.default_config(k, rank)
    rt, ct = cfg.rows_pattern.partition(m), cfg.cols_pattern.partition(n)
    rng = np.random.default_rng(0)
    a = cp.asarray(rng.integers(-127, 128, size=(m, k), dtype=np.int8))
    bt = cp.asarray(rng.integers(-127, 128, size=(n, k), dtype=np.int8))
    fns = {name: (lambda v=v: pearl_gpu.jackpots_dev(a, bt, rank, rt, ct, v)) for name, v in variants.items()}
    for f in fns.values():
        f()
    cp.cuda.Device().synchronize()
    best = {name: float("inf") for name in fns}
    for _ in range(reps):
        for name, f in fns.items():
            s, e = cp.cuda.Event(), cp.cuda.Event()
            s.record()
            for _ in range(iters):
                f()
            e.record()
            e.synchronize()
            best[name] = min(best[name], cp.cuda.get_elapsed_time(s, e) / iters)
    macs = m * n * (k - k % rank)
    return {name: (t, macs / t / 1e9) for name, t in best.items()}


if __name__ == "__main__":
    dims = [int(x) for x in sys.argv[1:4]] or [4096, 4096, 2048]
    res = bench({"v2": "v2"}, *dims)
    for name, (t, tm) in res.items():
        print(f"{name:10s} {t:.3f} ms  {tm:.1f} TMAC/s")
