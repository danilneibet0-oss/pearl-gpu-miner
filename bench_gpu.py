"""
bench_gpu.py — бенчмарк ядра джекпота и разбор по стадиям конвейера.

  .venv\\Scripts\\python bench_gpu.py            # ядро + стадии
  .venv\\Scripts\\python bench_gpu.py --kernel   # только ядро

Единицы: MAC = одно int8-умножение-сложение внутри хеш-тайлов
(как miner_base.noisy_gemm.inner_hash_per_matmul: h*w*k для каждого тайла).
ops = 2*MAC (умножение + сложение, «TOPS»). Какую из них рынок называет «хешем» —
гипотеза, см. README; поэтому печатаем обе.
"""

from __future__ import annotations

import argparse
import time

import cupy as cp
import numpy as np

import pearl_gpu
import pearlhash_ref as ref

PEAK_INT8_DENSE_TOPS = 129.6      # паспортная плотная int8 3060 Ti (оценка по спецификации, не измерение)


def bench_kernel(m, n, k, rank=128, iters=20, variant="auto"):
    cfg = ref.default_config(k, rank)
    rt, ct = cfg.rows_pattern.partition(m), cfg.cols_pattern.partition(n)
    rng = np.random.default_rng(0)
    a = cp.asarray(rng.integers(-127, 128, size=(m, k), dtype=np.int8))
    bt = cp.asarray(rng.integers(-127, 128, size=(n, k), dtype=np.int8))
    ridx = cp.asarray(rt.reshape(-1).astype(np.int32))
    cidx = cp.asarray(ct.reshape(-1).astype(np.int32))
    out = cp.empty((rt.shape[0], ct.shape[0], 16), dtype=cp.uint32)
    def kern(*_):
        pearl_gpu.jackpots_dev(a, bt, rank, rt, ct, variant)
    grid, args = None, None
    kern(grid, (32,), args)
    cp.cuda.Device().synchronize()
    best = float("inf")
    for _ in range(7):                                 # лучшее из 7 серий: GPU делится с рабочим столом (WDDM)
        s, e = cp.cuda.Event(), cp.cuda.Event()
        s.record()
        for _ in range(iters):
            kern(grid, (32,), args)
        e.record()
        e.synchronize()
        best = min(best, cp.cuda.get_elapsed_time(s, e) / 1e3 / iters)
    sec = best
    macs = m * n * (k - k % rank)
    return sec, macs / sec


def stage_times(m, n, k, cert=3, nbits=0x1B00FFFF):
    """Время стадий одной попытки (решение не ищется — идёт полный проход по тайлам)."""
    cfg = ref.default_config(k)
    header = ref.IncompleteBlockHeader(1, bytes(32), bytes(range(32)), 1_700_000_000, nbits)
    t = {}
    rng = np.random.default_rng(1)

    def tick(name, t0):
        t[name] = time.perf_counter() - t0
        return time.perf_counter()

    t0 = time.perf_counter()
    a = rng.integers(-64, 65, size=(m, k), dtype=np.int8)
    b = rng.integers(-64, 65, size=(k, n), dtype=np.int8)
    bt = np.ascontiguousarray(b.T)
    t0 = tick("1 случайные A,B (CPU numpy)", t0)
    jkey = ref.job_key(header, cfg)
    ha = ref.b3(ref.pad_to_chunk(a.tobytes()), key=jkey)
    hb = ref.b3(ref.pad_to_chunk(bt.tobytes()), key=jkey)
    t0 = tick("2 BLAKE3 матриц (CPU)", t0)
    b_seed, a_seed = ref.noise_seeds(jkey, ha, hb, m, n, cert)
    na, nbt = ref.compute_noise(k, cfg.rank, b_seed, a_seed, np.arange(m), np.arange(n))
    t0 = tick("3 шум (CPU, python+blake3)", t0)
    an = (a.astype(np.int32) + na).astype(np.int8)
    btn = (bt.astype(np.int32) + nbt).astype(np.int8)
    t0 = tick("4 сложение с шумом (CPU)", t0)
    rt, ct = cfg.rows_pattern.partition(m), cfg.cols_pattern.partition(n)
    da, db = cp.asarray(an), cp.asarray(btn)
    cp.cuda.Device().synchronize()
    t0 = tick("5 копия на GPU", t0)
    jp = pearl_gpu.jackpots(da, db, cfg.rank, rt, ct)
    t0 = tick("6 ядро джекпота + возврат", t0)
    for ti in range(rt.shape[0]):
        for tj in range(ct.shape[0]):
            ref.jackpot_hash(jp[ti, tj], a_seed)
    tick("7 blake3 джекпотов по тайлам (CPU)", t0)
    return t, rt.shape[0] * ct.shape[0]


def gpu_stage_times(m, n, k, cert=3, nbits=0x1B00FFFF, reps=5):
    """Время стадий GPU-конвейера (с синхронизацией после каждой стадии), среднее по reps."""
    import pearl_pipeline as pp
    cfg = ref.default_config(k)
    header = ref.IncompleteBlockHeader(1, bytes(32), bytes(range(32)), 1_700_000_000, nbits)
    rank = cfg.rank
    acc = {}

    def sync():
        cp.cuda.Device().synchronize()

    for rep in range(reps + 1):
        t = {}
        t0 = time.perf_counter()

        def tick(name):
            nonlocal t0
            sync()
            t[name] = time.perf_counter() - t0
            t0 = time.perf_counter()

        a = pp.gen_signal((m, k), rep * 2); bt = pp.gen_signal((n, k), rep * 2 + 1); tick("1 случайные A,B^T")
        jkey = ref.job_key(header, cfg)
        jw = pp._words(jkey); haw, hbw = pp.hash_matrix_dev(a, jw), pp.hash_matrix_dev(bt, jw); tick("2 BLAKE3 матриц"); ha, hb = pp._bytes(haw), pp._bytes(hbw)
        b_seed, a_seed = ref.noise_seeds(jkey, ha, hb, m, n, cert); tick("3 сиды (GPU)")
        an = pp.noised(a, a_seed, ref.SEED_LABEL_A, rank); btn = pp.noised(bt, b_seed, ref.SEED_LABEL_B, rank)
        tick("4 шум + сложение")
        rt, ct = pp.tiles_for(cfg, m, n)
        jp = pearl_gpu.jackpots_dev(an, btn, rank, rt, ct); tick("5 ядро джекпота")
        pp.search_jackpots(jp, a_seed, ref.difficulty_bound(nbits, cfg)); tick("6 blake3 джекпотов + поиск")
        if rep:                                   # первый прогон — прогрев/компиляция
            for key_, v in t.items():
                acc[key_] = acc.get(key_, 0) + v / reps
    return acc, rt.shape[0] * ct.shape[0]


def throughput(m, n, k, depth=3, attempts=60, reps=4):
    """Устойчивая пропускная способность: попытки в нескольких CUDA-потоках, лучшее из reps прогонов."""
    import pearl_pipeline as pp
    cfg = ref.default_config(k)
    header = ref.IncompleteBlockHeader(1, bytes(32), bytes(range(32)), 1_700_000_000, 0x1B00FFFF)

    def run():
        t = time.perf_counter()
        for _ in pp.mine_random_pipelined(m, n, k, header, cfg, 3, range(attempts), depth=depth):
            pass
        cp.cuda.Device().synchronize()
        return (time.perf_counter() - t) / attempts

    run()
    return min(run() for _ in range(reps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", action="store_true", help="только ядро")
    args = ap.parse_args()

    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"GPU: {props['name'].decode()}, SM: {props['multiProcessorCount']}")
    print(f"\n== Ядро джекпота (чистое время ядра, cudaEvent) ==")
    print(f"{'m':>6} {'n':>6} {'k':>6} {'мс':>8} {'TMAC/s':>8} {'TOPS':>8} {'% пика':>7}")
    for m, n, k in [(1024, 1024, 2048), (2048, 2048, 2048), (4096, 4096, 2048),
                    (4096, 4096, 4096), (8192, 4096, 2048)]:        # протокол: k >= 16*rank = 2048
        sec, macs = bench_kernel(m, n, k)
        tops = 2 * macs / 1e12
        print(f"{m:6d} {n:6d} {k:6d} {sec*1e3:8.2f} {macs/1e12:8.2f} {tops:8.2f} {100*tops/PEAK_INT8_DENSE_TOPS:6.1f}%")

    if args.kernel:
        return
    for m, n, k in [(256, 256, 2048), (1024, 1024, 2048)]:
        t, tiles = stage_times(m, n, k)
        total = sum(t.values())
        print(f"\n== Стадии одной попытки m={m} n={n} k={k} ({tiles} тайлов), всего {total:.2f} c ==")
        for name, v in t.items():
            print(f"  {name:<42} {v*1e3:9.1f} мс  {100*v/total:5.1f}%")
        macs = m * n * k
        print(f"  эффективно по всему конвейеру: {macs/total/1e9:.2f} GMAC/s = {2*macs/total/1e9:.2f} GOPS")

    print("\n##### GPU-конвейер (всё на GPU) #####")
    for m, n, k in [(256, 256, 2048), (1024, 1024, 2048), (2048, 2048, 2048), (4096, 4096, 2048)]:
        t, tiles = gpu_stage_times(m, n, k)
        total = sum(t.values())
        print(f"\n== GPU-стадии m={m} n={n} k={k} ({tiles} тайлов), всего {total*1e3:.2f} мс ==")
        for name, v in t.items():
            print(f"  {name:<34} {v*1e3:9.3f} мс  {100*v/total:5.1f}%")
        macs = m * n * k
        print(f"  попыток/с: {1/total:8.1f}   эффективно: {macs/total/1e12:.3f} TMAC/s = {2*macs/total/1e12:.3f} TOPS")


def _tail():
    print("\n##### Пропускная способность (3 попытки в разных CUDA-потоках, лучшее из 4 прогонов) #####")
    for m, n, k in [(2048, 2048, 2048), (4096, 4096, 2048), (4096, 4096, 4096)]:
        t = throughput(m, n, k)
        macs = m * n * k
        print(f"  {m}x{n}x{k}: {t*1e3:6.3f} мс/попытку, {1/t:6.0f} попыток/с, "
              f"{macs/t/1e12:5.1f} TMAC/s = {2*macs/t/1e12:5.1f} TOPS эффективно")


if __name__ == "__main__":
    main()
    if True:
        _tail()
