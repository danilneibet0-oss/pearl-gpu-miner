"""
selftest.py — быстрая самопроверка сборки (в том числе Miner.exe без Python): GPU, NVRTC, ядра, BLAKE3 и ядро
джекпота сравниваются с CPU-эталоном побитово. Запуск: Miner.exe --selftest
"""

import time


def run() -> bool:
    import numpy as np
    import cupy as cp
    from blake3 import blake3

    import pearl_gpu
    import pearl_pipeline as pp
    import pearlhash_ref as ref

    t0 = time.time()
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"GPU: {props['name'].decode()}, SM: {props['multiProcessorCount']}, "
          f"compute capability {cp.cuda.Device().compute_capability}")
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'FAIL'}] {name}")

    rng = np.random.default_rng(1)
    key = rng.bytes(32)
    data = rng.integers(0, 256, size=5 * 1024 + 64, dtype=np.uint8)
    check("BLAKE3 матрицы (GPU) = эталон",
          pp.hash_matrix(cp.asarray(data), key) == blake3(ref.pad_to_chunk(data.tobytes()), key=key).digest())

    cfg = ref.default_config(2048)
    rt, ct = cfg.rows_pattern.partition(128), cfg.cols_pattern.partition(128)
    a = rng.integers(-127, 128, size=(128, 2048), dtype=np.int8)
    bt = rng.integers(-127, 128, size=(128, 2048), dtype=np.int8)
    want = ref.jackpots_all_tiles(a.astype(np.int32), bt.astype(np.int32), 128, rt, ct)
    check("ядро джекпота v2 (INT8 тензорные ядра) = эталон",
          np.array_equal(pearl_gpu.jackpots(a, bt, 128, rt, ct, variant="v2"), want))
    check("ядро джекпота v3 (рабочее для rank 128) = эталон",
          np.array_equal(pearl_gpu.jackpots(a, bt, 128, rt, ct, variant="v3"), want))

    header = ref.IncompleteBlockHeader(1, bytes(32), bytes(range(32)), 1_700_000_000, 0x1E00FFFF)
    cfg2 = ref.default_config(2048)
    seed = 3
    a2 = rng.integers(-64, 65, size=(256, 2048), dtype=np.int8)
    b2 = rng.integers(-64, 65, size=(2048, 256), dtype=np.int8)
    for cert in (2, 3):
        want_sol = ref.try_mine_one(np.random.default_rng(seed), 256, 256, 2048, header, cfg2, cert)
        r = np.random.default_rng(seed)
        a_ = r.integers(-64, 65, size=(256, 2048), dtype=np.int8)
        b_ = r.integers(-64, 65, size=(2048, 256), dtype=np.int8)
        got = pp.mine_attempt(cp.asarray(a_), cp.asarray(np.ascontiguousarray(b_.T)), header, cfg2, cert)
        same = (want_sol is None) == (got is None) and (want_sol is None or want_sol.jackpot_hash == got.jackpot_hash)
        check(f"полная попытка (cert {cert}) = эталон", same)

    try:
        import pearl_mining as pm
        import pearl_proof as pr
        sol = None
        for s in range(100):
            sol = pp.mine_random(256, 256, 2048, header, cfg2, 3, seed=s)
            if sol:
                break
        proof = pr.build_plain_proof(sol, header, cfg2)
        good, msg = pr.verify_before_submit(3, header, proof)
        check(f"официальный верификатор принимает PlainProof ({msg})", good)
    except Exception as e:      # noqa: BLE001
        check(f"официальный верификатор: {type(e).__name__}: {e}", False)

    print(f"Самопроверка {'пройдена' if ok else 'НЕ ПРОЙДЕНА'} за {time.time() - t0:.1f} с")
    return ok
