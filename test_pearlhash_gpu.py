"""
Тесты корректности GPU-ядра джекпота: результат обязан побитово совпадать
с эталоном pearlhash_ref.jackpots_all_tiles (источник истины).

Запуск:  .venv\\Scripts\\python -m pytest test_pearlhash_gpu.py -q
"""

import numpy as np
import pytest

import pearlhash_ref as ref

cp = pytest.importorskip("cupy")
import pearl_gpu  # noqa: E402

RANK = 128


def _patterns(name):
    P = ref.PeriodicPattern.from_list
    return {
        "default": (P([0, 8, 64, 72]),
                    P([0, 1, 8, 9, 32, 33, 40, 41, 64, 65, 72, 73, 96, 97, 104, 105])),
        "contig_4x8": (P([0, 1, 2, 3]), P(list(range(8)))),
        "contig_8x8": (P(list(range(8))), P(list(range(8)))),
        "contig_16x8": (P(list(range(16))), P(list(range(8)))),
        "contig_4x24": (P([0, 1, 2, 3]), P(list(range(24)))),
    }[name]


def _cfg(k, pat="default"):
    rows, cols = _patterns(pat)
    return ref.MiningConfiguration(common_dim=k, rank=RANK, rows_pattern=rows, cols_pattern=cols)


# ---------------------------------------------------------------------------
# 1. Чистое ядро: произвольные int8 (включая крайние значения) против эталона
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pat", ["default", "contig_4x8", "contig_8x8", "contig_16x8", "contig_4x24"])
@pytest.mark.parametrize("k", [1024, 1088, 2048, 4096])   # 1088: хвост < rank игнорируется; 4096 > 16*rank
def test_kernel_matches_ref(k, pat):
    rng = np.random.default_rng(1234 + k)
    m, n = 256, 384   # 384 делится на периоды всех паттернов (128, 24, 8)
    a = rng.integers(-127, 128, size=(m, k), dtype=np.int8)
    bt = rng.integers(-127, 128, size=(n, k), dtype=np.int8)
    cfg = _cfg(k, pat)
    rt, ct = cfg.rows_pattern.partition(m), cfg.cols_pattern.partition(n)
    want = ref.jackpots_all_tiles(a.astype(np.int32), bt.astype(np.int32), RANK, rt, ct)
    got = pearl_gpu.jackpots(a, bt, RANK, rt, ct)
    assert got.dtype == np.uint32 and got.shape == want.shape
    np.testing.assert_array_equal(got, want)


def test_kernel_slot_reuse_many_blocks():
    """k = 40*rank: каждый из 16 слотов джекпота переиспользуется 2-3 раза."""
    k = 40 * RANK
    rng = np.random.default_rng(7)
    a = rng.integers(-127, 128, size=(128, k), dtype=np.int8)
    bt = rng.integers(-127, 128, size=(128, k), dtype=np.int8)
    cfg = _cfg(k)
    rt, ct = cfg.rows_pattern.partition(128), cfg.cols_pattern.partition(128)
    want = ref.jackpots_all_tiles(a.astype(np.int32), bt.astype(np.int32), RANK, rt, ct)
    np.testing.assert_array_equal(pearl_gpu.jackpots(a, bt, RANK, rt, ct), want)


def test_kernel_int32_wraparound():
    """Максимальные по модулю значения: накопитель переполняется по модулю 2^32 как i32."""
    k = 4096
    a = np.full((128, k), -128, dtype=np.int8)
    bt = np.full((128, k), -128, dtype=np.int8)      # 128*128*4096 = 2^26 — без переполнения
    a[:, ::2] = 127
    cfg = _cfg(k)
    rt, ct = cfg.rows_pattern.partition(128), cfg.cols_pattern.partition(128)
    want = ref.jackpots_all_tiles(a.astype(np.int32), bt.astype(np.int32), RANK, rt, ct)
    np.testing.assert_array_equal(pearl_gpu.jackpots(a, bt, RANK, rt, ct), want)


# ---------------------------------------------------------------------------
# 2. Весь конвейер (cert 2 и 3): те же rng/заголовок -> то же решение, что у эталона
# ---------------------------------------------------------------------------
def _header(nbits):
    return ref.IncompleteBlockHeader(1, bytes(range(32)), bytes(range(32, 64)), 1_700_000_000, nbits)


@pytest.mark.parametrize("cert", [2, 3])
@pytest.mark.parametrize("k", [1024, 2176])      # 2176 = 17*rank > 16*rank
def test_noised_jackpots_match_ref(cert, k):
    m = n = 256
    cfg = _cfg(k)
    rng = np.random.default_rng(99)
    a = rng.integers(-64, 65, size=(m, k), dtype=np.int8)
    b = rng.integers(-64, 65, size=(k, n), dtype=np.int8)
    bt = np.ascontiguousarray(b.T)
    header = _header(0x1C00FFFF)
    jkey = ref.job_key(header, cfg)
    hash_a = ref.b3(ref.pad_to_chunk(a.tobytes()), key=jkey)
    hash_b = ref.b3(ref.pad_to_chunk(bt.tobytes()), key=jkey)
    b_seed, a_seed = ref.noise_seeds(jkey, hash_a, hash_b, m, n, cert)
    na, nbt = ref.compute_noise(k, RANK, b_seed, a_seed, np.arange(m), np.arange(n))
    a_n, bt_n = a.astype(np.int32) + na, bt.astype(np.int32) + nbt
    assert a_n.min() >= -128 and a_n.max() <= 127 and bt_n.min() >= -128 and bt_n.max() <= 127
    rt, ct = cfg.rows_pattern.partition(m), cfg.cols_pattern.partition(n)
    want = ref.jackpots_all_tiles(a_n, bt_n, RANK, rt, ct)
    got = pearl_gpu.jackpots(a_n.astype(np.int8), bt_n.astype(np.int8), RANK, rt, ct)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("cert", [2, 3])
@pytest.mark.parametrize("nbits", [0x2100FFFF, 0x1E00FFFF, 0x1B00FFFF])   # любой тайл, редкие тайлы, ничего не найдено
def test_try_mine_one_gpu_equals_ref(cert, nbits):
    m = n = 256
    k = 1024
    cfg = _cfg(k)
    header = _header(nbits)
    bound = ref.difficulty_bound(nbits, cfg)
    for seed in range(3):
        want = ref.try_mine_one(np.random.default_rng(seed), m, n, k, header, cfg, cert, bound)
        got = pearl_gpu.try_mine_one(np.random.default_rng(seed), m, n, k, header, cfg, cert, bound)
        assert (want is None) == (got is None)
        if want is not None:
            assert got.a_rows == want.a_rows and got.b_cols == want.b_cols
            assert got.jackpot_hash == want.jackpot_hash
            np.testing.assert_array_equal(got.a, want.a)
            np.testing.assert_array_equal(got.b, want.b)


def test_cert2_and_cert3_differ():
    """Защита от тихого игнорирования cert_version: сиды V2 и V3 обязаны различаться."""
    cfg = _cfg(1024)
    h = _header(0x1C00FFFF)
    jk = ref.job_key(h, cfg)
    ha, hb = bytes(32), bytes(range(32))
    assert ref.noise_seeds(jk, ha, hb, 256, 256, 2) != ref.noise_seeds(jk, ha, hb, 256, 256, 3)


def test_mining_cases_are_not_vacuous():
    """Тест выше имеет смысл, только если решения реально находятся не только в первом тайле."""
    cfg = _cfg(1024)
    found = [pearl_gpu.try_mine_one(np.random.default_rng(s), 256, 256, 1024, _header(0x1E00FFFF), cfg, 3)
             for s in range(6)]
    assert all(f is not None for f in found)
    assert any((f.a_rows[0], f.b_cols[0]) != (0, 0) for f in found)
    assert pearl_gpu.try_mine_one(np.random.default_rng(0), 256, 256, 1024, _header(0x1B00FFFF), cfg, 3) is None


# ---------------------------------------------------------------------------
# Оптимизированное ядро v2 (shared memory + cp.async): отдельные проверки
# ---------------------------------------------------------------------------
def test_auto_dispatch_uses_v2_for_default_pattern():
    assert pearl_gpu.v2_supported(256, 384, 2048, 128, 4, 16)
    assert not pearl_gpu.v2_supported(200, 384, 2048, 128, 4, 16)       # m не кратно 128
    assert not pearl_gpu.v2_supported(256, 384, 2048, 128, 8, 16)       # другой тайл -> простое ядро


@pytest.mark.parametrize("m,n,k,rank", [
    (128, 128, 2048, 128),      # один блок
    (640, 384, 2048, 128),      # 5x3 блоков
    (256, 256, 2176, 128),      # 17 rank-блоков, слоты переиспользуются
    (256, 256, 2112, 128),      # хвост 64 < rank игнорируется
    (256, 256, 5120, 128),      # 40 rank-блоков
    (256, 256, 4096, 256),      # rank 256: 4 стадии на rank-блок
    (256, 256, 4096, 64),       # rank 64: 1 стадия на rank-блок
])
def test_v2_matches_ref_and_simple(m, n, k, rank):
    rng = np.random.default_rng(m + n + k + rank)
    a = rng.integers(-128, 128, size=(m, k), dtype=np.int8)           # включая -128
    bt = rng.integers(-128, 128, size=(n, k), dtype=np.int8)
    cfg = ref.MiningConfiguration(k, rank, _patterns("default")[0], _patterns("default")[1])
    rt, ct = cfg.rows_pattern.partition(m), cfg.cols_pattern.partition(n)
    want = ref.jackpots_all_tiles(a.astype(np.int32), bt.astype(np.int32), rank, rt, ct)
    got = pearl_gpu.jackpots(a, bt, rank, rt, ct, variant="v2")
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(pearl_gpu.jackpots(a, bt, rank, rt, ct, variant="simple"), want)


def test_v2_all_zero_and_constant_inputs():
    m = n = 256
    k = 2048
    cfg = _cfg(k)
    rt, ct = cfg.rows_pattern.partition(m), cfg.cols_pattern.partition(n)
    for val in (0, 1, -128, 127):
        a = np.full((m, k), val, dtype=np.int8)
        bt = np.full((n, k), val, dtype=np.int8)
        want = ref.jackpots_all_tiles(a.astype(np.int32), bt.astype(np.int32), RANK, rt, ct)
        np.testing.assert_array_equal(pearl_gpu.jackpots(a, bt, RANK, rt, ct, variant="v2"), want)
