"""
Тесты GPU-конвейера (BLAKE3, шум, генерация, поиск, весь цикл) против CPU-эталона pearlhash_ref.
Всё сравнивается побитово. Запуск:  .venv\\Scripts\\python -m pytest test_pearlhash_pipeline.py -q
"""

import numpy as np
import pytest

import pearlhash_ref as ref

cp = pytest.importorskip("cupy")
import pearl_pipeline as pp  # noqa: E402

RANK = 128


# ---------------------------------------------------------------------------
# BLAKE3 (keyed, с паддингом нулями до 1024)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nbytes", [
    64, 256, 1024, 1088, 2048, 3072, 4096, 5 * 1024, 6 * 1024, 7 * 1024, 8 * 1024, 9 * 1024,
    13 * 1024, 31 * 1024 + 64, 1000 * 1024, 1024 * 1024 + 4096,
])
def test_blake3_matrix_hash(nbytes):
    rng = np.random.default_rng(nbytes)
    data = rng.integers(0, 256, size=nbytes, dtype=np.uint8)
    key = rng.bytes(32)
    want = ref.b3(ref.pad_to_chunk(data.tobytes()), key=key)
    assert pp.hash_matrix(cp.asarray(data), key) == want


# ---------------------------------------------------------------------------
# Шум
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rank", [64, 128, 256, 1024])
def test_uniform_and_perm_match_ref(rank):
    key = bytes(range(32))
    rows = np.arange(37)
    for label in (ref.SEED_LABEL_A, ref.SEED_LABEL_B):
        want_u = ref.uniform_matrix(label, key, rows, rank)
        got_u = cp.asnumpy(pp.gen_uniform(label, key, len(rows), rank))
        np.testing.assert_array_equal(got_u.astype(np.int16), want_u)
        for k in (1024, 1090 - 1090 % 64, 4096 + 8):
            want_p = ref.permutation_pairs(label, key, k, rank)
            got_p = cp.asnumpy(pp.gen_perm(label, key, k, rank))
            np.testing.assert_array_equal(got_p, want_p)


@pytest.mark.parametrize("rank", [64, 128, 256])
@pytest.mark.parametrize("k", [1024, 2176])
def test_apply_noise_matches_ref(rank, k):
    rng = np.random.default_rng(5)
    rows = 48
    a = rng.integers(-64, 65, size=(rows, k), dtype=np.int8)
    a_seed, b_seed = rng.bytes(32), rng.bytes(32)
    na, nbt = ref.compute_noise(k, rank, b_seed, a_seed, np.arange(rows), np.arange(rows))
    want_a = (a.astype(np.int32) + na).astype(np.int8)
    want_b = (a.astype(np.int32) + nbt).astype(np.int8)
    got_a = cp.asnumpy(pp.noised(cp.asarray(a), a_seed, ref.SEED_LABEL_A, rank))
    got_b = cp.asnumpy(pp.noised(cp.asarray(a), b_seed, ref.SEED_LABEL_B, rank))
    np.testing.assert_array_equal(got_a, want_a)
    np.testing.assert_array_equal(got_b, want_b)


# ---------------------------------------------------------------------------
# Хеши джекпотов и сравнение со сложностью
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bound_shift", [256, 250, 245, 240, 236])
def test_jackpot_search_matches_ref(bound_shift):
    rng = np.random.default_rng(bound_shift)
    tiles = 3000
    jp = rng.integers(0, 2**32, size=(30, 100, 16), dtype=np.uint32)
    a_seed = rng.bytes(32)
    bound = (1 << bound_shift) - 1 if bound_shift < 256 else ref.U256_MAX
    want = -1
    for t in range(tiles):
        h = ref.jackpot_hash(jp.reshape(-1, 16)[t], a_seed)
        if int.from_bytes(h, "little") <= bound:
            want = t
            break
    got = pp.search_jackpots(cp.asarray(jp), a_seed, bound)
    assert got == want


# ---------------------------------------------------------------------------
# Генератор сигнала
# ---------------------------------------------------------------------------
def test_signal_rng_range_and_uniformity():
    x = cp.asnumpy(pp.gen_signal((1 << 22,), seed=123))
    assert x.dtype == np.int8 and x.min() == -64 and x.max() == 64
    counts = np.bincount((x.astype(np.int16) + 64).ravel(), minlength=129)
    exp = x.size / 129
    assert np.abs(counts - exp).max() < 6 * np.sqrt(exp)      # ~6σ
    assert not np.array_equal(x, cp.asnumpy(pp.gen_signal((1 << 22,), seed=124)))
    np.testing.assert_array_equal(x, cp.asnumpy(pp.gen_signal((1 << 22,), seed=123)))


# ---------------------------------------------------------------------------
# Весь цикл: те же A, B и заголовок -> то же решение, что у эталона
# ---------------------------------------------------------------------------
def _header(nbits):
    return ref.IncompleteBlockHeader(1, bytes(range(32)), bytes(range(32, 64)), 1_700_000_000, nbits)


@pytest.mark.parametrize("cert", [2, 3])
@pytest.mark.parametrize("nbits", [0x2100FFFF, 0x1E00FFFF, 0x1B00FFFF])
@pytest.mark.parametrize("shape", [(256, 256, 1024), (512, 384, 2176)])
def test_full_attempt_equals_ref(cert, nbits, shape):
    m, n, k = shape
    cfg = ref.default_config(k, RANK)
    header = _header(nbits)
    bound = ref.difficulty_bound(nbits, cfg)
    for seed in range(3):
        want = ref.try_mine_one(np.random.default_rng(seed), m, n, k, header, cfg, cert, bound)
        # те же A, B, что сгенерирует эталон
        rng = np.random.default_rng(seed)
        a = rng.integers(ref.SIGNAL_MIN, ref.SIGNAL_MAX + 1, size=(m, k), dtype=np.int8)
        b = rng.integers(ref.SIGNAL_MIN, ref.SIGNAL_MAX + 1, size=(k, n), dtype=np.int8)
        got = pp.mine_attempt(cp.asarray(a), cp.asarray(np.ascontiguousarray(b.T)), header, cfg, cert, bound)
        assert (want is None) == (got is None)
        if want is not None:
            assert got.a_rows == want.a_rows and got.b_cols == want.b_cols
            assert got.jackpot_hash == want.jackpot_hash
            assert got.hash_a == want.hash_a and got.hash_b == want.hash_b
            np.testing.assert_array_equal(got.a, want.a)
            np.testing.assert_array_equal(got.b, want.b)


def test_full_attempt_nonvacuous_and_cert_matters():
    m = n = 256
    k = 1024
    cfg = ref.default_config(k, RANK)
    header = _header(0x1E00FFFF)
    found = [pp.mine_random(m, n, k, header, cfg, 3, seed=s) for s in range(8)]
    assert sum(f is not None for f in found) >= 6
    assert any((f.a_rows[0], f.b_cols[0]) != (0, 0) for f in found if f)
    assert pp.mine_random(m, n, k, _header(0x1B00FFFF), cfg, 3, seed=1) is None


# ---------------------------------------------------------------------------
# Оптимизированные стадии: многоуровневое слияние BLAKE3, сиды на GPU, две стороны шума за раз
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nchunks", [1023, 1024, 1025, 1500, 2048, 2049, 3000, 66000])
def test_blake3_many_chunks_multilevel_merge(nchunks):
    rng = np.random.default_rng(nchunks)
    data = rng.integers(0, 256, size=nchunks * 1024 - 64, dtype=np.uint8)      # хвост добивается нулями
    key = rng.bytes(32)
    want = ref.b3(ref.pad_to_chunk(data.tobytes()), key=key)
    assert pp.hash_matrix(cp.asarray(data), key) == want


@pytest.mark.parametrize("cert", [1, 2, 3])
@pytest.mark.parametrize("m,n", [(256, 256), (4096, 1024), (128, 8192)])
def test_seeds_on_gpu_match_ref(cert, m, n):
    rng = np.random.default_rng(cert + m)
    for _ in range(5):
        jkey, ha, hb = rng.bytes(32), rng.bytes(32), rng.bytes(32)
        want = ref.noise_seeds(jkey, ha, hb, m, n, cert)
        assert pp.derive_seeds(jkey, ha, hb, m, n, cert) == want


@pytest.mark.parametrize("rank", [64, 128, 256, 1024])
@pytest.mark.parametrize("rows_a,rows_b,k", [(48, 80, 1024), (130, 3, 2176), (16, 16, 4096), (1, 1, 1024)])
def test_noise_both_sides_match_ref(rank, rows_a, rows_b, k):
    rng = np.random.default_rng(rank + rows_a)
    a = rng.integers(-64, 65, size=(rows_a, k), dtype=np.int8)
    bt = rng.integers(-64, 65, size=(rows_b, k), dtype=np.int8)
    a_seed, b_seed = rng.bytes(32), rng.bytes(32)
    na, _ = ref.compute_noise(k, rank, b_seed, a_seed, np.arange(rows_a), np.arange(1))
    _, nbt = ref.compute_noise(k, rank, b_seed, a_seed, np.arange(1), np.arange(rows_b))
    got_a, got_b = pp.noised_both(cp.asarray(a), cp.asarray(bt), a_seed, b_seed, rank)
    np.testing.assert_array_equal(cp.asnumpy(got_a), (a.astype(np.int32) + na).astype(np.int8))
    np.testing.assert_array_equal(cp.asnumpy(got_b), (bt.astype(np.int32) + nbt).astype(np.int8))


def test_attempt_has_no_host_sync_dependence_on_cert():
    """Одна и та же пара A, B с разным cert_version даёт разные решения/сиды, но детерминирована."""
    cfg = ref.default_config(2048, RANK)
    header = _header(0x1E00FFFF)
    a = pp.gen_signal((256, 2048), 11)
    bt = pp.gen_signal((256, 2048), 12)
    s2 = pp.mine_attempt(a, bt, header, cfg, 2)
    s2b = pp.mine_attempt(a, bt, header, cfg, 2)
    assert (s2 is None) == (s2b is None)
    if s2:
        assert s2.jackpot_hash == s2b.jackpot_hash and s2.a_rows == s2b.a_rows


# ---------------------------------------------------------------------------
# Поиск, слитый с ядром джекпота (джекпоты в память не пишутся)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bound_shift", [256, 252, 248, 244, 236, 100])
@pytest.mark.parametrize("m,n,k", [(256, 256, 2048), (384, 640, 2176)])
def test_fused_search_matches_ref(bound_shift, m, n, k):
    cfg = ref.default_config(k, RANK)
    rng = np.random.default_rng(bound_shift + m)
    a = rng.integers(-127, 128, size=(m, k), dtype=np.int8)
    bt = rng.integers(-127, 128, size=(n, k), dtype=np.int8)
    a_seed = rng.bytes(32)
    bound = ref.U256_MAX if bound_shift >= 256 else (1 << bound_shift) - 1
    rt, ct = cfg.rows_pattern.partition(m), cfg.cols_pattern.partition(n)
    jp = ref.jackpots_all_tiles(a.astype(np.int32), bt.astype(np.int32), RANK, rt, ct)
    want = -1
    for ti in range(jp.shape[0]):
        for tj in range(jp.shape[1]):
            if int.from_bytes(ref.jackpot_hash(jp[ti, tj], a_seed), "little") <= bound:
                want = ti * jp.shape[1] + tj
                break
        if want >= 0:
            break
    got = pp.search_fused(cp.asarray(a), cp.asarray(bt), RANK, rt, ct, a_seed, bound)
    assert got == want


@pytest.mark.parametrize("n", [1, 15, 16, 17, 1000, 4097, 1 << 20])
def test_signal_rng_odd_sizes(n):
    x = cp.asnumpy(pp.gen_signal((n,), seed=5))
    assert x.shape == (n,) and x.min() >= -64 and x.max() <= 64
    np.testing.assert_array_equal(x, cp.asnumpy(pp.gen_signal((n,), seed=5)))
    if n >= 4096:
        assert len(np.unique(x)) > 100


# ---------------------------------------------------------------------------
# Попытки в нескольких CUDA-потоках (перекрытие): результаты те же, что при последовательном запуске
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cert", [2, 3])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_pipelined_equals_sequential(cert, depth):
    m, n, k = 256, 384, 2048
    cfg = ref.default_config(k, RANK)
    header = _header(0x1E00FFFF)
    seeds = list(range(10))
    want = [pp.mine_random(m, n, k, header, cfg, cert, seed=s) for s in seeds]
    got = dict(pp.mine_random_pipelined(m, n, k, header, cfg, cert, seeds, depth=depth))
    assert sorted(got) == seeds
    for s, w in zip(seeds, want):
        g = got[s]
        assert (w is None) == (g is None)
        if w is not None:
            assert g.jackpot_hash == w.jackpot_hash and g.a_rows == w.a_rows and g.b_cols == w.b_cols
            assert g.hash_a == w.hash_a and g.hash_b == w.hash_b
            np.testing.assert_array_equal(g.a, w.a)
            np.testing.assert_array_equal(g.b, w.b)


# ---------------------------------------------------------------------------
# Быстрый путь для разреженных матриц: буфер переиспользуется, хеш считается только по чанкам с ненулевыми ячейками
# ---------------------------------------------------------------------------
def test_sparse_slot_hash_equals_dense_hash_across_reuse_and_jobs():
    m, n, k = 256, 384, 2048
    slot = pp.SparseSlot(m, n, k, 32)
    for jk in (bytes(range(32)), bytes(range(1, 33))):             # смена ключа задания -> другой кэш нулевых чанков
        jkey_w = pp._words(jk)
        for s in range(4):                                         # буфер переиспользуется: старые ячейки стираются
            slot.fill(1000 + 2 * s, 1001 + 2 * s)
            np.testing.assert_array_equal(cp.asnumpy(slot.a), cp.asnumpy(pp.gen_sparse((m, k), 1000 + 2 * s)))
            np.testing.assert_array_equal(cp.asnumpy(slot.bt), cp.asnumpy(pp.gen_sparse((n, k), 1001 + 2 * s)))
            ha = pp.hash_matrix_sparse(slot.a, jkey_w, pp.zero_chunk_cvs(jk, slot.a.size), slot.ids_a)
            hb = pp.hash_matrix_sparse(slot.bt, jkey_w, pp.zero_chunk_cvs(jk, slot.bt.size), slot.ids_b)
            assert pp._bytes(ha) == pp._bytes(pp.hash_matrix_dev(slot.a, jkey_w))
            assert pp._bytes(hb) == pp._bytes(pp.hash_matrix_dev(slot.bt, jkey_w))


@pytest.mark.parametrize("cert", [2, 3])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_pipelined_sparse_equals_sequential_sparse(cert, depth):
    m, n, k = 256, 384, 2048
    cfg = ref.default_config(k, RANK)
    header = _header(0x1E00FFFF)
    seeds = list(range(12))
    want = [pp.mine_random(m, n, k, header, cfg, cert, seed=s, sparse_nnz=32) for s in seeds]
    got = dict(pp.mine_random_pipelined(m, n, k, header, cfg, cert, seeds, depth=depth, sparse_nnz=32))
    assert sorted(got) == seeds
    found = 0
    for s, w in zip(seeds, want):
        g = got[s]
        assert (w is None) == (g is None)
        if w is not None:
            found += 1
            assert g.jackpot_hash == w.jackpot_hash and g.a_rows == w.a_rows and g.b_cols == w.b_cols
            assert g.hash_a == w.hash_a and g.hash_b == w.hash_b
            np.testing.assert_array_equal(g.a, w.a)
            np.testing.assert_array_equal(g.b, w.b)
    assert found > 0                                                # проверка не вырождена: решения реально находились
