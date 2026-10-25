"""Тесты эталона: векторы из pearl-research-labs/pearl и сверка с прямым портом Rust-цикла."""

import numpy as np
import pytest

import pearlhash_ref as P

H = bytes.fromhex


def test_salts_match_context_strings():
    assert P.SEED_SALT_A == P.b3(b"pearl/cert-v3/noise-seed/A")
    assert P.SEED_SALT_B == P.b3(b"pearl/cert-v3/noise-seed/B")


# zk-pow/src/api/seed.rs :: commitment_hash_pinned_vectors (m=192, n=320)
@pytest.mark.parametrize("cert,exp_b,exp_a", [
    (2, "add6f7ea5feebf89c8a77e2ebfa0d82442e7dbb0046dbd48971861d12fcb0177",
        "483b07b6f73105030b9482255f37723f3fed69ae916724ee8291848b8c28794b"),
    (3, "60ed9b73c5a9599b200b6cd563e7f0d5d9a67d2402d85fd4ef966c580080d0e5",
        "301784168005ec833ab0aa60006f7fe7faaa95307d8c1fc6819b2ffdd717eccf"),
])
def test_commitment_hash_pinned_vectors(cert, exp_b, exp_a):
    b, a = P.noise_seeds(b"\x11" * 32, b"\xAA" * 32, b"\xBB" * 32, 192, 320, cert)
    assert b == H(exp_b)
    assert a == H(exp_a)


def test_pattern_roundtrip_and_partition():
    cfg = P.default_config()
    assert cfg.rows_pattern.to_list() == [0, 8, 64, 72]
    assert cfg.cols_pattern.to_list() == [0, 1, 8, 9, 32, 33, 40, 41, 64, 65, 72, 73, 96, 97, 104, 105]
    assert len(cfg.to_bytes()) == 52
    # наивный threads_partition из mine.rs
    for pat, total in ((cfg.rows_pattern, 256), (cfg.cols_pattern, 256)):
        base = pat.to_list()
        naive = [[o + d for d in base] for o in range(total) if pat.offset_is_valid(o)]
        assert pat.partition(total).tolist() == naive
        # тайлы покрывают все индексы ровно по одному разу
        assert sorted(sum(naive, [])) == list(range(total))


def test_header_roundtrip():
    h = P.IncompleteBlockHeader(1, bytes(range(32)), bytes(range(32, 64)), 123, 0x1D00FFFF)
    raw = h.to_bytes()
    assert len(raw) == 76 and raw[4:36] == bytes(range(32))[::-1]
    assert P.IncompleteBlockHeader.from_bytes(raw) == h


def test_nbits():
    assert P.nbits_to_target(0x1D00FFFF) == 0xFFFF << (8 * 26)
    assert P.nbits_to_target(0x04800000) == 0  # отрицательный бит


def test_noise_properties():
    k, r = 512, 128
    pairs = P.permutation_pairs(P.SEED_LABEL_A, b"\x01" * 32, k, r)
    assert pairs.shape == (k, 2)
    assert (pairs < r).all() and (pairs[:, 0] != pairs[:, 1]).all()
    u = P.uniform_matrix(P.SEED_LABEL_B, b"\x02" * 32, np.arange(5), r)
    assert u.min() >= -32 and u.max() <= 31
    # строки шума берутся по глобальному индексу: подвыборка = срез полной матрицы
    full = P.uniform_matrix(P.SEED_LABEL_B, b"\x02" * 32, np.arange(10), r)
    assert (P.uniform_matrix(P.SEED_LABEL_B, b"\x02" * 32, np.array([3, 7]), r) == full[[3, 7]]).all()


def naive_jackpot(a_noised, bt_noised, rank, a_rows, b_cols):
    """Прямой порт внутреннего цикла try_mine_one из zk-pow/src/ffi/mine.rs."""
    k = a_noised.shape[1]
    tile = np.zeros((len(a_rows), len(b_cols)), dtype=np.int64)
    jp = [0] * 16
    for ll in range(rank, k + 1, rank):
        for u, ai in enumerate(a_rows):
            for v, bj in enumerate(b_cols):
                tile[u, v] += int(np.dot(a_noised[ai, ll - rank:ll].astype(np.int64),
                                         bt_noised[bj, ll - rank:ll].astype(np.int64)))
                tile[u, v] = (tile[u, v] + 2**31) % 2**32 - 2**31   # wrapping i32
        x = 0
        for val in tile.flatten():
            x ^= int(val) & 0xFFFFFFFF
        tid = (ll // rank - 1) % 16
        j = jp[tid]
        jp[tid] = (((j << 13) | (j >> 19)) & 0xFFFFFFFF) ^ x
    return np.array(jp, dtype=np.uint32)


def test_vectorized_jackpot_matches_naive_port():
    rng = np.random.default_rng(1)
    m = n = 128
    k, r = 2304, 128          # 18 rank-блоков > 16: проверяем переиспользование слотов джекпота
    cfg = P.default_config(k, r)
    a = rng.integers(-64, 65, size=(m, k), dtype=np.int8)
    bt = rng.integers(-64, 65, size=(n, k), dtype=np.int8)
    na, nbt = P.compute_noise(k, r, b"\x05" * 32, b"\x06" * 32, np.arange(m), np.arange(n))
    an, btn = a.astype(np.int32) + na, bt.astype(np.int32) + nbt
    rt, ct = cfg.rows_pattern.partition(m), cfg.cols_pattern.partition(n)
    fast = P.jackpots_all_tiles(an, btn, r, rt, ct)
    for ti, tj in [(0, 0), (3, 5), (len(rt) - 1, len(ct) - 1)]:
        assert (fast[ti, tj] == naive_jackpot(an, btn, r, rt[ti], ct[tj])).all()


def test_end_to_end_easy_target_finds_and_rechecks():
    rng = np.random.default_rng(7)
    cfg = P.default_config(512)
    hdr = P.IncompleteBlockHeader(1, b"\x01" * 32, b"\x02" * 32, 0x66666666, 0x207FFFFF)
    sol = P.try_mine_one(rng, 128, 128, 512, hdr, cfg, cert_version=3)
    assert sol is not None
    # независимая перепроверка найденного тайла
    jkey = P.job_key(hdr, cfg)
    b_seed, a_seed = P.noise_seeds(jkey, sol.hash_a, sol.hash_b, 128, 128, 3)
    na, nbt = P.compute_noise(512, 128, b_seed, a_seed, np.arange(128), np.arange(128))
    an = sol.a.astype(np.int32) + na
    btn = np.ascontiguousarray(sol.b.T).astype(np.int32) + nbt
    jp = naive_jackpot(an, btn, 128, sol.a_rows, sol.b_cols)
    assert P.jackpot_hash(jp, a_seed) == sol.jackpot_hash


def test_hard_target_rejects():
    rng = np.random.default_rng(3)
    cfg = P.default_config(512)
    hdr = P.IncompleteBlockHeader(1, b"\0" * 32, b"\0" * 32, 0, 0x1D00FFFF)
    assert P.try_mine_one(rng, 128, 128, 512, hdr, cfg) is None
