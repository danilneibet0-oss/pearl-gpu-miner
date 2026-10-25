"""
pearlhash_ref.py — эталонная (CPU, NumPy) реализация PoW-поиска Pearl (NoisyGEMM).

Портировано с официального кода pearl-research-labs/pearl:
  zk-pow/src/ffi/mine.rs            — цикл майнинга (try_mine_one)
  zk-pow/src/circuit/pearl_noise.rs — генерация шума
  zk-pow/src/api/seed.rs            — V3 (salted) сиды
  zk-pow/src/api/proof_utils.rs     — сериализация, jackpot hash, nbits
  zk-pow/src/api/sanity_checks.rs   — граница сложности

Покрыт обычный (dense) режим. MoE-режим и упаковка PlainProof (Merkle-доказательства)
— следующий этап.

Зависимости: pip install numpy blake3
"""

from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass, field

import numpy as np
from blake3 import blake3

# ---------------------------------------------------------------------------
# Константы протокола
# ---------------------------------------------------------------------------
JACKPOT_SIZE = 16        # число u32 в джекпоте
LROT_PER_TILE = 13       # циклический сдвиг на каждый rank-блок
SIGNAL_MIN, SIGNAL_MAX = -64, 64
CHUNK_LEN = 1024         # BLAKE3 chunk
DIGEST = 32

NOISE_RANGE = 128
IDXS_PER_COL = 2
UNIFORM_NOISE_RANGE = NOISE_RANGE // IDXS_PER_COL      # 64
ZERO_POINT = UNIFORM_NOISE_RANGE // 2                  # 32
RANGE_MASK = UNIFORM_NOISE_RANGE - 1                   # 63

SEED_LABEL_A = b"A_tensor".ljust(32, b"\0")
SEED_LABEL_B = b"B_tensor".ljust(32, b"\0")

SEED_SALT_A = bytes.fromhex("8249406ca0ed15169616f692fcf076f892dbdb2a7023b852f0d47719c390017b")
SEED_SALT_B = bytes.fromhex("11300632ec6301ca2be2af718b3f4d4f1ae9c63988e8cc044844301d71b89aa9")

U256_MAX = (1 << 256) - 1


def b3(data: bytes, key: bytes | None = None) -> bytes:
    if key is None:
        return blake3(data).digest()
    return blake3(data, key=key).digest()


def pad_to_chunk(data: bytes) -> bytes:
    rem = len(data) % CHUNK_LEN
    return data if rem == 0 else data + b"\0" * (CHUNK_LEN - rem)


# ---------------------------------------------------------------------------
# PeriodicPattern — 3D арифметическая прогрессия индексов
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PeriodicPattern:
    shape: tuple  # ((stride, length), (stride, length), (stride, length))

    @classmethod
    def from_list(cls, pattern: list[int]) -> "PeriodicPattern":
        assert pattern and pattern[0] == 0
        assert all(a < b for a, b in zip(pattern, pattern[1:]))
        p = list(pattern)
        shape = []
        while len(p) > 1:
            for period in range(1, len(p)):
                if len(p) % period == 0:
                    s = p[period]
                    if all(p[i] + s == p[i + period] for i in range(len(p) - period)):
                        shape.append((s, len(p) // period))
                        p = p[:period]
                        break
            else:
                raise ValueError("pattern is not periodic")
        shape.reverse()
        per = shape[-1][0] * shape[-1][1] if shape else 1
        while len(shape) < 3:
            shape.append((per, 1))
        return cls(tuple(shape))

    def to_bytes(self) -> bytes:
        out = bytearray(6)
        min_stride = 1
        for i, (stride, length) in enumerate(self.shape):
            out[2 * i] = stride // min_stride - 1
            out[2 * i + 1] = length - 1
            min_stride = stride * length
        return bytes(out)

    def to_list(self) -> list[int]:
        res = [0]
        for stride, length in self.shape:
            res = [r + i * stride for i in range(length) for r in res]
        return res

    def period(self) -> int:
        s, l = self.shape[-1]
        return s * l

    def size(self) -> int:
        n = 1
        for _, l in self.shape:
            n *= l
        return n

    def offset_is_valid(self, offset: int) -> bool:
        for stride, length in reversed(self.shape):
            offset %= stride * length
            if offset >= stride:
                return False
        return True

    def partition(self, total: int) -> np.ndarray:
        """Все тайлы: массив (число_тайлов, size) индексов (threads_partition)."""
        assert total % self.period() == 0, "dimension must be divisible by pattern period"
        base = np.array(self.to_list(), dtype=np.int64)
        offsets = [o for o in range(total) if self.offset_is_valid(o)]
        return np.array(offsets, dtype=np.int64)[:, None] + base[None, :]


# ---------------------------------------------------------------------------
# Заголовок и конфигурация
# ---------------------------------------------------------------------------
@dataclass
class IncompleteBlockHeader:
    version: int
    prev_block: bytes      # 32 байта (в сериализации разворачиваются)
    merkle_root: bytes     # 32 байта (в сериализации разворачиваются)
    timestamp: int
    nbits: int

    def to_bytes(self) -> bytes:
        return (struct.pack("<I", self.version) + self.prev_block[::-1] + self.merkle_root[::-1]
                + struct.pack("<II", self.timestamp, self.nbits))

    @classmethod
    def from_bytes(cls, d: bytes) -> "IncompleteBlockHeader":
        assert len(d) == 76
        v, = struct.unpack_from("<I", d, 0)
        ts, nb = struct.unpack_from("<II", d, 68)
        return cls(v, d[4:36][::-1], d[36:68][::-1], ts, nb)


@dataclass
class MiningConfiguration:
    common_dim: int                 # k
    rank: int                       # r
    rows_pattern: PeriodicPattern
    cols_pattern: PeriodicPattern
    mma_type: int = 0               # Int7xInt7ToInt32

    def to_bytes(self) -> bytes:
        return (struct.pack("<IHH", self.common_dim, self.rank, self.mma_type)
                + self.rows_pattern.to_bytes() + self.cols_pattern.to_bytes() + b"\0" * 32)

    def dot_product_length(self) -> int:
        return self.common_dim - self.common_dim % self.rank

    def tile_size(self) -> int:
        return self.rows_pattern.size() * self.cols_pattern.size()


def job_key(header: IncompleteBlockHeader, cfg: MiningConfiguration) -> bytes:
    return b3(header.to_bytes() + cfg.to_bytes())


# ---------------------------------------------------------------------------
# Сложность
# ---------------------------------------------------------------------------
def nbits_to_target(nbits: int) -> int:
    exp, mant = nbits >> 24, nbits & 0x00FFFFFF
    if mant == 0 or exp == 0 or mant & 0x00800000:
        return 0
    return mant >> (8 * (3 - exp)) if exp <= 3 else (mant << (8 * (exp - 3))) & U256_MAX


def difficulty_bound(nbits: int, cfg: MiningConfiguration) -> int:
    base = nbits_to_target(nbits)
    factor = cfg.tile_size() * cfg.dot_product_length()
    return U256_MAX if base > U256_MAX // factor else base * factor


# ---------------------------------------------------------------------------
# Сиды шума (V2 = legacy, V3 = salted)
# ---------------------------------------------------------------------------
def bind_root(root: bytes, dim: int, salt: bytes) -> bytes:
    return b3(root + struct.pack("<I", dim) + b"\0" * 28, key=salt)


def noise_seeds(jkey: bytes, hash_a: bytes, hash_b: bytes, m: int, n: int,
                cert_version: int) -> tuple[bytes, bytes]:
    if cert_version >= 3:
        hash_a = bind_root(hash_a, m, SEED_SALT_A)
        hash_b = bind_root(hash_b, n, SEED_SALT_B)
    b_seed = b3(jkey + hash_b)
    a_seed = b3(b_seed + hash_a)
    return b_seed, a_seed


# ---------------------------------------------------------------------------
# Генерация шума (низкоранговая: E_L · E_R)
# ---------------------------------------------------------------------------
def _random_hash(index: int, label: bytes, key: bytes, prepend_slot: int) -> bytes:
    msg = bytearray(64)
    msg[prepend_slot * 4: prepend_slot * 4 + 4] = struct.pack("<i", 1 + index)
    msg[32:] = label
    return b3(bytes(msg), key=key)


def uniform_matrix(label: bytes, key: bytes, rows: np.ndarray, rank: int) -> np.ndarray:
    """Строки (rows) равномерной матрицы шума, значения в [-32, 31]."""
    blocks_per_row = rank // DIGEST
    out = np.empty((len(rows), rank), dtype=np.int16)
    for i, r in enumerate(rows):
        raw = b"".join(_random_hash(int(r) * blocks_per_row + b, label, key, 0)
                       for b in range(blocks_per_row))
        out[i] = (np.frombuffer(raw, dtype=np.uint8) & RANGE_MASK).astype(np.int16) - ZERO_POINT
    return out


def permutation_pairs(label: bytes, key: bytes, k: int, rank: int) -> np.ndarray:
    """k пар (first, second): строка разреженной матрицы с +1 и -1."""
    lines_per_hash = DIGEST // 4
    n_hash = -(-k // lines_per_hash)
    raw = b"".join(_random_hash(i, label, key, 1) for i in range(n_hash))
    u = np.frombuffer(raw, dtype="<u4")[:k].astype(np.uint64)
    first = u & np.uint64(rank - 1)
    second = first ^ (np.uint64(1) + ((np.uint64(rank - 1) * u) >> np.uint64(32)))
    return np.stack([first, second], axis=1).astype(np.int64)


def compute_noise(k: int, rank: int, b_seed: bytes, a_seed: bytes,
                  a_rows: np.ndarray, b_cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает noise_A (len(a_rows) × k) и noise_B^T (len(b_cols) × k)."""
    e_al = uniform_matrix(SEED_LABEL_A, a_seed, a_rows, rank)
    e_ar = permutation_pairs(SEED_LABEL_A, a_seed, k, rank)
    e_bl = permutation_pairs(SEED_LABEL_B, b_seed, k, rank)
    e_br = uniform_matrix(SEED_LABEL_B, b_seed, b_cols, rank)
    noise_a = (e_al[:, e_ar[:, 0]] - e_al[:, e_ar[:, 1]]).astype(np.int8)
    noise_bt = (e_br[:, e_bl[:, 0]] - e_br[:, e_bl[:, 1]]).astype(np.int8)
    return noise_a, noise_bt


# ---------------------------------------------------------------------------
# Джекпот
# ---------------------------------------------------------------------------
def _rotl32(x: np.ndarray, r: int) -> np.ndarray:
    return ((x << np.uint32(r)) | (x >> np.uint32(32 - r))).astype(np.uint32)


def jackpots_all_tiles(a_noised: np.ndarray, bt_noised: np.ndarray, rank: int,
                       row_tiles: np.ndarray, col_tiles: np.ndarray) -> np.ndarray:
    """
    Векторизованный джекпот для всех тайлов сразу.
    a_noised: m×k, bt_noised: n×k (int32). Возвращает (T_r, T_c, 16) uint32.
    """
    m, k = a_noised.shape
    a = a_noised.astype(np.int64)
    bt = bt_noised.astype(np.int64)
    acc = np.zeros((m, bt.shape[0]), dtype=np.int64)
    jp = np.zeros((row_tiles.shape[0], col_tiles.shape[0], JACKPOT_SIZE), dtype=np.uint32)
    for ll in range(rank, k + 1, rank):
        acc += a[:, ll - rank:ll] @ bt[:, ll - rank:ll].T
        u = (acc & 0xFFFFFFFF).astype(np.uint32)            # i32 → u32 (two's complement)
        g = u[row_tiles][:, :, col_tiles]                   # (T_r, h, T_c, w)
        x = np.bitwise_xor.reduce(np.bitwise_xor.reduce(g, axis=3), axis=1)  # (T_r, T_c)
        tid = (ll // rank - 1) % JACKPOT_SIZE
        jp[:, :, tid] = _rotl32(jp[:, :, tid], LROT_PER_TILE) ^ x
    return jp


def jackpot_hash(jackpot: np.ndarray, a_seed: bytes) -> bytes:
    return b3(jackpot.astype("<u4").tobytes(), key=a_seed)


# ---------------------------------------------------------------------------
# Одна попытка майнинга
# ---------------------------------------------------------------------------
@dataclass
class Solution:
    a: np.ndarray            # m×k int8 (без шума — это и коммитится)
    b: np.ndarray            # k×n int8
    a_rows: list[int]
    b_cols: list[int]
    jackpot_hash: bytes
    hash_a: bytes = field(repr=False, default=b"")
    hash_b: bytes = field(repr=False, default=b"")


def try_mine_one(rng: np.random.Generator, m: int, n: int, k: int,
                 header: IncompleteBlockHeader, cfg: MiningConfiguration,
                 cert_version: int = 3, bound: int | None = None) -> Solution | None:
    rank = cfg.rank
    a = rng.integers(SIGNAL_MIN, SIGNAL_MAX + 1, size=(m, k), dtype=np.int8)
    b = rng.integers(SIGNAL_MIN, SIGNAL_MAX + 1, size=(k, n), dtype=np.int8)
    bt = np.ascontiguousarray(b.T)

    jkey = job_key(header, cfg)
    hash_a = b3(pad_to_chunk(a.tobytes()), key=jkey)     # row-major A
    hash_b = b3(pad_to_chunk(bt.tobytes()), key=jkey)    # column-major B
    b_seed, a_seed = noise_seeds(jkey, hash_a, hash_b, m, n, cert_version)

    noise_a, noise_bt = compute_noise(k, rank, b_seed, a_seed, np.arange(m), np.arange(n))
    a_noised = a.astype(np.int32) + noise_a
    bt_noised = bt.astype(np.int32) + noise_bt

    row_tiles = cfg.rows_pattern.partition(m)
    col_tiles = cfg.cols_pattern.partition(n)
    jp = jackpots_all_tiles(a_noised, bt_noised, rank, row_tiles, col_tiles)

    if bound is None:
        bound = difficulty_bound(header.nbits, cfg)
    for ti in range(row_tiles.shape[0]):          # тот же порядок обхода, что в Rust
        for tj in range(col_tiles.shape[0]):
            h = jackpot_hash(jp[ti, tj], a_seed)
            if int.from_bytes(h, "little") <= bound:
                return Solution(a, b, row_tiles[ti].tolist(), col_tiles[tj].tolist(), h, hash_a, hash_b)
    return None


# ---------------------------------------------------------------------------
# Бенчмарк / демо
# ---------------------------------------------------------------------------
def default_config(k: int = 1024, rank: int = 128) -> MiningConfiguration:
    return MiningConfiguration(
        common_dim=k, rank=rank,
        rows_pattern=PeriodicPattern.from_list([0, 8, 64, 72]),
        cols_pattern=PeriodicPattern.from_list([0, 1, 8, 9, 32, 33, 40, 41,
                                                64, 65, 72, 73, 96, 97, 104, 105]),
    )


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Pearl NoisyGEMM CPU reference miner (demo)")
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--nbits", type=lambda s: int(s, 0), default=0x1C00FFFF,
                    help="сложность в compact-формате (демо: лёгкая)")
    ap.add_argument("--cert", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=20)
    args = ap.parse_args()

    cfg = default_config(args.k)
    header = IncompleteBlockHeader(1, os.urandom(32), os.urandom(32), int(time.time()), args.nbits)
    rng = np.random.default_rng()
    t0, attempts, found = time.time(), 0, 0
    while time.time() - t0 < args.seconds:
        sol = try_mine_one(rng, args.m, args.n, args.k, header, cfg, args.cert)
        attempts += 1
        if sol:
            found += 1
            print(f"[+] решение: строки {sol.a_rows[:4]}…, столбцы {sol.b_cols[:4]}…, "
                  f"hash {sol.jackpot_hash[::-1].hex()[:16]}…")
    dt = time.time() - t0
    macs = attempts * args.m * args.n * args.k
    print(f"попыток: {attempts}, решений: {found}, {macs / dt / 1e9:.3f} GMAC/s за {dt:.1f} c")


if __name__ == "__main__":
    main()
