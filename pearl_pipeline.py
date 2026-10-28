"""
pearl_pipeline.py — весь цикл попытки майнинга Pearl на GPU:
  случайные A, B^T -> BLAKE3 (keyed, паддинг до 1024) -> сиды -> шум -> ядро джекпота
  -> blake3 джекпотов по тайлам + сравнение со сложностью.

Матрицы остаются в памяти GPU. Хеши матриц, сиды, шум и поиск связаны между собой через память GPU:
хост не синхронизируется до самого конца попытки (одно чтение индекса найденного тайла).
Эталон для проверки — pearlhash_ref (побитовое совпадение, см. test_pearlhash_pipeline.py).
"""

from __future__ import annotations

import numpy as np

import cupy as cp

import pearl_gpu
import pearlhash_ref as ref
from blake3_cuda import CORE
from gpu_arch import nvrtc_arch

_SRC = CORE + r"""
// ---- BLAKE3 keyed хеш большого буфера (хвост до 1024 дополняется нулями) ----
// Один поток = один чанк (16 блоков по 64 байта); чтение uint4.
extern "C" __global__ void blake3_chunks(const u32* __restrict__ data, u64 nbytes,
                                         const u32* __restrict__ key, u32* __restrict__ cvs, u32 nchunks)
{
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nchunks) return;
    u32 cv[8], nx[8], blk[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = key[i];
    for (int b = 0; b < 16; ++b) {
        const u64 off = (u64)c * 1024 + b * 64;
        if (off + 64 <= nbytes) {
            const uint4* p = (const uint4*)(data + (off >> 2));
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint4 v = __ldg(p + w);
                blk[4*w] = v.x; blk[4*w+1] = v.y; blk[4*w+2] = v.z; blk[4*w+3] = v.w;
            }
        } else {
            #pragma unroll
            for (int w = 0; w < 16; ++w) {
                const u64 o = off + w * 4;
                blk[w] = o < nbytes ? data[o >> 2] : 0u;
            }
        }
        u32 fl = F_KEYED | (b == 0 ? F_CHUNK_START : 0u) | (b == 15 ? F_CHUNK_END : 0u);
        if (b == 15 && nchunks == 1) fl |= F_ROOT;
        compress_cv(cv, blk, c, 64, fl, nx);
        #pragma unroll
        for (int i = 0; i < 8; ++i) cv[i] = nx[i];
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) cvs[(u64)c * 8 + i] = cv[i];
}

// Слияние до 1024 соседних CV в один (выровненные группы — полные поддеревья BLAKE3;
// непарный последний узел переносится наверх). is_final: вся вход-последовательность — одна группа, корень с ROOT.
extern "C" __global__ void __launch_bounds__(512)
blake3_merge(const u32* __restrict__ in, u32* __restrict__ out, u32 n_in,
             const u32* __restrict__ key, int is_final)
{
    __shared__ u32 s[1024 * 8];
    const u32 tid = threadIdx.x;
    const u32 base = blockIdx.x * 1024u;
    u32 cnt = n_in - base < 1024u ? n_in - base : 1024u;
    for (u32 i = tid; i < cnt * 8; i += blockDim.x) s[i] = in[(u64)base * 8 + i];
    __syncthreads();
    u32 k[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) k[j] = key[j];
    while (cnt > 1) {
        const u32 nxt = (cnt + 1) / 2;
        u32 reg[16];
        const bool active = tid < nxt;
        const bool pair = active && (2 * tid + 1 < cnt);
        if (active) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) reg[j] = s[(2 * tid) * 8 + j];
            if (pair) {
                #pragma unroll
                for (int j = 8; j < 16; ++j) reg[j] = s[(2 * tid) * 8 + j];
            }
        }
        __syncthreads();
        if (active) {
            u32 o[8];
            if (pair) {
                compress_cv(k, reg, 0, 64, F_PARENT | F_KEYED | ((is_final && cnt == 2) ? F_ROOT : 0u), o);
            } else {
                #pragma unroll
                for (int j = 0; j < 8; ++j) o[j] = reg[j];
            }
            #pragma unroll
            for (int j = 0; j < 8; ++j) s[tid * 8 + j] = o[j];
        }
        __syncthreads();
        cnt = nxt;
    }
    if (tid < 8) out[(u64)blockIdx.x * 8 + tid] = s[tid];
}

// сиды шума: b_seed = out[0..8], a_seed = out[8..16]  (как pearlhash_ref.noise_seeds)
extern "C" __global__ void derive_seeds(const u32* __restrict__ jkey, const u32* __restrict__ ha_g,
                                        const u32* __restrict__ hb_g, int m, int n, int salted,
                                        const u32* __restrict__ salt_a, const u32* __restrict__ salt_b,
                                        u32* __restrict__ out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    u32 ha[8], hb[8], msg[16], sa[8], sb[8], bs[8], as_[8];
    for (int i = 0; i < 8; ++i) { ha[i] = ha_g[i]; hb[i] = hb_g[i]; }
    if (salted) {
        for (int i = 0; i < 8; ++i) sa[i] = salt_a[i];
        for (int i = 0; i < 8; ++i) sb[i] = salt_b[i];
        for (int i = 0; i < 16; ++i) msg[i] = 0u;
        for (int i = 0; i < 8; ++i) msg[i] = ha[i];
        msg[8] = (u32)m;
        u32 t[8]; hash64(sa, msg, t);
        for (int i = 0; i < 8; ++i) ha[i] = t[i];
        for (int i = 0; i < 16; ++i) msg[i] = 0u;
        for (int i = 0; i < 8; ++i) msg[i] = hb[i];
        msg[8] = (u32)n;
        hash64(sb, msg, t);
        for (int i = 0; i < 8; ++i) hb[i] = t[i];
    }
    for (int i = 0; i < 8; ++i) { msg[i] = jkey[i]; msg[8 + i] = hb[i]; }
    hash64_plain(msg, bs);
    for (int i = 0; i < 8; ++i) { msg[i] = bs[i]; msg[8 + i] = ha[i]; }
    hash64_plain(msg, as_);
    for (int i = 0; i < 8; ++i) { out[i] = bs[i]; out[8 + i] = as_[i]; }
}

__device__ __forceinline__ void random_hash(int index, int slot, const u32 label[8], const u32 key[8], u32 out[8]) {
    u32 msg[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) { msg[i] = 0u; msg[8 + i] = label[i]; }
    msg[slot] = (u32)(1 + index);
    hash64(key, msg, out);
}

// Таблицы шума для обеих сторон за один запуск:
//   E: rows x rank, int8 в [-32, 31];  perm: k пар (first, second) как uint16
extern "C" __global__ void gen_tables(
    const u32* __restrict__ labA, const u32* __restrict__ keyA, signed char* __restrict__ EA,
    ushort2* __restrict__ permA, int rowsA,
    const u32* __restrict__ labB, const u32* __restrict__ keyB, signed char* __restrict__ EB,
    ushort2* __restrict__ permB, int rowsB, int k, int rank)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int bpr = rank / 32;
    const long long nUA = (long long)rowsA * bpr, nUB = (long long)rowsB * bpr, nP = (k + 7) / 8;
    long long i = t;
    const u32 *lab, *key;
    u32 label[8], kk[8], h[8];
    if (i < nUA + nUB) {
        const bool b = i >= nUA;
        if (b) i -= nUA;
        lab = b ? labB : labA; key = b ? keyB : keyA;
        signed char* E = b ? EB : EA;
        for (int j = 0; j < 8; ++j) { label[j] = lab[j]; kk[j] = key[j]; }
        random_hash((int)i, 0, label, kk, h);             // индекс = row*bpr + blk = i
        const long long row = i / bpr, blkn = i % bpr;
        #pragma unroll
        for (int w = 0; w < 8; ++w)
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                E[row * rank + blkn * 32 + w * 4 + j] = (signed char)((int)((h[w] >> (8 * j)) & 63u) - 32);
        return;
    }
    i -= nUA + nUB;
    if (i >= 2 * nP) return;
    const bool b = i >= nP;
    if (b) i -= nP;
    lab = b ? labB : labA; key = b ? keyB : keyA;
    ushort2* perm = b ? permB : permA;
    for (int j = 0; j < 8; ++j) { label[j] = lab[j]; kk[j] = key[j]; }
    random_hash((int)i, 1, label, kk, h);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const long long c = i * 8 + j;
        if (c < k) {
            const u32 u = h[j];
            const u32 first = u & (u32)(rank - 1);
            const u32 second = first ^ (1u + (u32)(((u64)(rank - 1) * (u64)u) >> 32));
            perm[c] = make_ushort2((unsigned short)first, (unsigned short)second);
        }
    }
}

// dst = src + E[:, first] - E[:, second] для обеих сторон. Блок: 16 строк, E-строки в shared;
// перестановка читается один раз на 16 строк. Поток обрабатывает 4 соседних столбца.
#define RB 16
extern "C" __global__ void __launch_bounds__(256)
apply_noise2(const signed char* __restrict__ srcA, signed char* __restrict__ dstA,
             const signed char* __restrict__ EA, const ushort2* __restrict__ permA, int rowsA,
             const signed char* __restrict__ srcB, signed char* __restrict__ dstB,
             const signed char* __restrict__ EB, const ushort2* __restrict__ permB, int rowsB,
             int k, int rank)
{
    const bool sb = blockIdx.y != 0;
    const signed char* src = sb ? srcB : srcA;
    signed char* dst = sb ? dstB : dstA;
    const signed char* E = sb ? EB : EA;
    const ushort2* perm = sb ? permB : permA;
    const int rows = sb ? rowsB : rowsA;
    const int r0 = blockIdx.x * RB;
    if (r0 >= rows) return;
    const int nr = min(RB, rows - r0);
    __shared__ signed char sE[RB * 1024];
    for (int i = threadIdx.x; i < nr * rank; i += 256) sE[i] = E[(long long)r0 * rank + i];
    __syncthreads();
    for (int cg = threadIdx.x; cg < k / 4; cg += 256) {
        const uint4 p = __ldg((const uint4*)perm + cg);
        const int f[4] = {(int)(p.x & 0xffffu), (int)(p.y & 0xffffu), (int)(p.z & 0xffffu), (int)(p.w & 0xffffu)};
        const int s[4] = {(int)(p.x >> 16), (int)(p.y >> 16), (int)(p.z >> 16), (int)(p.w >> 16)};
        for (int r = 0; r < nr; ++r) {
            const long long o = (long long)(r0 + r) * k + cg * 4;
            const u32 sv = *(const u32*)(src + o);
            u32 res = 0;
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int v = (int)(signed char)((sv >> (8 * b)) & 0xffu)
                            + (int)sE[r * rank + f[b]] - (int)sE[r * rank + s[b]];
                res |= ((u32)v & 0xffu) << (8 * b);
            }
            *(u32*)(dst + o) = res;
        }
    }
}

// равномерный сигнал в [-64, 64] (счётчиковый splitmix64); поток пишет 16 значений одним uint4
__device__ __forceinline__ int sig_map(u32 v) { return (int)(((u64)v * 129ull) >> 32) - 64; }
extern "C" __global__ void gen_signal(signed char* __restrict__ out, long long n, u64 seed)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long base = t * 16;
    if (base >= n) return;
    u32 w[4] = {0u, 0u, 0u, 0u};
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        u64 z = seed + (u64)(t * 8 + j) * 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        const u32 b0 = (u32)sig_map((u32)z) & 0xffu, b1 = (u32)sig_map((u32)(z >> 32)) & 0xffu;
        w[j >> 1] |= b0 << (16 * (j & 1)) | b1 << (16 * (j & 1) + 8);
    }
    if (base + 16 <= n) {
        *(uint4*)(out + base) = make_uint4(w[0], w[1], w[2], w[3]);
    } else {
        for (long long i = 0; base + i < n; ++i) out[base + i] = (signed char)((w[i >> 2] >> (8 * (i & 3))) & 0xffu);
    }
}

// разреженный сигнал: нули + nnz случайных ненулевых ячеек (позиции и значения из splitmix64)
extern "C" __global__ void scatter_sparse(signed char* __restrict__ out, long long n, u64 seed, int nnz)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nnz) return;
    u64 z = seed + (u64)(j + 1) * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    u64 w = (z ^ 0xD1B54A32D192ED03ull) * 0x9E3779B97F4A7C15ull;
    w = (w ^ (w >> 29)) * 0xBF58476D1CE4E5B9ull;
    w ^= w >> 32;
    const long long pos = (long long)(z % (u64)n);
    out[pos] = (signed char)sig_map((u32)w);
}

// первый (в порядке ti, tj) тайл с jackpot_hash <= bound
extern "C" __global__ void search(const u32* __restrict__ jp, const u32* __restrict__ key_g,
                                  const u32* __restrict__ bound_g, int ntiles, u32* __restrict__ best)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ntiles) return;
    u32 key[8], msg[16], h[8];
    for (int i = 0; i < 8; ++i) key[i] = key_g[i];
    #pragma unroll
    for (int i = 0; i < 16; ++i) msg[i] = jp[(long long)t * 16 + i];
    hash64(key, msg, h);
    const bool le = le256(h, bound_g);
    if (le) atomicMin(best, t);
}
"""

_mod = None
_words_cache: dict = {}


def _fn(name):
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=_SRC, backend="nvrtc", options=(nvrtc_arch(), "--std=c++17"))
    return _mod.get_function(name)


def _words(b: bytes) -> cp.ndarray:
    """32 байта -> 8 u32 на GPU (кэшируется: метки, соли, ключ задания)."""
    assert len(b) == 32
    arr = _words_cache.get(b)
    if arr is None:
        if len(_words_cache) > 256:
            _words_cache.clear()
        arr = _words_cache[b] = cp.asarray(np.frombuffer(b, dtype="<u4").copy())
    return arr


def _as_words(x) -> cp.ndarray:
    return _words(x) if isinstance(x, (bytes, bytearray)) else x


def _bytes(words: cp.ndarray) -> bytes:
    return cp.asnumpy(words).astype("<u4").tobytes()


def _blocks(n: int, threads: int = 256) -> tuple:
    return ((n + threads - 1) // threads,), (threads,)


# ---------------------------------------------------------------------------
def hash_matrix_dev(buf: cp.ndarray, key_w: cp.ndarray) -> cp.ndarray:
    """blake3(keyed) от buf (хвост дополнен нулями до 1024); ключ и результат — 8 u32 на GPU, без синхронизации."""
    data = cp.ascontiguousarray(buf).reshape(-1).view(cp.uint8)
    nbytes = int(data.size)
    assert nbytes % 4 == 0 and nbytes > 0
    nchunks = (nbytes + 1023) // 1024
    cur = cp.empty((nchunks, 8), dtype=cp.uint32)
    g, b = _blocks(nchunks, 64)
    _fn("blake3_chunks")(g, b, (data, np.uint64(nbytes), key_w, cur, np.uint32(nchunks)))
    n = nchunks
    while n > 1:
        groups = (n + 1023) // 1024
        nxt = cp.empty((groups, 8), dtype=cp.uint32)
        _fn("blake3_merge")((groups,), (512,), (cur, nxt, np.uint32(n), key_w, np.int32(1 if groups == 1 else 0)))
        cur, n = nxt, groups
    return cur[0]


def hash_matrix(buf: cp.ndarray, key: bytes) -> bytes:
    return _bytes(hash_matrix_dev(buf, _words(key)))


def derive_seeds_dev(jkey_w, ha_w, hb_w, m: int, n: int, cert_version: int) -> cp.ndarray:
    """(16,) u32: b_seed = [0:8], a_seed = [8:16]. cert >= 3 -> «солёные» сиды."""
    out = cp.empty(16, dtype=cp.uint32)
    _fn("derive_seeds")((1,), (32,), (jkey_w, ha_w, hb_w, np.int32(m), np.int32(n),
                                      np.int32(1 if cert_version >= 3 else 0),
                                      _words(ref.SEED_SALT_A), _words(ref.SEED_SALT_B), out))
    return out


def derive_seeds(jkey: bytes, hash_a: bytes, hash_b: bytes, m: int, n: int, cert_version: int):
    out = derive_seeds_dev(_words(jkey), _words(hash_a), _words(hash_b), m, n, cert_version)
    raw = _bytes(out)
    return raw[:32], raw[32:]


def _noise_both(src_a, src_b, key_a, label_a, key_b, label_b, rank: int):
    """Шум для обеих сторон: 2 запуска (таблицы, применение). src_*: rows×k int8 (src_b может быть пустым)."""
    k = src_a.shape[1]
    rows_a, rows_b = src_a.shape[0], src_b.shape[0]
    assert rank & (rank - 1) == 0 and 32 <= rank <= 1024 and k % 4 == 0
    e_a = cp.empty((max(rows_a, 1), rank), dtype=cp.int8)
    e_b = cp.empty((max(rows_b, 1), rank), dtype=cp.int8)
    p_a = cp.empty((k, 2), dtype=cp.uint16)
    p_b = cp.empty((k, 2), dtype=cp.uint16)
    total = rows_a * (rank // 32) + rows_b * (rank // 32) + 2 * ((k + 7) // 8)
    g, b = _blocks(total)
    _fn("gen_tables")(g, b, (label_a, key_a, e_a, p_a, np.int32(rows_a),
                             label_b, key_b, e_b, p_b, np.int32(rows_b), np.int32(k), np.int32(rank)))
    dst_a, dst_b = cp.empty_like(src_a), cp.empty_like(src_b)
    nb = (max(rows_a, rows_b) + 15) // 16
    _fn("apply_noise2")((nb, 2), (256,), (src_a, dst_a, e_a, p_a, np.int32(rows_a),
                                          src_b, dst_b, e_b, p_b, np.int32(rows_b), np.int32(k), np.int32(rank)))
    return dst_a, dst_b


def noised_both(a: cp.ndarray, bt: cp.ndarray, a_seed, b_seed, rank: int):
    """(A + шум_A, B^T + шум_B). Сиды — bytes или 8 u32 на GPU."""
    a, bt = cp.ascontiguousarray(a), cp.ascontiguousarray(bt)
    return _noise_both(a, bt, _as_words(a_seed), _words(ref.SEED_LABEL_A),
                       _as_words(b_seed), _words(ref.SEED_LABEL_B), rank)


def noised(src: cp.ndarray, seed: bytes, label: bytes, rank: int) -> cp.ndarray:
    """src (rows×k int8) + шум E_L·E_R, посчитанный на GPU из seed/label (одна сторона)."""
    src = cp.ascontiguousarray(src)
    empty = cp.empty((0, src.shape[1]), dtype=cp.int8)
    dst, _ = _noise_both(src, empty, _words(seed), _words(label), _words(seed), _words(label), rank)
    return dst


def gen_uniform(label: bytes, key: bytes, rows: int, rank: int) -> cp.ndarray:
    k = 64
    e = cp.empty((rows, rank), dtype=cp.int8)
    p = cp.empty((k, 2), dtype=cp.uint16)
    total = rows * (rank // 32) + 2 * ((k + 7) // 8)
    g, b = _blocks(total)
    z = _words(label)
    _fn("gen_tables")(g, b, (z, _words(key), e, p, np.int32(rows), z, _words(key), e, p, np.int32(0),
                             np.int32(k), np.int32(rank)))
    return e


def gen_perm(label: bytes, key: bytes, k: int, rank: int) -> cp.ndarray:
    e = cp.empty((1, rank), dtype=cp.int8)
    p = cp.empty((k, 2), dtype=cp.uint16)
    total = 2 * ((k + 7) // 8)
    g, b = _blocks(total)
    z = _words(label)
    _fn("gen_tables")(g, b, (z, _words(key), e, p, np.int32(0), z, _words(key), e, p, np.int32(0),
                             np.int32(k), np.int32(rank)))
    return p


def gen_signal(shape, seed: int) -> cp.ndarray:
    out = cp.empty(shape, dtype=cp.int8)
    g, b = _blocks((out.size + 15) // 16)
    _fn("gen_signal")(g, b, (out, np.int64(out.size), np.uint64(seed & (2**64 - 1))))
    return out


def gen_sparse(shape, seed: int, nnz: int = 32) -> cp.ndarray:
    """
    Разреженный сигнал: нули и nnz случайных ячеек в [-64, 64]. Каждая попытка остаётся уникальной (seed меняет
    позиции/значения, а значит и хеш матрицы и весь шум), а доказательство сжимается gzip в десятки раз.
    """
    out = cp.empty(shape, dtype=cp.int8)
    cp.cuda.runtime.memsetAsync(out.data.ptr, 0, out.nbytes, cp.cuda.get_current_stream().ptr)
    g, b = _blocks(max(nnz, 1), 64)
    _fn("scatter_sparse")(g, b, (out, np.int64(out.size), np.uint64(seed & (2**64 - 1)), np.int32(nnz)))
    return out


_bound_cache: dict = {}


def _bound_words(bound: int) -> cp.ndarray:
    bw = _bound_cache.get(bound)
    if bw is None:
        if len(_bound_cache) > 64:
            _bound_cache.clear()
        bw = _bound_cache[bound] = cp.asarray(np.frombuffer(int(bound).to_bytes(32, "little"), dtype="<u4").copy())
    return bw


def _new_best() -> cp.ndarray:
    best = cp.empty(1, dtype=cp.uint32)
    cp.cuda.runtime.memsetAsync(best.data.ptr, 0xFF, 4, cp.cuda.get_current_stream().ptr)   # 0xFFFFFFFF = «не найдено»
    return best


def _read_best(best: cp.ndarray) -> int:
    r = int(cp.asnumpy(best)[0])
    return -1 if r == 0xFFFFFFFF else r


def search_jackpots(jp: cp.ndarray, a_seed, bound: int) -> int:
    """Индекс (ti*T_c + tj) первого тайла с jackpot_hash <= bound или -1. a_seed — bytes или 8 u32 на GPU."""
    jp = cp.ascontiguousarray(jp)
    ntiles = jp.shape[0] * jp.shape[1]
    best = _new_best()
    g, b = _blocks(ntiles)
    _fn("search")(g, b, (jp, _as_words(a_seed), _bound_words(bound), np.int32(ntiles), best))
    return _read_best(best)


def search_fused(a_n, bt_n, rank: int, rt: np.ndarray, ct: np.ndarray, a_seed, bound: int) -> int:
    """Ядро джекпота (v2) со встроенным хешем тайлов и сравнением со сложностью; массив джекпотов не пишется."""
    best = _new_best()
    pearl_gpu.jackpots_dev(a_n, bt_n, rank, rt, ct, "auto", search=(_as_words(a_seed), _bound_words(bound), best))
    return _read_best(best)


# ---------------------------------------------------------------------------
_tiles_cache: dict = {}


def tiles_for(cfg: ref.MiningConfiguration, m: int, n: int):
    """Индексы тайлов (partition) — считаются один раз на конфигурацию и размер."""
    key = (cfg.rows_pattern.shape, cfg.cols_pattern.shape, m, n)
    if key not in _tiles_cache:
        _tiles_cache[key] = (cfg.rows_pattern.partition(m), cfg.cols_pattern.partition(n))
    return _tiles_cache[key]


class _Attempt:
    """Запущенная, но ещё не завершённая попытка (всё лежит в памяти GPU, хост ждёт только в finish_attempt)."""
    __slots__ = ("stream", "a", "bt", "a_n", "bt_n", "rank", "rt", "ct", "a_seed_w", "ha", "hb", "best", "idx", "jp")


def launch_attempt(a: cp.ndarray, bt: cp.ndarray, header: ref.IncompleteBlockHeader, cfg: ref.MiningConfiguration,
                   cert_version: int, bound: int | None = None) -> _Attempt:
    """Ставит в очередь текущего CUDA-потока всю попытку для A (m×k) и B^T (n×k), int8 в [-64, 64]; без ожидания."""
    rank = cfg.rank
    m, k = a.shape
    n = bt.shape[0]
    jkey_w = _words(ref.job_key(header, cfg))
    ha = hash_matrix_dev(a, jkey_w)
    hb = hash_matrix_dev(bt, jkey_w)
    seeds = derive_seeds_dev(jkey_w, ha, hb, m, n, cert_version)
    b_seed_w, a_seed_w = seeds[:8], seeds[8:]

    a_n, bt_n = noised_both(a, bt, a_seed_w, b_seed_w, rank)

    rt, ct = tiles_for(cfg, m, n)
    if bound is None:
        bound = ref.difficulty_bound(header.nbits, cfg)
    st = _Attempt()
    st.stream, st.a, st.bt, st.a_n, st.bt_n, st.rank, st.rt, st.ct = cp.cuda.get_current_stream(), a, bt, a_n, bt_n, rank, rt, ct
    st.a_seed_w, st.ha, st.hb, st.best, st.idx, st.jp = a_seed_w, ha, hb, None, None, None
    if pearl_gpu.v2_supported(m, n, k, rank, rt.shape[1], ct.shape[1]):
        st.best = _new_best()
        pearl_gpu.jackpots_dev(a_n, bt_n, rank, rt, ct, "auto",
                               search=(a_seed_w, _bound_words(bound), st.best))
    else:                                                   # универсальное ядро: поиск отдельным запуском
        st.jp = pearl_gpu.jackpots_dev(a_n, bt_n, rank, rt, ct)
        st.idx = search_jackpots(st.jp, a_seed_w, bound)
    return st


def finish_attempt(st: _Attempt) -> ref.Solution | None:
    """Ждёт завершения попытки и, если найден тайл под сложность, собирает решение."""
    with st.stream:
        idx = _read_best(st.best) if st.best is not None else st.idx
        if idx < 0:
            return None
        jp = st.jp
        if jp is None:                                      # решение найдено (редко): джекпоты нужны для хеша
            jp = pearl_gpu.jackpots_dev(st.a_n, st.bt_n, st.rank, st.rt, st.ct)
        ti, tj = divmod(idx, st.ct.shape[0])
        h = ref.jackpot_hash(cp.asnumpy(jp[ti, tj]), _bytes(st.a_seed_w))
        return ref.Solution(cp.asnumpy(st.a), np.ascontiguousarray(cp.asnumpy(st.bt).T),
                            st.rt[ti].tolist(), st.ct[tj].tolist(), h, _bytes(st.ha), _bytes(st.hb))


def mine_attempt(a: cp.ndarray, bt: cp.ndarray, header: ref.IncompleteBlockHeader, cfg: ref.MiningConfiguration,
                 cert_version: int, bound: int | None = None) -> ref.Solution | None:
    """Одна попытка на GPU для заданных A (m×k) и B^T (n×k), int8 в [-64, 64]."""
    return finish_attempt(launch_attempt(a, bt, header, cfg, cert_version, bound))


_streams: list = []


def mine_random_pipelined(m: int, n: int, k: int, header: ref.IncompleteBlockHeader, cfg: ref.MiningConfiguration,
                          cert_version: int, seeds, bound: int | None = None, depth: int = 2,
                          sparse_nnz: int | None = None):
    """
    Попытки со случайными A, B^T для каждого seed, `depth` штук одновременно в разных CUDA-потоках:
    пока хост дожидается результата предыдущей, GPU уже считает следующую (малые ядра BLAKE3/шума
    перекрываются с тяжёлым ядром джекпота). Генератор отдаёт (seed, решение или None) по порядку seed.
    """
    while len(_streams) < depth:
        _streams.append(cp.cuda.Stream(non_blocking=True))
    pending: list = []
    for i, seed in enumerate(seeds):
        stream = _streams[i % depth]
        with stream:
            a, bt = _gen_matrices(m, n, k, seed, sparse_nnz)
            pending.append((seed, launch_attempt(a, bt, header, cfg, cert_version, bound)))
        if len(pending) >= depth:
            sd, st = pending.pop(0)
            yield sd, finish_attempt(st)
    for sd, st in pending:
        yield sd, finish_attempt(st)


def _gen_matrices(m: int, n: int, k: int, seed: int, sparse_nnz: int | None):
    if sparse_nnz:
        return gen_sparse((m, k), seed * 2, sparse_nnz), gen_sparse((n, k), seed * 2 + 1, sparse_nnz)
    return gen_signal((m, k), seed * 2), gen_signal((n, k), seed * 2 + 1)


def mine_random(m: int, n: int, k: int, header: ref.IncompleteBlockHeader, cfg: ref.MiningConfiguration,
                cert_version: int, seed: int, bound: int | None = None,
                sparse_nnz: int | None = None) -> ref.Solution | None:
    """Попытка со случайными A, B^T, сгенерированными на GPU (sparse_nnz — разреженные матрицы)."""
    a, bt = _gen_matrices(m, n, k, seed, sparse_nnz)
    return mine_attempt(a, bt, header, cfg, cert_version, bound)
