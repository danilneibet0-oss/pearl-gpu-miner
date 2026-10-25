"""
pearl_gpu.py — GPU-ядро джекпота Pearl (NoisyGEMM) на INT8 тензорных ядрах.

Версия 1: простая и корректная (прямые загрузки из глобальной памяти, один варп на блок).
Эталон — pearlhash_ref.jackpots_all_tiles; результат обязан совпадать с ним побитово.

Схема: варп считает C-блок 16 строк × TW столбцов (mma.sync m16n8k32 s8, аккумуляторы в регистрах).
Строки A собираются по индексам тайлов (16/TH тайлов на варп), столбцы B^T — один col-тайл.
После каждого rank-блока по k аккумуляторы XOR-сворачиваются по тайлам (shuffle-butterfly),
затем jp[tid] = rotl(jp[tid], 13) ^ xor, tid = (ll/rank - 1) % 16. Матрица C в память не пишется.
"""

from __future__ import annotations

import numpy as np

import cupy as cp

import pearlhash_ref as ref
from blake3_cuda import CORE
from gpu_arch import nvrtc_arch

_KERNEL_SRC = r"""
typedef unsigned int u32;

__device__ __forceinline__ u32 rotl13(u32 v) { return (v << 13) | (v >> 19); }

extern "C" __global__ void jackpot_kernel(
    const signed char* __restrict__ A,    // m  x k, row-major (уже с шумом)
    const signed char* __restrict__ Bt,   // n  x k, row-major (= B^T с шумом)
    const int* __restrict__ ridx,         // T_r*TH индексов строк A (по тайлам)
    const int* __restrict__ cidx,         // T_c*TW индексов строк B^T (по тайлам)
    u32* __restrict__ out,                // T_r x T_c x 16
    int k, int n_col_tiles)
{
    constexpr int TPW = 16 / TH;          // row-тайлов на варп
    constexpr int NF  = TW / 8;           // n8-фрагментов на col-тайл
    const int lane = threadIdx.x;
    const int g = lane >> 2, q = lane & 3;
    const int rg = blockIdx.y, tj = blockIdx.x;

    __shared__ u32 jp[TPW][16];
    for (int i = lane; i < TPW * 16; i += 32) (&jp[0][0])[i] = 0u;
    __syncwarp();

    const signed char* a0p = A + (long long)ridx[rg * 16 + g]     * k + 4 * q;
    const signed char* a1p = A + (long long)ridx[rg * 16 + g + 8] * k + 4 * q;
    const signed char* bp[NF];
    #pragma unroll
    for (int j = 0; j < NF; ++j) bp[j] = Bt + (long long)cidx[tj * TW + j * 8 + g] * k + 4 * q;

    int acc[NF][4];
    #pragma unroll
    for (int j = 0; j < NF; ++j) { acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0; }

    const int nblk = k / RANK;
    for (int blk = 0; blk < nblk; ++blk) {
        #pragma unroll 1
        for (int kk = 0; kk < RANK; kk += 32) {
            const int off = blk * RANK + kk;
            const u32 a0 = *(const u32*)(a0p + off);
            const u32 a1 = *(const u32*)(a1p + off);
            const u32 a2 = *(const u32*)(a0p + off + 16);
            const u32 a3 = *(const u32*)(a1p + off + 16);
            #pragma unroll
            for (int j = 0; j < NF; ++j) {
                const u32 b0 = *(const u32*)(bp[j] + off);
                const u32 b1 = *(const u32*)(bp[j] + off + 16);
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+r"(acc[j][0]), "+r"(acc[j][1]), "+r"(acc[j][2]), "+r"(acc[j][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }

        // XOR-свёртка текущих аккумуляторов по тайлам
        u32 x0 = 0, x1 = 0;                // x0: строки g (0..7), x1: строки g+8
        #pragma unroll
        for (int j = 0; j < NF; ++j) {
            x0 ^= (u32)acc[j][0] ^ (u32)acc[j][1];
            x1 ^= (u32)acc[j][2] ^ (u32)acc[j][3];
        }
        const int tid = blk & 15;          // (ll/rank - 1) % 16
        if (TH == 16) {
            u32 x = x0 ^ x1;
            #pragma unroll
            for (int msk = 1; msk <= 16; msk <<= 1) x ^= __shfl_xor_sync(0xffffffffu, x, msk);
            if (lane == 0) jp[0][tid] = rotl13(jp[0][tid]) ^ x;
        } else {
            x0 ^= __shfl_xor_sync(0xffffffffu, x0, 1); x1 ^= __shfl_xor_sync(0xffffffffu, x1, 1);
            x0 ^= __shfl_xor_sync(0xffffffffu, x0, 2); x1 ^= __shfl_xor_sync(0xffffffffu, x1, 2);
            #pragma unroll
            for (int s = 1; s < TH; s <<= 1) {          // маски 4*s: объединяют строки внутри тайла
                x0 ^= __shfl_xor_sync(0xffffffffu, x0, 4 * s);
                x1 ^= __shfl_xor_sync(0xffffffffu, x1, 4 * s);
            }
            if (q == 0 && (g % TH) == 0) {
                const int t0 = g / TH, t1 = 8 / TH + g / TH;
                jp[t0][tid] = rotl13(jp[t0][tid]) ^ x0;
                jp[t1][tid] = rotl13(jp[t1][tid]) ^ x1;
            }
        }
        __syncwarp();
    }

    for (int i = lane; i < TPW * 16; i += 32) {
        const int t = i >> 4, s = i & 15;
        out[((long long)(rg * TPW + t) * n_col_tiles + tj) * 16 + s] = (&jp[0][0])[i];
    }
}
"""

_KERNEL_V2_SRC = CORE + r"""
#ifndef STAGES
#define STAGES 3
#endif
#define BM 128
#define BN 128
#define BK 64

__device__ __forceinline__ u32 rotl13(u32 v) { return (v << 13) | (v >> 19); }
__device__ __forceinline__ void cp_async16(u32 dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src));
}

// Блок 256 потоков (8 варпов 2x4) считает C-блок 128x128 (32 row-тайла x 8 col-тайлов по 4x16).
// Данные идут через shared memory (cp.async, STAGES стадий, swizzle без банк-конфликтов);
// варп: 64x32 (4x4 mma m16n8k32), аккумуляторы в регистрах. XOR-свёртка тайлов — reduce-scatter
// через shuffle (15 shfl на 16 значений): каждый поток в итоге владеет ровно одним тайлом.
extern "C" __global__ void __launch_bounds__(256)
jackpot_v2(const signed char* __restrict__ A, const signed char* __restrict__ Bt,
           const int* __restrict__ ridx, const int* __restrict__ cidx, u32* __restrict__ out,
           int k, int n_col_tiles, int nst, int spr,
           int do_search, const u32* __restrict__ skey, const u32* __restrict__ sbound, u32* __restrict__ best)
{
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* sA = smem;
    unsigned char* sB = smem + STAGES * BM * BK;
    u32* jps = (u32*)(smem + STAGES * (BM + BN) * BK);        // [16 слотов][256 потоков]
    int* sri = (int*)(jps + 16 * 256);
    int* sci = sri + BM;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, q = lane & 3, sw = (g >> 1) & 3;
    const int wm = warp >> 2, wn = warp & 3;
    const int bx = blockIdx.x, by = blockIdx.y;

    for (int i = tid; i < BM; i += 256) { sri[i] = ridx[by * BM + i]; sci[i] = cidx[bx * BN + i]; }
    #pragma unroll
    for (int s = 0; s < 16; ++s) jps[s * 256 + tid] = 0u;
    __syncthreads();

    int acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) { acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0; }

    auto load_stage = [&](int stage, int it) {
        const int kb = it * BK;
        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            const int cid = tid + 256 * t, row = cid >> 2, c = cid & 3;
            const int off = row * 64 + ((c ^ ((row >> 1) & 3)) << 4);
            cp_async16((u32)__cvta_generic_to_shared(sA + stage * BM * BK + off),
                       A + (long long)sri[row] * k + kb + c * 16);
            cp_async16((u32)__cvta_generic_to_shared(sB + stage * BN * BK + off),
                       Bt + (long long)sci[row] * k + kb + c * 16);
        }
    };

    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nst) load_stage(s, s);
        asm volatile("cp.async.commit_group;");
    }

    for (int it = 0; it < nst; ++it) {
        asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 2));
        __syncthreads();
        if (it + STAGES - 1 < nst) load_stage((it + STAGES - 1) % STAGES, it + STAGES - 1);
        asm volatile("cp.async.commit_group;");

        const u32* As = (const u32*)(sA + (it % STAGES) * BM * BK);
        const u32* Bs = (const u32*)(sB + (it % STAGES) * BN * BK);
        #pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int c0 = (((2 * kk) ^ sw) << 2) + q, c1 = (((2 * kk + 1) ^ sw) << 2) + q;
            u32 af[4][4], bf[4][2];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r0 = wm * 64 + i * 16 + g;
                af[i][0] = As[r0 * 16 + c0];
                af[i][1] = As[(r0 + 8) * 16 + c0];
                af[i][2] = As[r0 * 16 + c1];
                af[i][3] = As[(r0 + 8) * 16 + c1];
            }
            #pragma unroll
            for (int nf = 0; nf < 4; ++nf) {
                const int n0 = wn * 32 + nf * 8 + g;
                bf[nf][0] = Bs[n0 * 16 + c0];
                bf[nf][1] = Bs[n0 * 16 + c1];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int nf = 0; nf < 4; ++nf)
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(acc[i][nf][0]), "+r"(acc[i][nf][1]), "+r"(acc[i][nf][2]), "+r"(acc[i][nf][3])
                        : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]),
                          "r"(bf[nf][0]), "r"(bf[nf][1]));
        }

        if ((it + 1) % spr == 0) {                        // конец rank-блока
            const int slot = ((it + 1) / spr - 1) & 15;
            u32 v[16];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    v[(i << 2) | (j << 1)]     = (u32)(acc[i][2*j][0] ^ acc[i][2*j][1] ^ acc[i][2*j+1][0] ^ acc[i][2*j+1][1]);
                    v[(i << 2) | (j << 1) | 1] = (u32)(acc[i][2*j][2] ^ acc[i][2*j][3] ^ acc[i][2*j+1][2] ^ acc[i][2*j+1][3]);
                }
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                const bool hi = lane & 8;
                const u32 send = hi ? v[t] : v[t + 8], keep = hi ? v[t + 8] : v[t];
                v[t] = keep ^ __shfl_xor_sync(0xffffffffu, send, 8);
            }
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const bool hi = lane & 4;
                const u32 send = hi ? v[t] : v[t + 4], keep = hi ? v[t + 4] : v[t];
                v[t] = keep ^ __shfl_xor_sync(0xffffffffu, send, 4);
            }
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                const bool hi = lane & 2;
                const u32 send = hi ? v[t] : v[t + 2], keep = hi ? v[t + 2] : v[t];
                v[t] = keep ^ __shfl_xor_sync(0xffffffffu, send, 2);
            }
            {
                const bool hi = lane & 1;
                const u32 send = hi ? v[0] : v[1], keep = hi ? v[1] : v[0];
                v[0] = keep ^ __shfl_xor_sync(0xffffffffu, send, 1);
            }
            u32* p = &jps[slot * 256 + tid];
            *p = rotl13(*p) ^ v[0];
        }
    }

    const int l16 = lane & 15, h = lane >> 4;
    const int ii = l16 >> 2, jj = (l16 >> 1) & 1, xx = l16 & 1;
    const long long rt = by * 32 + wm * 16 + ii * 4 + 2 * xx + h;
    const long long ct = bx * 8 + wn * 2 + jj;
    if (do_search) {
        // джекпот-хеш тайла и сравнение со сложностью прямо в ядре: массив джекпотов в память не пишется
        u32 msg[16], key[8], h[8];
        #pragma unroll
        for (int s = 0; s < 16; ++s) msg[s] = jps[s * 256 + tid];
        #pragma unroll
        for (int i = 0; i < 8; ++i) key[i] = skey[i];
        hash64(key, msg, h);
        if (le256(h, sbound)) atomicMin(best, (u32)(rt * n_col_tiles + ct));
    } else {
        #pragma unroll
        for (int s = 0; s < 16; ++s) out[(rt * n_col_tiles + ct) * 16 + s] = jps[s * 256 + tid];
    }
}
"""

_V2_STAGES = 2      # 2 стадии -> 49 КБ shared, 2 блока на SM (замер: быстрее 3 и 4 стадий на 3060 Ti)
_V2_SMEM = _V2_STAGES * (128 + 128) * 64 + 16 * 256 * 4 + 2 * 128 * 4
_v2 = None
_idx_cache: dict = {}


def _kernel_v2():
    global _v2
    if _v2 is None:
        mod = cp.RawModule(code=_KERNEL_V2_SRC, backend="nvrtc",
                           options=(nvrtc_arch(), "--std=c++17", f"-DSTAGES={_V2_STAGES}"))
        _v2 = mod.get_function("jackpot_v2")
        _v2.max_dynamic_shared_size_bytes = _V2_SMEM
    return _v2


def _device_indices(tiles: np.ndarray) -> cp.ndarray:
    # ключ — id массива; сам массив хранится в значении, чтобы id не переиспользовался
    hit = _idx_cache.get(id(tiles))
    if hit is None or hit[0] is not tiles:
        hit = (tiles, cp.asarray(tiles.reshape(-1).astype(np.int32)))
        _idx_cache[id(tiles)] = hit
    return hit[1]


def v2_supported(m: int, n: int, k: int, rank: int, th: int, tw: int) -> bool:
    return th == 4 and tw == 16 and m % 128 == 0 and n % 128 == 0 and k % 64 == 0 and rank % 64 == 0 and k >= rank


_modules: dict = {}


def _kernel(th: int, tw: int, rank: int):
    key = (th, tw, rank)
    if key not in _modules:
        assert 16 % th == 0 and tw % 8 == 0 and rank % 32 == 0
        mod = cp.RawModule(code=_KERNEL_SRC, backend="nvrtc",
                           options=(nvrtc_arch(), "--std=c++17",
                                    f"-DTH={th}", f"-DTW={tw}", f"-DRANK={rank}"))
        _modules[key] = mod.get_function("jackpot_kernel")
    return _modules[key]


def jackpots(a_noised, bt_noised, rank: int, row_tiles: np.ndarray, col_tiles: np.ndarray,
             variant: str = "auto") -> np.ndarray:
    """То же, что jackpots_dev, но результат копируется на CPU (numpy)."""
    return cp.asnumpy(jackpots_dev(a_noised, bt_noised, rank, row_tiles, col_tiles, variant))


def jackpots_dev(a_noised, bt_noised, rank: int, row_tiles: np.ndarray, col_tiles: np.ndarray,
                 variant: str = "auto", search=None) -> cp.ndarray:
    """
    search: (ключ a_seed [8 u32], граница [8 u32 LE], best [1 u32]) — только v2: джекпот-хеш и сравнение со
    сложностью делаются в самом ядре, массив джекпотов не пишется; в best — atomicMin индекса тайла ti*T_c+tj.
    variant: "auto" (v2 для тайла 4x16 при m,n кратных 128, иначе простое ядро), "simple" или "v2".
    a_noised: m×k int8, bt_noised: n×k int8 (numpy или cupy).
    row_tiles/col_tiles: индексы тайлов из PeriodicPattern.partition (T×size).
    Возвращает (T_r, T_c, 16) uint32 на GPU — как pearlhash_ref.jackpots_all_tiles.
    """
    a = cp.asarray(a_noised)
    bt = cp.asarray(bt_noised)
    assert a.dtype == cp.int8 and bt.dtype == cp.int8
    a, bt = cp.ascontiguousarray(a), cp.ascontiguousarray(bt)
    m, k = a.shape
    n = bt.shape[0]
    assert bt.shape[1] == k and k % 4 == 0 and k >= rank
    t_r, th = row_tiles.shape
    t_c, tw = col_tiles.shape
    assert t_r * th == m and t_c * tw == n, "паттерны должны покрывать все строки/столбцы"
    assert (t_r * th) % 16 == 0, "число строк A должно делиться на 16"

    ridx, cidx = _device_indices(row_tiles), _device_indices(col_tiles)
    use_v2 = v2_supported(m, n, k, rank, th, tw) if variant == "auto" else variant == "v2"
    if search is not None:
        assert use_v2, "поиск внутри ядра есть только у v2"
    out = cp.empty(1 if search is not None else (t_r, t_c, 16), dtype=cp.uint32)
    if use_v2:
        assert v2_supported(m, n, k, rank, th, tw), "v2: нужен тайл 4x16, m,n кратны 128, k кратно 64"
        nst = (k - k % rank) // 64
        skey, sbound, best = search if search is not None else (out, out, out)
        _kernel_v2()((n // 128, m // 128), (256,),
                     (a, bt, ridx, cidx, out, np.int32(k), np.int32(t_c), np.int32(nst), np.int32(rank // 64),
                      np.int32(search is not None), skey, sbound, best),
                     shared_mem=_V2_SMEM)
        return out
    _kernel(th, tw, rank)((t_c, (t_r * th) // 16), (32,), (a, bt, ridx, cidx, out, np.int32(k), np.int32(t_c)))
    return out


def try_mine_one(rng: np.random.Generator, m: int, n: int, k: int,
                 header: ref.IncompleteBlockHeader, cfg: ref.MiningConfiguration,
                 cert_version: int = 3, bound: int | None = None) -> ref.Solution | None:
    """Как pearlhash_ref.try_mine_one, но джекпоты считает GPU (остальное пока на CPU)."""
    rank = cfg.rank
    a = rng.integers(ref.SIGNAL_MIN, ref.SIGNAL_MAX + 1, size=(m, k), dtype=np.int8)
    b = rng.integers(ref.SIGNAL_MIN, ref.SIGNAL_MAX + 1, size=(k, n), dtype=np.int8)
    bt = np.ascontiguousarray(b.T)

    jkey = ref.job_key(header, cfg)
    hash_a = ref.b3(ref.pad_to_chunk(a.tobytes()), key=jkey)
    hash_b = ref.b3(ref.pad_to_chunk(bt.tobytes()), key=jkey)
    b_seed, a_seed = ref.noise_seeds(jkey, hash_a, hash_b, m, n, cert_version)

    noise_a, noise_bt = ref.compute_noise(k, rank, b_seed, a_seed, np.arange(m), np.arange(n))
    a_noised = (a.astype(np.int32) + noise_a).astype(np.int8)    # |a|<=64, |шум|<=63 -> влезает в int8
    bt_noised = (bt.astype(np.int32) + noise_bt).astype(np.int8)

    row_tiles = cfg.rows_pattern.partition(m)
    col_tiles = cfg.cols_pattern.partition(n)
    jp = jackpots(a_noised, bt_noised, rank, row_tiles, col_tiles)

    if bound is None:
        bound = ref.difficulty_bound(header.nbits, cfg)
    for ti in range(row_tiles.shape[0]):
        for tj in range(col_tiles.shape[0]):
            h = ref.jackpot_hash(jp[ti, tj], a_seed)
            if int.from_bytes(h, "little") <= bound:
                return ref.Solution(a, b, row_tiles[ti].tolist(), col_tiles[tj].tolist(), h, hash_a, hash_b)
    return None
