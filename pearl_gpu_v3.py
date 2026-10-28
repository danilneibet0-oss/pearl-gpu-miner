"""
pearl_gpu_v3.py — ядро джекпота v3 (rank 128, тайл 4×16, m и n кратны 128).

Отличия от v2 (см. pearl_gpu.py):
  * фрагменты A и B читаются из shared memory через ldmatrix.x4 (6 инструкций на k32-шаг варпа вместо 24 LDS.32);
  * XOR-свёртка rank-блока откладывается: сразу после блока аккумуляторы «снимаются» в 16 регистров (XOR),
    а дорогая часть (reduce-scatter через shuffle + обновление джекпота) выполняется внутри следующего блока
    между mma, а не простаивающим тензорным конвейером;
  * параметры (число стадий, минимум блоков на SM) задаются макросами, чтобы подбирать их замером.
Результат побитово совпадает с v2/эталоном (test_pearlhash_gpu.py).
"""

import cupy as cp

from blake3_cuda import CORE
from gpu_arch import nvrtc_arch

OPTS = {"STAGES": 2, "MINB": 2, "ABL_NOEPI": 0, "ABL_NOLOAD": 0, "ABL_NOSYNC": 0, "ABL_EPI": 0}

_SRC = CORE + r"""
#ifndef STAGES
#define STAGES 2
#endif
#ifndef MINB
#define MINB 2
#endif
#ifndef ABL_NOEPI
#define ABL_NOEPI 0        // только для замеров (результат неверный): без свёртки
#endif
#ifndef ABL_NOLOAD
#define ABL_NOLOAD 0       // только для замеров: без копирования global -> shared
#endif
#ifndef ABL_EPI
#define ABL_EPI 0          // замеры: 1 — без обновления слота в shared (копим в регистре), 2 — XOR только 8 из 64 значений
#endif
#ifndef SWZ
#define SWZ 8              // ширина «группы» блоков по строкам для повторного использования A/B в L2
#endif
#ifndef ABL_NOSYNC
#define ABL_NOSYNC 0       // только для замеров: без барьеров
#endif
#define BM 128
#define BN 128
#define BK 64

__device__ __forceinline__ u32 rotl13(u32 v) { return (v << 13) | (v >> 19); }
__device__ __forceinline__ void cp_async16(u32 dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src));
}
__device__ __forceinline__ void ldsm4(u32& r0, u32& r1, u32& r2, u32& r3, u32 addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_s8(int (&c)[4], const u32 (&a)[4], u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Блок 256 потоков (8 варпов 4x2) считает C-блок 128x128 = 32 row-тайла x 8 col-тайлов (тайл 4x16).
// Варп: 32 строки x 64 столбца = 2 фрагмента m16 x 8 фрагментов n8. Строки A и B в shared переставлены так, что
// ВЕСЬ тайл 4x16 (64 элемента C) целиком лежит в регистрах ОДНОГО потока: поток (g, q) владеет row-тайлом g и col-тайлом q
// варпа. Поэтому XOR-свёртка тайла — это XOR 64 собственных регистров (около 32 инструкций LOP3), без shuffle и без обмена.
extern "C" __global__ void __launch_bounds__(256, MINB)
jackpot_v3(const signed char* __restrict__ A, const signed char* __restrict__ Bt,
           const int* __restrict__ ridx, const int* __restrict__ cidx, u32* __restrict__ out,
           int k, int n_col_tiles, int nst, int spr,
           int do_search, const u32* __restrict__ skey, const u32* __restrict__ sbound, u32* __restrict__ best,
           int gx, int gy)
{
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* sA = smem;
    unsigned char* sB = smem + STAGES * BM * BK;
    u32* jps = (u32*)(smem + STAGES * (BM + BN) * BK);        // [16 слотов][256 потоков]
    int* sri = (int*)(jps + 16 * 256);
    int* sci = sri + BM;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;                  // 4 варпа по строкам, 2 по столбцам
    const int g = lane >> 2, q = lane & 3;
    // порядок блоков по группам из SWZ строк-блоков: одновременно работающие блоки (~76 на карту) читают ~8 полос A и ~9 полос B,
    // которые помещаются в L2 — трафик DRAM падает в разы по сравнению с построчным обходом
    int bx, by;
    {
        const int L = blockIdx.x, gsize = SWZ * gx;
        const int grp = L / gsize, first = grp * SWZ;
        const int gsz = (gy - first) < SWZ ? (gy - first) : SWZ;
        const int inn = L - grp * gsize;
        bx = inn / gsz;
        by = first + inn % gsz;
    }

    for (int i = tid; i < BM; i += 256) { sri[i] = ridx[by * BM + i]; sci[i] = cidx[bx * BN + i]; }
    #pragma unroll
    for (int s = 0; s < 16; ++s) jps[s * 256 + tid] = 0u;
    __syncthreads();

    int acc[2][8][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j) { acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0; }

    // адреса ldmatrix: строки по 64 байта, 16-байтные куски переставлены (chunk ^ ((row>>1)&3)) — без банк-конфликтов
    const u32 sAaddr = (u32)__cvta_generic_to_shared(sA), sBaddr = (u32)__cvta_generic_to_shared(sB);
    const int swz = (lane >> 1) & 3;
    const u32 arow = (u32)((wm * 32 + ((lane >> 3) & 1) * 8 + (lane & 7)) * 64);
    const u32 brow = (u32)((wn * 64 + (lane >> 4) * 8 + (lane & 7)) * 64);
    const u32 aoff0 = arow + (u32)(((0 | (lane >> 4)) ^ swz) << 4), aoff1 = arow + (u32)(((2 | (lane >> 4)) ^ swz) << 4);
    const u32 boff0 = brow + (u32)(((0 | ((lane >> 3) & 1)) ^ swz) << 4), boff1 = brow + (u32)(((2 | ((lane >> 3) & 1)) ^ swz) << 4);

    // загрузка global -> shared. Строка shared s берётся из глобальной строки по перестановке (см. комментарий выше):
    //   A: s = w*32 + i*16 + r  ->  ridx[(w*8 + (r&7))*4 + (r>>3) + 2*i]
    //   B: s = w*64 + nf*8 + n  ->  cidx[(w*4 + (n>>1))*16 + 2*nf + (n&1)]
    u32 gofsA[2], gofsB[2], soff[2];
    #pragma unroll
    for (int t = 0; t < 2; ++t) {
        const int cid = tid + 256 * t, row = cid >> 2, c = cid & 3;
        const int wa = row >> 5, sla = row & 31, ia = sla >> 4, ra = sla & 15;
        const int arow_src = sri[(wa * 8 + (ra & 7)) * 4 + (ra >> 3) + 2 * ia];
        const int wb = row >> 6, slb = row & 63, nfb = slb >> 3, nb = slb & 7;
        const int brow_src = sci[(wb * 4 + (nb >> 1)) * 16 + 2 * nfb + (nb & 1)];
        gofsA[t] = (u32)arow_src * (u32)k + (u32)(c * 16);
        gofsB[t] = (u32)brow_src * (u32)k + (u32)(c * 16);
        soff[t] = (u32)(row * 64 + ((c ^ ((row >> 1) & 3)) << 4));
    }
    auto load_stage = [&](int stage, int it) {
        const u32 kb = (u32)(it * BK);
        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            cp_async16(sAaddr + stage * BM * BK + soff[t], A + (size_t)(gofsA[t] + kb));
            cp_async16(sBaddr + stage * BN * BK + soff[t], Bt + (size_t)(gofsB[t] + kb));
        }
    };
    auto stage_begin = [&](int it) {
        asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 2));
#if !ABL_NOSYNC
        __syncthreads();
#endif
#if !ABL_NOLOAD
        if (it + STAGES - 1 < nst) load_stage((it + STAGES - 1) % STAGES, it + STAGES - 1);
#endif
        asm volatile("cp.async.commit_group;");
    };
    // k32-шаг варпа: 2 + 4 ldmatrix.x4 и 16 mma
    auto compute = [&](int stage, int kk) {
        u32 af[2][4], bf[8][2];
        const u32 abase = sAaddr + stage * BM * BK + (kk ? aoff1 : aoff0);
        const u32 bbase = sBaddr + stage * BN * BK + (kk ? boff1 : boff0);
        #pragma unroll
        for (int i = 0; i < 2; ++i) ldsm4(af[i][0], af[i][1], af[i][2], af[i][3], abase + i * 1024);
        #pragma unroll
        for (int jp = 0; jp < 4; ++jp) ldsm4(bf[2*jp][0], bf[2*jp][1], bf[2*jp+1][0], bf[2*jp+1][1], bbase + jp * 1024);
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int nf = 0; nf < 8; ++nf) mma_s8(acc[i][nf], af[i], bf[nf][0], bf[nf][1]);
    };

    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nst) load_stage(s, s);
        asm volatile("cp.async.commit_group;");
    }

    const int nblk = nst >> 1;                              // rank 128 = 2 стадии по BK=64
    u32 xs = 0;
    for (int blk = 0; blk < nblk; ++blk) {
        const int it0 = blk * 2;
        stage_begin(it0);
        compute(it0 % STAGES, 0);
        compute(it0 % STAGES, 1);
        stage_begin(it0 + 1);
        compute((it0 + 1) % STAGES, 0);
        compute((it0 + 1) % STAGES, 1);
#if !ABL_NOEPI
        // XOR всех 64 элементов тайла (он целиком в регистрах этого потока) и обновление слота джекпота
        u32 x = 0;
#if ABL_EPI == 2
        x = (u32)(acc[0][0][0] ^ acc[0][1][1] ^ acc[1][2][2] ^ acc[1][3][3] ^ acc[0][4][0] ^ acc[0][5][1] ^ acc[1][6][2] ^ acc[1][7][3]);
#else
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int nf = 0; nf < 8; ++nf) x ^= (u32)(acc[i][nf][0] ^ acc[i][nf][1] ^ acc[i][nf][2] ^ acc[i][nf][3]);
#endif
#if ABL_EPI == 1
        xs = rotl13(xs) ^ x;
#else
        u32* slot = &jps[(blk & 15) * 256 + tid];
        *slot = rotl13(*slot) ^ x;
#endif
#endif
    }
#if ABL_NOEPI
    { u32 x = 0;
      #pragma unroll
      for (int i = 0; i < 2; ++i)
          #pragma unroll
          for (int nf = 0; nf < 8; ++nf) x ^= (u32)(acc[i][nf][0] ^ acc[i][nf][1] ^ acc[i][nf][2] ^ acc[i][nf][3]);
      jps[tid] = x; }
#endif

#if ABL_EPI == 1
    jps[tid] = xs;
#endif
    const long long rt = by * 32 + wm * 8 + g;              // row-тайл этого потока
    const long long ct = bx * 8 + wn * 4 + q;               // col-тайл
    if (do_search) {
        u32 msg[16], key[8], hh[8];
        #pragma unroll
        for (int s = 0; s < 16; ++s) msg[s] = jps[s * 256 + tid];
        #pragma unroll
        for (int i = 0; i < 8; ++i) key[i] = skey[i];
        hash64(key, msg, hh);
        if (le256(hh, sbound)) atomicMin(best, (u32)(rt * n_col_tiles + ct));
    } else {
        #pragma unroll
        for (int s = 0; s < 16; ++s) out[(rt * n_col_tiles + ct) * 16 + s] = jps[s * 256 + tid];
    }
}
"""

_fn = None
_key = None


def smem_bytes() -> int:
    return OPTS["STAGES"] * (128 + 128) * 64 + 16 * 256 * 4 + 2 * 128 * 4


def kernel():
    """Скомпилированное ядро для текущих OPTS (пересоздаётся при смене параметров)."""
    global _fn, _key
    key = tuple(sorted(OPTS.items()))
    if _fn is None or key != _key:
        mod = cp.RawModule(code=_SRC, backend="nvrtc",
                           options=(nvrtc_arch(), "--std=c++17") + tuple(f"-D{k}={v}" for k, v in OPTS.items()))
        _fn = mod.get_function("jackpot_v3")
        _fn.max_dynamic_shared_size_bytes = smem_bytes()
        _key = key
    return _fn
