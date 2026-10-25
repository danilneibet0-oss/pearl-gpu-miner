"""
blake3_cuda.py — общее device-ядро BLAKE3 (compress, хеш 64-байтного сообщения) для CUDA-источников проекта.
Подставляется в начало исходников pearl_gpu и pearl_pipeline.
"""

CORE = r"""
typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __constant__ u32 IV[8] = {0x6A09E667u,0xBB67AE85u,0x3C6EF372u,0xA54FF53Au,
                                     0x510E527Fu,0x9B05688Cu,0x1F83D9ABu,0x5BE0CD19u};
__device__ __constant__ int MSG_PERM[16] = {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8};

#define F_CHUNK_START 1u
#define F_CHUNK_END   2u
#define F_PARENT      4u
#define F_ROOT        8u
#define F_KEYED      16u

__device__ __forceinline__ u32 rotr(u32 x, int n) { return (x >> n) | (x << (32 - n)); }

#define G(a, b, c, d, mx, my)                  \
    a = a + b + (mx); d = rotr(d ^ a, 16);     \
    c = c + d;        b = rotr(b ^ c, 12);     \
    a = a + b + (my); d = rotr(d ^ a, 8);      \
    c = c + d;        b = rotr(b ^ c, 7);

__device__ __forceinline__ void round_fn(u32 s[16], const u32 m[16]) {
    G(s[0], s[4], s[8],  s[12], m[0],  m[1]);
    G(s[1], s[5], s[9],  s[13], m[2],  m[3]);
    G(s[2], s[6], s[10], s[14], m[4],  m[5]);
    G(s[3], s[7], s[11], s[15], m[6],  m[7]);
    G(s[0], s[5], s[10], s[15], m[8],  m[9]);
    G(s[1], s[6], s[11], s[12], m[10], m[11]);
    G(s[2], s[7], s[8],  s[13], m[12], m[13]);
    G(s[3], s[4], s[9],  s[14], m[14], m[15]);
}

// compress -> первые 8 слов (chaining value / 32 байта выхода)
__device__ void compress_cv(const u32 cv[8], const u32 blk[16], u64 counter, u32 blen, u32 flags, u32 out[8]) {
    u32 s[16], m[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = cv[i];
    #pragma unroll
    for (int i = 0; i < 4; ++i) s[8 + i] = IV[i];
    s[12] = (u32)counter; s[13] = (u32)(counter >> 32); s[14] = blen; s[15] = flags;
    #pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = blk[i];
    #pragma unroll
    for (int r = 0; r < 7; ++r) {
        round_fn(s, m);
        if (r < 6) {
            u32 t[16];
            #pragma unroll
            for (int i = 0; i < 16; ++i) t[i] = m[MSG_PERM[i]];
            #pragma unroll
            for (int i = 0; i < 16; ++i) m[i] = t[i];
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = s[i] ^ s[i + 8];
}

// ---- хеш одного 64-байтного сообщения ----
__device__ __forceinline__ void hash64(const u32 key[8], const u32 msg[16], u32 out[8]) {
    compress_cv(key, msg, 0, 64, F_CHUNK_START | F_CHUNK_END | F_ROOT | F_KEYED, out);
}
__device__ __forceinline__ void hash64_plain(const u32 msg[16], u32 out[8]) {
    compress_cv(IV, msg, 0, 64, F_CHUNK_START | F_CHUNK_END | F_ROOT, out);
}

// h <= bound как 256-битные little-endian числа (слова от старшего к младшему)
__device__ __forceinline__ bool le256(const u32 h[8], const u32* __restrict__ bound) {
    for (int i = 7; i >= 0; --i) {
        const u32 bw = bound[i];
        if (h[i] != bw) return h[i] < bw;
    }
    return true;
}
"""
