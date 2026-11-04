/* sha256.h -- SHA-256 (FIPS 180-4), integer arithmetic only.  Used by capcert to bind the survivor files to the
 * completion stamps (mesh.done, mesh2.done): the chain checker recomputes the hashes with Python's hashlib.
 * Tested against hashlib in tests/test_capcert_stamps.py.  Not part of any bound; a wrong hash makes the chain
 * check fail, it cannot make a false bound pass. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct { uint32_t h[8]; uint64_t len; unsigned char buf[64]; size_t nbuf; } Sha256;

static const uint32_t SHA_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

#define SHA_ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

static void sha256_block(Sha256 *s, const unsigned char *p) {
    uint32_t w[64], a, b, c, d, e, f, g, h;
    for (int i = 0; i < 16; i++)
        w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | (uint32_t)p[4 * i + 3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = SHA_ROR(w[i - 15], 7) ^ SHA_ROR(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = SHA_ROR(w[i - 2], 17) ^ SHA_ROR(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    a = s->h[0]; b = s->h[1]; c = s->h[2]; d = s->h[3]; e = s->h[4]; f = s->h[5]; g = s->h[6]; h = s->h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = SHA_ROR(e, 6) ^ SHA_ROR(e, 11) ^ SHA_ROR(e, 25), ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + SHA_K[i] + w[i];
        uint32_t S0 = SHA_ROR(a, 2) ^ SHA_ROR(a, 13) ^ SHA_ROR(a, 22), mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    s->h[0] += a; s->h[1] += b; s->h[2] += c; s->h[3] += d; s->h[4] += e; s->h[5] += f; s->h[6] += g; s->h[7] += h;
}

static void sha256_init(Sha256 *s) {
    static const uint32_t h0[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    memcpy(s->h, h0, sizeof h0); s->len = 0; s->nbuf = 0;
}

static void sha256_update(Sha256 *s, const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    s->len += n;
    if (s->nbuf) {
        size_t k = 64 - s->nbuf; if (k > n) k = n;
        memcpy(s->buf + s->nbuf, p, k); s->nbuf += k; p += k; n -= k;
        if (s->nbuf == 64) { sha256_block(s, s->buf); s->nbuf = 0; }
    }
    while (n >= 64) { sha256_block(s, p); p += 64; n -= 64; }
    if (n) { memcpy(s->buf, p, n); s->nbuf = n; }
}

/* hex digest (65 bytes with the terminating 0) */
static void sha256_hex(Sha256 *s, char *hex) {
    uint64_t bits = s->len * 8; unsigned char pad[72]; size_t np = 0;
    pad[np++] = 0x80;
    while ((s->nbuf + np) % 64 != 56) pad[np++] = 0;
    for (int i = 7; i >= 0; i--) pad[np++] = (unsigned char)(bits >> (8 * i));
    uint64_t len = s->len; sha256_update(s, pad, np); s->len = len;
    for (int i = 0; i < 8; i++) snprintf(hex + 8 * i, 9, "%08x", s->h[i]);
}

/* SHA-256 of a whole file; *nbytes gets its size.  Returns 0 on success. */
static int sha256_file(const char *fn, char *hex, long long *nbytes) {
    FILE *f = fopen(fn, "rb"); if (!f) return -1;
    Sha256 s; sha256_init(&s); static unsigned char blk[1 << 16]; size_t k; long long tot = 0;
    while ((k = fread(blk, 1, sizeof blk, f)) > 0) { sha256_update(&s, blk, k); tot += (long long)k; }
    int err = ferror(f); fclose(f);
    if (err) return -1;
    sha256_hex(&s, hex); if (nbytes) *nbytes = tot; return 0;
}
