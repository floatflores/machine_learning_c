#ifndef PRNG_H
#define PRNG_H

#include "base.h"

#define PI 3.14159265358979

typedef struct
{
    u64 state;
    u64 inc;

    f32 prev_norm;
} prng_state;

void prng_seed_r(prng_state* rng, u64 init_state, u64 init_seq);
void prng_seed(u64 init_state, u64 init_seq);

u32 prng_rand_r(prng_state* rng);
u32 prng_rand(void);

f32 prng_randf_r(prng_state* rng);
f32 prng_randf(void);

f32 prng_rand_norm_r(prng_state* rng);
f32 prng_rand_norm(void);

void plat_get_entropy(void* data, u32 size);

// #define PRNG_IMPLEMENTATION
#ifdef PRNG_IMPLEMENTATION

static prng_state prng_state_ = {0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL, NAN};

void
prng_seed_r(prng_state* rng, u64 init_state, u64 init_seq)
{
    rng->state = 0U;
    rng->inc = (init_seq << 1u) | 1u;
    prng_rand_r(rng);
    rng->state += init_state;
    prng_rand_r(rng);

    rng->prev_norm = NAN;
}

void
prng_seed(u64 init_state, u64 init_seq)
{
    prng_seed_r(&prng_state_, init_state, init_seq);
}

u32
prng_rand_r(prng_state* rng)
{
    u64 oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    u32 xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    u32 rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

u32
prng_rand(void)
{
    return prng_rand_r(&prng_state_);
}

f32
prng_randf_r(prng_state* rng)
{
    return (f32)prng_rand_r(rng) / (f32)UINT32_MAX;
}

f32
prng_randf(void)
{
    return prng_randf_r(&prng_state_);
}

f32
prng_rand_norm_r(prng_state* rng)
{
    if(!isnanf(rng->prev_norm))
    {
        f32 out = rng->prev_norm;
        rng->prev_norm = NAN;
        return out;
    }

    f32 u1 = 0.0f;
    do
    {
        u1 = prng_randf_r(rng);
    } while(u1 == 0.0f);

    f32 u2 = prng_randf_r(rng);

    f32 mag = sqrtf(-2 * logf(u1));

    f32 z0 = mag * cosf(2.0f * PI * u2);
    f32 z1 = mag * sinf(2.0f * PI * u2);

    rng->prev_norm = z1;

    return z0;
}

f32
prng_rand_norm(void)
{
    return prng_rand_norm_r(&prng_state_);
}

#if defined(_WIN32)

#include <Windows.h>
#include <bcrypt.h>

void
plat_get_entropy(void* data, u32 size)
{
    BCryptGenRandom(NULL, data, size, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
}

#elif defined(__linux__)

#include <sys/random.h>

void
plat_get_entropy(void* data, u32 size)
{
    getentropy(data, size);
}
#endif

#endif // PRNG_IMPLEMENTATION

#endif // !PRNG_H
