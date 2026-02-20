#include "arena.h"
#include "base.h"
#include "prng.h"

int
main(void)
{
    u64 seeds[2] = {0};
    plat_get_entropy(seeds, sizeof(seeds));

    prng_state rng = {};
    prng_seed_r(&rng, seeds[0], seeds[1]);

    for(u32 i = 0; i < 10; i++)
    {
        printf("%f\n", prng_rand_norm_r(&rng));
    }

    mem_arena* perm_arena = arena_create(GiB(1), MiB(1));

    while(1)
    {
        arena_push(perm_arena, MiB(16), false);
        getc(stdin);
    }

    arena_destroy(perm_arena);

    return 0;
}
