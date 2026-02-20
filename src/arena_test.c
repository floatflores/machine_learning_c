#include "arena_test.h"

u32*
child_task(mem_arena* target_arena, u32 count, u32 val)
{
    mem_arena_temp scratch = arena_scratch_get(&target_arena, 1);

    u32* junk = PUSH_ARRAY_ARENA(scratch.arena, u32, 1000);
    for(u32 i = 0; i < 1000; i++)
        junk[i] = 0xDEADBEEF;

    u32* result = PUSH_ARRAY_ARENA(target_arena, u32, count);
    for(u32 i = 0; i < count; i++)
        result[i] = val;

    arena_scratch_release(scratch);
    return result;
}

void
test_double_scratch()
{
    printf("Running Double Scratch Test...\n");

    mem_arena_temp parent_scratch = arena_scratch_get(NULL, 0);

    u32 n = 100;
    u32* parent_data = PUSH_ARRAY_ARENA(parent_scratch.arena, u32, n);
    for(u32 i = 0; i < n; i++)
        parent_data[i] = i;

    printf("  Parent data allocated on Arena %p\n", (void*)parent_scratch.arena);

    u32* child_result = child_task(parent_scratch.arena, n, 999);

    b32 data_ok = true;
    for(u32 i = 0; i < n; i++)
    {
        if(parent_data[i] != i)
        {
            data_ok = false;
            break;
        }
    }

    b32 result_ok = true;
    for(u32 i = 0; i < n; i++)
    {
        if(child_result[i] != 999)
        {
            result_ok = false;
            break;
        }
    }

    if(data_ok && result_ok)
    {
        printf("  SUCCESS: Parent data is intact and child result is correct!\n");
    }
    else
    {
        printf("  FAILED: Memory corruption detected!\n");
        if(!data_ok)
            printf("    -> Parent data was overwritten.\n");
        if(!result_ok)
            printf("    -> Child result is invalid.\n");
    }

    arena_scratch_release(parent_scratch);
}
