#ifndef ARENA_TEST_H

#include "arena.h"
#include "base.h"

u32* child_task(mem_arena* target_arena, u32 count, u32 val);
void test_double_scratch();

#endif // !ARENA_TEST_H
