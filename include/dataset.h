#ifndef DATASET_H
#define DATASET_H

#include "arena.h"
#include "base.h"
#include "matrix.h"

matrix* dataset_load_csv(mem_arena* target,
                         const char* filepath, char delim);

matrix* dataset_load_mat(mem_arena* target, u32 rows,
                         u32 cols, const char* filepath);

// #define DATASET_IMPLEMENTATION
#ifdef DATASET_IMPLEMENTATION

matrix*
dataset_load_csv(mem_arena* target, const char* filepath,
                 char delim)
{
    matrix* result = NULL;

    FILE* file = fopen(filepath, "rb");
    if(!file)
    {
        printf("[ERROR] Failed to open CSV: %s\n",
               filepath);
        return result;
    }

    fseek(file, 0, SEEK_END);
    u64 file_size = (u64)ftell(file);
    fseek(file, 0, SEEK_SET);

    if(file_size == 0)
    {
        printf("[INFO] Current file size is 0: %s\n",
               filepath);
        fclose(file);
        return result;
    }

    u64 reserve_size = file_size + GiB(1);
    u64 commit_size = file_size + MiB(64);
    mem_arena* scratch = arena_create(reserve_size,
                                      commit_size);
    char* file_data = PUSH_ARRAY_ARENA(scratch, char,
                                       file_size + 1);

    fread(file_data, 1, file_size, file);
    file_data[file_size] = '\0';

    fclose(file);

    u64 rows = 0;
    u64 cols = 0;
    b32 first_row = true;
    for(u64 i = 0; i < file_size; i++)
    {
        if(file_data[i] == '\n')
        {
            rows++;
            first_row = false;
        }
        else if(file_data[i] == delim && first_row)
        {
            cols++;
        }
    }
    if(file_data[file_size - 1] != '\n')
        rows++;
    cols++;

    result = mat_create(target, rows, cols);

    char* cursor = file_data;
    char* next_cursor = NULL;
    u32 index = 0;
    u32 total_elements = rows * cols;

    while(index < total_elements && *cursor != '\0')
    {
        result->data[index++] = strtof(cursor,
                                       &next_cursor);

        if(cursor == next_cursor)
        {
            cursor++;
        }
        else
        {
            cursor = next_cursor;
            while(*cursor == delim || *cursor == '\n'
                  || *cursor == '\r')
            {
                cursor++;
            }
        }
    }

    arena_destroy(scratch);

    return result;
}

matrix*
dataset_load_mat(mem_arena* target, u32 rows, u32 cols,
                 const char* filepath)
{
    matrix* result = mat_create(target, rows, cols);

    FILE* file = fopen(filepath, "rb");
    fseek(file, 0, SEEK_END);
    u64 file_size = (u64)ftell(file);
    fseek(file, 0, SEEK_SET);

    file_size = MIN(file_size, sizeof(f32) * rows * cols);

    fread(result->data, 1, file_size, file);

    fclose(file);

    return result;
}

#endif // DATASET_IMPLEMENTATION

#endif // !DATASET_H
