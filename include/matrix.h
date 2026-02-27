#ifndef MATRIX_H
#define MATRIX_H

#include "arena.h"
#include "base.h"
#include "prng.h"

typedef struct
{
    u32 rows;
    u32 cols;
    f32* data;
} matrix;

matrix* mat_create(mem_arena* arena, u32 rows, u32 cols);
void mat_print(matrix* mat);

b32 mat_copy(matrix* dst, matrix* src);
void mat_clear(matrix* mat);
void mat_fill(matrix* mat, f32 x);
void mat_fill_random(matrix* mat, f32 lower, f32 upper);
void mat_scale(matrix* mat, f32 scale);
f32 mat_sum(matrix* mat);
u32 mat_argmax(matrix* mat);
b32 mat_add(matrix* out, const matrix* a, const matrix* b);
b32 mat_sub(matrix* out, const matrix* a, const matrix* b);
b32 mat_mul(matrix* out, const matrix* a, const matrix* b,
            b8 zero_out, b8 transpose_a, b8 transpose_b);
b32 mat_relu(matrix* out, const matrix* in);
b32 mat_softmax(matrix* out, const matrix* in);
b32 mat_cross_entropy(matrix* out, const matrix* p,
                      const matrix* q);
b32 mat_relu_add_grad(matrix* out, const matrix* in,
                      const matrix* grad);
b32 mat_softmax_add_grad(matrix* out,
                         const matrix* softmax_out,
                         const matrix* grad);
b32 mat_cross_entropy_add_grad(matrix* p_grad,
                               matrix* q_grad,
                               const matrix* p,
                               const matrix* q,
                               const matrix* grad);

// #define MATRIX_IMPLEMENTATION
#ifdef MATRIX_IMPLEMENTATION

matrix*
mat_create(mem_arena* arena, u32 rows, u32 cols)
{
    matrix* mat = PUSH_STRUCT_ARENA(arena, matrix);

    mat->rows = rows;
    mat->cols = cols;
    mat->data = PUSH_ARRAY_ARENA(arena, f32,
                                 (u64)rows * cols);

    return mat;
}

void
mat_print(matrix* mat)
{
    printf("Matrix (%u x %u):\n", mat->rows, mat->cols);

    u32 p_rows   = 0;
    b32 huge_mat = false;

    if(mat->rows > 20)
    {
        huge_mat = true;
        p_rows   = 6;
    }
    else
    {
        p_rows = mat->rows;
    }

    for(u32 row = 0; row < p_rows; ++row)
    {
        for(u32 col = 0; col < mat->cols; ++col)
        {
            printf("%8.4f ",
                   mat->data[row * mat->cols + col]);
        }
        printf("\n");
    }

    if(huge_mat)
    {
        printf("......\n");
    }
}

b32
mat_copy(matrix* dst, matrix* src)
{
    if(dst->rows != src->rows || dst->cols != src->cols)
    {
        return false;
    }
    memcpy(dst->data, src->data,
           sizeof(f32) * (u64)dst->rows * dst->cols);

    return true;
}

void
mat_clear(matrix* mat)
{
    memset(mat->data, 0,
           sizeof(f32) * (u64)mat->rows * mat->cols);
}

void
mat_fill(matrix* mat, f32 x)
{
    u64 size = (u64)mat->rows * mat->cols;

    for(u64 i = 0; i < size; i++)
    {
        mat->data[i] = x;
    }
}

void
mat_fill_random(matrix* mat, f32 lower, f32 upper)
{
    u64 size = (u64)mat->rows * mat->cols;

    for(u64 i = 0; i < size; i++)
    {
        mat->data[i] = prng_randf() * (upper - lower)
            + lower;
    }
}

void
mat_scale(matrix* mat, f32 scale)
{
    u64 size = (u64)mat->rows * mat->cols;

    for(u64 i = 0; i < size; i++)
    {
        mat->data[i] *= scale;
    }
}

f32
mat_sum(matrix* mat)
{
    u64 size = (u64)mat->rows * mat->cols;

    f32 sum = 0.0f;
    for(u64 i = 0; i < size; i++)
    {
        sum += mat->data[i];
    }

    return sum;
}

u32
mat_argmax(matrix* mat)
{
    u64 size = (u64)mat->rows * mat->cols;

    u32 max_i = 0;
    for(u64 i = 0; i < size; i++)
    {
        if(mat->data[i] > mat->data[max_i])
        {
            max_i = i;
        }
    }

    return max_i;
}

b32
mat_add(matrix* out, const matrix* a, const matrix* b)
{
    if(a->rows != b->rows || a->cols != b->cols)
    {
        return false;
    }
    if(out->rows != a->rows || out->cols != a->cols)
    {
        return false;
    }

    u64 size = (u64)out->rows * out->cols;
    for(u64 i = 0; i < size; i++)
    {
        out->data[i] = a->data[i] + b->data[i];
    }

    return true;
}

b32
mat_sub(matrix* out, const matrix* a, const matrix* b)
{
    if(a->rows != b->rows || a->cols != b->cols)
    {
        return false;
    }
    if(out->rows != a->rows || out->cols != a->cols)
    {
        return false;
    }

    u64 size = (u64)out->rows * out->cols;
    for(u64 i = 0; i < size; i++)
    {
        out->data[i] = a->data[i] - b->data[i];
    }

    return true;
}

void
_mat_mul_nn(matrix* out, const matrix* a, const matrix* b)
{
    for(u64 i = 0; i < out->rows; i++)
    {
        for(u64 k = 0; k < a->cols; k++)
        {
            f32 val_a = a->data[i * a->cols + k];
            for(u64 j = 0; j < out->cols; j++)
            {
                out->data[i * out->cols + j] += val_a
                    * b->data[k * b->cols + j];
            }
        }
    }
}

void
_mat_mul_nt(matrix* out, const matrix* a, const matrix* b)
{
    for(u64 i = 0; i < out->rows; i++)
    {
        for(u64 j = 0; j < out->cols; j++)
        {
            for(u64 k = 0; k < a->cols; k++)
            {
                out->data[i * out->cols + j]
                    += a->data[i * a->cols + k]
                    * b->data[j * b->cols + k];
            }
        }
    }
}

void
_mat_mul_tn(matrix* out, const matrix* a, const matrix* b)
{
    for(u64 k = 0; k < a->rows; k++)
    {
        for(u64 i = 0; i < out->rows; i++)
        {
            f32 val_a = a->data[k * a->cols + i];
            for(u64 j = 0; j < out->cols; j++)
            {
                out->data[i * out->cols + j] += val_a
                    * b->data[k * b->cols + j];
            }
        }
    }
}

void
_mat_mul_tt(matrix* out, const matrix* a, const matrix* b)
{
    for(u64 k = 0; k < a->cols; k++)
    {
        for(u64 i = 0; i < out->rows; i++)
        {
            f32 val_a = a->data[k * a->cols + i];
            for(u64 j = 0; j < out->cols; j++)
            {
                out->data[i * out->cols + j] += val_a
                    * b->data[j * b->cols + k];
            }
        }
    }
}

b32
mat_mul(matrix* out, const matrix* a, const matrix* b,
        b8 zero_out, b8 transpose_a, b8 transpose_b)
{
    u32 a_rows = transpose_a ? a->cols : a->rows;
    u32 a_cols = transpose_a ? a->rows : a->cols;
    u32 b_rows = transpose_b ? b->cols : b->rows;
    u32 b_cols = transpose_b ? b->rows : b->cols;

    if(a_cols != b_rows)
    {
        return false;
    }
    if(out->rows != a_rows || out->cols != b_cols)
    {
        return false;
    }

    if(zero_out)
    {
        mat_clear(out);
    }

    u32 transpose = (transpose_a << 1) | transpose_b;
    switch(transpose)
    {
        case 0b00:
        {
            _mat_mul_nn(out, a, b);
        }
        break;
        case 0b01:
        {
            _mat_mul_nt(out, a, b);
        }
        break;
        case 0b10:
        {
            _mat_mul_tn(out, a, b);
        }
        break;
        case 0b11:
        {
            _mat_mul_tt(out, a, b);
        }
        break;
    }

    return true;
}

b32
mat_relu(matrix* out, const matrix* in)
{
    if(out->rows != in->rows || out->cols != in->cols)
    {
        return false;
    }

    u64 size = (u64)out->rows * out->cols;
    for(u64 i = 0; i < size; i++)
    {
        out->data[i] = MAX(0, in->data[i]);
    }

    return true;
}

b32
mat_softmax(matrix* out, const matrix* in)
{
    if(out->rows != in->rows || out->cols != in->cols)
    {
        return false;
    }

    u64 size = (u64)out->rows * out->cols;

    f32 sum = 0.0f;
    for(u64 i = 0; i < size; i++)
    {
        out->data[i] = expf(in->data[i]);
        sum += out->data[i];
    }

    mat_scale(out, 1.0f / sum);

    return true;
}

b32
mat_cross_entropy(matrix* out, const matrix* p,
                  const matrix* q)
{
    if(p->rows != q->rows || p->cols != q->cols)
    {
        return false;
    }

    if(out->rows != p->rows || out->cols != p->cols)
    {
        return false;
    }

    u64 size = (u64)out->rows * out->cols;
    for(u64 i = 0; i < size; i++)
    {
        out->data[i] = p->data[i] == 0.0f
            ? 0.0f
            : p->data[i] * -logf(q->data[i]);
    }

    return true;
}

b32
mat_relu_add_grad(matrix* out, const matrix* in,
                  const matrix* grad)
{
    if(out->rows != in->rows || out->cols != in->cols)
    {
        return false;
    }
    if(out->rows != grad->rows || out->cols != grad->cols)
    {
        return false;
    }

    u64 size = (u64)out->rows * out->cols;
    for(u64 i = 0; i < size; i++)
    {
        out->data[i] = in->data[i] > 0.0f ? grad->data[i]
                                          : 0.0f;
    }

    return true;
}

b32
mat_softmax_add_grad(matrix* out, const matrix* softmax_out,
                     const matrix* grad)
{
    if(softmax_out->rows != 1 && softmax_out->cols != 1)
    {
        return false;
    }

    mem_arena_temp scratch = arena_scratch_get(NULL, 0);

    u32 size = MAX(softmax_out->rows, softmax_out->cols);
    matrix* jacobiana = mat_create(scratch.arena, size,
                                   size);

    for(u32 i = 0; i < size; i++)
    {
        for(u32 j = 0; j < size; j++)
        {
            jacobiana->data[i * size + j] = softmax_out
                                                ->data[i]
                * ((i == j) - softmax_out->data[j]);
        }
    }

    mat_mul(out, jacobiana, grad, 0, 0, 0);

    arena_scratch_release(scratch);

    return true;
}

b32
mat_cross_entropy_add_grad(matrix* p_grad, matrix* q_grad,
                           const matrix* p, const matrix* q,
                           const matrix* grad)
{
    if(p->rows != q->rows || p->cols != q->cols)
    {
        return false;
    }

    u64 size = (u64)p->rows * p->cols;

    if(p_grad != NULL)
    {
        if(p_grad->rows != p->rows
           || p_grad->cols != p->cols)
        {
            return false;
        }
        for(u64 i = 0; i < size; i++)
        {
            p_grad->data[i] += -logf(q->data[i])
                * grad->data[i];
        }
    }
    if(q_grad != NULL)
    {
        if(q_grad->rows != p->rows
           || q_grad->cols != p->cols)
        {
            return false;
        }
        for(u64 i = 0; i < size; i++)
        {
            q_grad->data[i] += -p->data[i] / q->data[i]
                * grad->data[i];
        }
    }

    return true;
}
#endif // !MATRIX_IMPLEMENTATION

#endif // !MATRIX_H
