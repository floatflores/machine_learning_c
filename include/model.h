#ifndef MODEL_H
#define MODEL_H

#include "base.h"
#include "matrix.h"

typedef enum
{
    MV_FLAG_NONE = 0,

    MV_FLAG_REQUIRES_GRAD  = (1 << 0),
    MV_FLAG_PARAMETER      = (1 << 1),
    MV_FLAG_INPUT          = (1 << 2),
    MV_FLAG_OUTPUT         = (1 << 3),
    MV_FLAG_DESIRED_OUTPUT = (1 << 4),
    MV_FLAG_COST           = (1 << 5),
} model_var_flags;

typedef enum
{
    MV_OP_NULL,
    MV_OP_CREATE,

    _MV_OP_UNARY_START,

    MV_OP_RELU,
    MV_OP_SOFTMAX,

    _MV_OP_BINARY_START,

    MV_OP_ADD,
    MV_OP_SUB,

    MV_OP_MATMUL,
    MV_OP_CROSS_ENTROPY
} model_var_ops;

#define MODEL_VAR_MAX_INPUTS 2
#define MV_NUM_INPUTS(op)                                  \
    ((op) < _MV_OP_UNARY_START                             \
         ? 0                                               \
         : ((op) < _MV_OP_BINARY_START ? 1 : 2))

typedef struct model_var
{
    u32 index;
    u32 flags;

    matrix* val;
    matrix* grad;

    model_var_ops op;
    struct model_var* inputs[MODEL_VAR_MAX_INPUTS];
} model_var;

typedef struct
{
    model_var** vars;
    u32 size;
} model_program;

typedef struct
{
    u32 num_vars;

    model_var* input;
    model_var* output;
    model_var* desired_output;
    model_var* cost;

    model_program forward_prog;
    model_program cost_prog;
} model_context;

typedef struct
{
    matrix* train_images;
    matrix* train_labels;

    matrix* test_images;
    matrix* test_labels;

    u32 epoch;
    u32 batch_size;
    f32 learning_rate;
} model_training_desc;

model_var* mv_create(mem_arena* arena, model_context* model,
                     u32 rows, u32 cols, u32 flags);

model_var* mv_relu(mem_arena* arena, model_context* model,
                   model_var* input, u32 flags);

model_var* mv_softmax(mem_arena* arena,
                      model_context* model,
                      model_var* input, u32 flags);

model_var* mv_add(mem_arena* arena, model_context* model,
                  model_var* a, model_var* b, u32 flags);

model_var* mv_sub(mem_arena* arena, model_context* model,
                  model_var* a, model_var* b, u32 flags);

model_var* mv_matmul(mem_arena* arena, model_context* model,
                     model_var* a, model_var* b, u32 flags);

model_var* mv_cross_entropy(mem_arena* arena,
                            model_context* model,
                            model_var* p, model_var* q,
                            u32 flags);

model_var* _mv_unary_impl(mem_arena* arena,
                          model_context* model,
                          model_var* input, u32 rows,
                          u32 cols, model_var_ops op,
                          u32 flags);

model_var* _mv_binary_impl(mem_arena* arena,
                           model_context* model,
                           model_var* a, model_var* b,
                           u32 rows, u32 cols,
                           model_var_ops op, u32 flags);

model_program model_prog_create(mem_arena* arena,
                                model_context* model,
                                model_var* out_var);

void model_prog_compute(model_program* prog);

void model_prog_compute_grads(model_program* prog);

model_context* model_create(mem_arena* arena);

void model_compile(mem_arena* arena, model_context* model);

void model_feedforward(model_context* model);
void model_train(model_context* model,
                 const model_training_desc* training_desc);

// #define MODEL_IMPLEMENTATION
#ifdef MODEL_IMPLEMENTATION

model_var*
mv_create(mem_arena* arena, model_context* model, u32 rows,
          u32 cols, u32 flags)
{
    model_var* out = PUSH_STRUCT_ARENA(arena, model_var);

    out->index = model->num_vars++;
    out->flags = flags;
    out->op    = MV_OP_CREATE;
    out->val   = mat_create(arena, rows, cols);

    if(flags & MV_FLAG_REQUIRES_GRAD)
    {
        out->grad = mat_create(arena, rows, cols);
    }
    if(flags & MV_FLAG_INPUT)
    {
        model->input = out;
    }
    if(flags & MV_FLAG_OUTPUT)
    {
        model->output = out;
    }
    if(flags & MV_FLAG_DESIRED_OUTPUT)
    {
        model->desired_output = out;
    }
    if(flags & MV_FLAG_COST)
    {
        model->cost = out;
    }

    return out;
}

model_var*
_mv_unary_impl(mem_arena* arena, model_context* model,
               model_var* input, u32 rows, u32 cols,
               model_var_ops op, u32 flags)
{
    if(input->flags & MV_FLAG_REQUIRES_GRAD)
    {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }

    model_var* out = mv_create(arena, model, rows, cols,
                               flags);

    out->op        = op;
    out->inputs[0] = input;

    return out;
}

model_var*
_mv_binary_impl(mem_arena* arena, model_context* model,
                model_var* a, model_var* b, u32 rows,
                u32 cols, model_var_ops op, u32 flags)
{
    if((a->flags & MV_FLAG_REQUIRES_GRAD)
       || (b->flags & MV_FLAG_REQUIRES_GRAD))
    {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }

    model_var* out = mv_create(arena, model, rows, cols,
                               flags);

    out->op = op;

    out->inputs[0] = a;
    out->inputs[1] = b;

    return out;
}

model_var*
mv_relu(mem_arena* arena, model_context* model,
        model_var* input, u32 flags)
{
    return _mv_unary_impl(
        arena, model, input, input->val->rows,
        input->val->cols, MV_OP_RELU, flags);
}

model_var*
mv_softmax(mem_arena* arena, model_context* model,
           model_var* input, u32 flags)
{
    return _mv_unary_impl(
        arena, model, input, input->val->rows,
        input->val->cols, MV_OP_SOFTMAX, flags);
}

model_var*
mv_add(mem_arena* arena, model_context* model, model_var* a,
       model_var* b, u32 flags)
{
    if(a->val->rows != b->val->rows
       || a->val->cols != b->val->cols)
    {
        return NULL;
    }
    return _mv_binary_impl(arena, model, a, b, a->val->rows,
                           a->val->cols, MV_OP_ADD, flags);
}

model_var*
mv_sub(mem_arena* arena, model_context* model, model_var* a,
       model_var* b, u32 flags)
{
    if(a->val->rows != b->val->rows
       || a->val->cols != b->val->cols)
    {
        return NULL;
    }
    return _mv_binary_impl(arena, model, a, b, a->val->rows,
                           a->val->cols, MV_OP_SUB, flags);
}

model_var*
mv_matmul(mem_arena* arena, model_context* model,
          model_var* a, model_var* b, u32 flags)
{
    if(a->val->cols != b->val->rows)
    {
        return NULL;
    }

    return _mv_binary_impl(arena, model, a, b, a->val->rows,
                           b->val->cols, MV_OP_MATMUL,
                           flags);
}

model_var*
mv_cross_entropy(mem_arena* arena, model_context* model,
                 model_var* p, model_var* q, u32 flags)
{
    if(p->val->rows != q->val->rows
       || p->val->cols != q->val->cols)
    {
        return NULL;
    }
    return _mv_binary_impl(arena, model, p, q, p->val->rows,
                           p->val->cols,
                           MV_OP_CROSS_ENTROPY, flags);
}

model_program
model_prog_create(mem_arena* arena, model_context* model,
                  model_var* out_var)
{
    mem_arena_temp scratch = arena_scratch_get(&arena, 1);

    b8* visited = PUSH_ARRAY_ARENA(scratch.arena, b8,
                                   model->num_vars);

    u32 stack_size    = 0;
    u32 out_size      = 0;
    model_var** stack = PUSH_ARRAY_ARENA(
        scratch.arena, model_var*, model->num_vars);
    model_var** out = PUSH_ARRAY_ARENA(
        scratch.arena, model_var*, model->num_vars);

    stack[stack_size++] = out_var;

    while(stack_size > 0)
    {
        model_var* cur = stack[--stack_size];

        if(cur->index >= model->num_vars)
        {
            continue;
        }
        if(visited[cur->index])
        {
            if(out_size < model->num_vars)
            {
                out[out_size++] = cur;
            }
            continue;
        }
        visited[cur->index] = true;
        if(stack_size < model->num_vars)
        {
            stack[stack_size++] = cur;
        }

        u32 num_inputs = MV_NUM_INPUTS(cur->op);
        for(u32 i = 0; i < num_inputs; i++)
        {
            model_var* input = cur->inputs[i];
            if(input->index >= model->num_vars
               || visited[input->index])
            {
                continue;
            }

            for(u32 j = 0; j < stack_size; j++)
            {
                if(stack[j] == input)
                {
                    for(u32 k = j; k < stack_size - 1; k++)
                    {
                        stack[k] = stack[k + 1];
                    }
                    stack_size--;
                }
            }
            stack[stack_size++] = input;
        }
    }

    model_program prog = {.size = out_size,
                          .vars = PUSH_ARRAY_ARENA_NZ(
                              arena, model_var*, out_size)};
    memcpy(prog.vars, out, sizeof(model_var*) * out_size);

    arena_scratch_release(scratch);

    return prog;
}

void
model_prog_compute(model_program* prog)
{
    for(u32 i = 0; i < prog->size; i++)
    {
        model_var* cur = prog->vars[i];

        model_var* a = cur->inputs[0];
        model_var* b = cur->inputs[1];

        switch(cur->op)
        {
            case MV_OP_NULL:
            case MV_OP_CREATE:
                break;
            case _MV_OP_UNARY_START:
                break;
            case MV_OP_RELU:
            {
                mat_relu(cur->val, a->val);
            }
            break;
            case MV_OP_SOFTMAX:
            {
                mat_softmax(cur->val, a->val);
            }
            break;
            case _MV_OP_BINARY_START:
                break;
            case MV_OP_ADD:
            {
                mat_add(cur->val, a->val, b->val);
            }
            break;
            case MV_OP_SUB:
            {
                mat_sub(cur->val, a->val, b->val);
            }
            break;
            case MV_OP_MATMUL:
            {
                mat_mul(cur->val, a->val, b->val, 1, 0, 0);
            }
            break;
            case MV_OP_CROSS_ENTROPY:
            {
                mat_cross_entropy(cur->val, a->val, b->val);
            }
            break;
        }
    }
}

void
model_prog_compute_grads(model_program* prog)
{
    for(u32 i = 0; i < prog->size; i++)
    {
        model_var* cur = prog->vars[i];

        if((cur->flags & MV_FLAG_REQUIRES_GRAD)
           != MV_FLAG_REQUIRES_GRAD)
        {
            continue;
        }

        if(cur->flags & MV_FLAG_PARAMETER)
        {
            continue;
        }
        mat_clear(cur->grad);
    }

    mat_fill(prog->vars[prog->size - 1]->grad, 1.0f);

    for(i64 i = (u64)prog->size - 1; i >= 0; i--)
    {
        model_var* cur = prog->vars[i];

        model_var* a = cur->inputs[0];
        model_var* b = cur->inputs[1];

        u32 num_inputs = MV_NUM_INPUTS(cur->op);

        if(num_inputs == 1
           && (a->flags & MV_FLAG_REQUIRES_GRAD)
               != MV_FLAG_REQUIRES_GRAD)
        {
            continue;
        }
        if(num_inputs == 2
           && (a->flags & MV_FLAG_REQUIRES_GRAD)
               != MV_FLAG_REQUIRES_GRAD
           && (b->flags & MV_FLAG_REQUIRES_GRAD)
               != MV_FLAG_REQUIRES_GRAD)

        {
            continue;
        }

        switch(cur->op)
        {
            case MV_OP_NULL:
            case MV_OP_CREATE:
                break;
            case _MV_OP_UNARY_START:
                break;
            case MV_OP_RELU:
            {
                mat_relu_add_grad(a->grad, a->val,
                                  cur->grad);
            }
            break;
            case MV_OP_SOFTMAX:
            {
                mat_softmax_add_grad(a->grad, cur->val,
                                     cur->grad);
            }
            break;
            case _MV_OP_BINARY_START:
                break;
            case MV_OP_ADD:
            {
                if(a->flags & MV_FLAG_REQUIRES_GRAD)
                {
                    mat_add(a->grad, a->grad, cur->grad);
                }

                if(b->flags & MV_FLAG_REQUIRES_GRAD)
                {
                    mat_add(b->grad, b->grad, cur->grad);
                }
            }
            break;
            case MV_OP_SUB:
            {
                if(a->flags & MV_FLAG_REQUIRES_GRAD)
                {
                    mat_sub(a->grad, a->grad, cur->grad);
                }

                if(b->flags & MV_FLAG_REQUIRES_GRAD)
                {
                    mat_sub(b->grad, b->grad, cur->grad);
                }
            }
            break;
            case MV_OP_MATMUL:
            {
                if(a->flags & MV_FLAG_REQUIRES_GRAD)
                {
                    mat_mul(a->grad, cur->grad, b->val,
                            false, 0, 1);
                }

                if(b->flags & MV_FLAG_REQUIRES_GRAD)
                {
                    mat_mul(b->grad, a->val, cur->grad,
                            false, 1, 0);
                }
            }
            break;
            case MV_OP_CROSS_ENTROPY:
            {
                model_var* p = a;
                model_var* q = b;

                mat_cross_entropy_add_grad(p->grad, q->grad,
                                           p->val, q->val,
                                           cur->grad);
            }
            break;
        }
    }
}

model_context*
model_create(mem_arena* arena)
{
    model_context* model = PUSH_STRUCT_ARENA(arena,
                                             model_context);

    return model;
}

void
model_compile(mem_arena* arena, model_context* model)
{
    if(model->output != NULL)
    {
        model->forward_prog = model_prog_create(
            arena, model, model->output);
    }

    if(model->cost != NULL)
    {
        model->cost_prog = model_prog_create(arena, model,
                                             model->cost);
    }
}

void
model_feedforward(model_context* model)
{
    model_prog_compute(&model->forward_prog);
}

void
model_train(model_context* model,
            const model_training_desc* training_desc)
{
    matrix* train_images = training_desc->train_images;
    matrix* train_labels = training_desc->train_labels;
    matrix* test_images  = training_desc->test_images;
    matrix* test_labels  = training_desc->test_labels;

    u32 num_examples = train_images->rows;
    u32 input_size   = train_images->cols;
    u32 output_size  = train_labels->cols;
    u32 num_tests    = test_images->rows;
    u32 num_batches  = num_examples
        / training_desc->batch_size;

    mem_arena_temp scratch = arena_scratch_get(NULL, 0);

    u32* training_order = PUSH_ARRAY_ARENA_NZ(
        scratch.arena, u32, num_examples);
    for(u32 i = 0; i < num_examples; i++)
    {
        training_order[i] = i;
    }

    for(u32 epoch = 0; epoch < training_desc->epoch;
        epoch++)
    {
        for(u32 i = 0; i < num_examples; i++)
        {
            u32 a = prng_rand() % num_examples;
            u32 b = prng_rand() % num_examples;

            u32 tmp = training_order[b];

            training_order[b] = training_order[a];
            training_order[a] = tmp;
        }

        for(u32 batch = 0; batch < num_batches; batch++)
        {
            for(u32 i = 0; i < model->cost_prog.size; i++)
            {
                model_var* cur = model->cost_prog.vars[i];

                if(cur->flags & MV_FLAG_PARAMETER)
                {
                    mat_clear(cur->grad);
                }
            }

            f32 avg_cost = 0.0f;
            for(u32 i = 0; i < training_desc->batch_size;
                i++)
            {
                u32 order_index = batch
                        * training_desc->batch_size
                    + i;
                u32 index = training_order[order_index];

                memcpy(model->input->val->data,
                       train_images->data
                           + index * input_size,
                       sizeof(f32) * input_size);

                memcpy(model->desired_output->val->data,
                       train_labels->data
                           + index * output_size,
                       sizeof(f32) * output_size);

                model_prog_compute(&model->cost_prog);
                model_prog_compute_grads(&model->cost_prog);

                avg_cost += mat_sum(model->cost->val);
            }
            avg_cost /= (f32)training_desc->batch_size;

            for(u32 i = 0; i < model->cost_prog.size; i++)
            {
                model_var* cur = model->cost_prog.vars[i];

                if((cur->flags & MV_FLAG_PARAMETER)
                   != MV_FLAG_PARAMETER)
                {
                    continue;
                }
                mat_scale(cur->grad,
                          training_desc->learning_rate
                              / training_desc->batch_size);
                mat_sub(cur->val, cur->val, cur->grad);
            }
            printf("Epoch %2d / %2d, Batch %4d / %4d, "
                   "Average Cost: %.4f\r",
                   epoch + 1, training_desc->epoch,
                   batch + 1, num_batches, avg_cost);
        }
        printf("\n");

        u32 num_corrects = 0;
        f32 avg_cost     = 0.0f;
        for(u32 i = 0; i < num_tests; i++)
        {
            memcpy(model->input->val->data,
                   test_images->data + i * input_size,
                   sizeof(f32) * input_size);

            memcpy(model->desired_output->val->data,
                   test_labels->data + i * output_size,
                   sizeof(f32) * output_size);

            model_prog_compute(&model->cost_prog);

            avg_cost += mat_sum(model->cost->val);
            num_corrects += mat_argmax(model->output->val)
                == mat_argmax(model->desired_output->val);
        }
        avg_cost /= (f32)num_tests;
        printf("Test Completed. Accuracy %5d / %d "
               "(%.1f%%), Average Cost: %.4f\n",
               num_corrects, num_tests,
               (f32)num_corrects / num_tests * 100.0f,
               avg_cost);
    }

    arena_scratch_release(scratch);
}

#endif // MODEL_IMPLEMENTATION

#endif // !MODEL_H
