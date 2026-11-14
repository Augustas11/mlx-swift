/* Copyright © 2026 Malibu. macprovider fork addition (#2338). */

#ifndef MLX_EVAL_WAIT_H
#define MLX_EVAL_WAIT_H

#include <stddef.h>

#include "mlx/c/array.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * A synchronous eval split in three phases so that the caller can release its
 * eval lock while the device runs:
 *
 *   1. mlx_eval_wait_begin   (eval lock held)     schedule, copy the events
 *   2. mlx_eval_wait_wait    (eval lock NOT held) wait on the copied events
 *   3. mlx_eval_wait_finish  (eval lock held)     mark the outputs available
 *
 * Phase 2 touches no array: it only waits on its own copies of the completion
 * events, so a second thread may use the same arrays (under the lock) at the
 * same time. Every array field write (event, status) happens in phase 1 or 3.
 */
typedef struct mlx_eval_wait_ {
  void* ctx;
} mlx_eval_wait;

/**
 * Phase 1. Evaluates `arrays` the way mlx_eval does, except that work on a GPU
 * stream is scheduled asynchronously. On return `res->ctx` is NULL when there
 * is nothing left to wait for (all outputs available, or the eval ran
 * synchronously because it targets a CPU stream or the asynchronous schedule
 * failed and the stock synchronous eval ran instead). Otherwise `res` must go
 * through phases 2 and 3 and then mlx_eval_wait_free.
 *
 * Returns 0 on success. Returns 1 after reporting an error through the mlx-c
 * error handler, exactly as mlx_eval would; `res->ctx` is then NULL.
 */
int mlx_eval_wait_begin(mlx_eval_wait* res, const mlx_array* arrays, size_t num);

/**
 * Phase 2. Blocks until every copied event is signaled. Call WITHOUT the eval
 * lock. Returns 1 after reporting a device error through the error handler.
 */
int mlx_eval_wait_wait(const mlx_eval_wait w);

/**
 * Phase 3. Detaches the signaled events and marks the outputs available, as the
 * end of a synchronous eval would. Call with the eval lock held, after phase 2.
 */
void mlx_eval_wait_finish(const mlx_eval_wait w);

/** Frees `w`. Safe on a NULL ctx. */
void mlx_eval_wait_free(mlx_eval_wait w);

#ifdef __cplusplus
}
#endif

#endif
