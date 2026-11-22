/* Copyright © 2026 Apple Inc. */

#ifndef MLX_SMALL_M_ROUTE_H
#define MLX_SMALL_M_ROUTE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_small_m_route Small-row quantized matmul route
 * The route affine quantized matmul takes for small row counts on non-NAX
 * GPUs (see mlx/small_m_route.h):
 *   0 serial_only    process-wide only, never a scope;
 *   1 row_invariant  a row equals itself in any batch;
 *   2 stock_order    a 2-row product equals each lone row.
 * Each op captures the route when it is added to the graph, so a scope only
 * needs to be set while the graph is built.
 */
/**@{*/

/**
 * Set this thread's route scope. `route` must be 1 or 2; any other value
 * reports an error through the mlx-c error handler and returns non-zero,
 * leaving the scope unchanged.
 */
int mlx_set_small_m_route_scope(int route);
/**
 * Clear this thread's route scope so ops take the process route again.
 */
int mlx_clear_small_m_route_scope(void);
/**
 * This thread's route scope, or -1 when none is set.
 */
int mlx_small_m_route_scope(int* res);
/**
 * The process route, read once from MLX_METAL_SMALL_M_ROW_INVARIANT.
 */
int mlx_process_small_m_route(int* res);
/**
 * The route ops built on this thread capture: the scope, else the process
 * route.
 */
int mlx_current_small_m_route(int* res);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
