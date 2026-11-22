/* Copyright © 2026 Malibu. macprovider fork addition (#2451). */

#ifndef MLX_QMM_TILE_H
#define MLX_QMM_TILE_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The row tile (BM) the core's transposed dense qmm kernel takes on a GPU
 * without NAX for an M-row product (mlx::core::metal::qmm_t_block_rows): 64
 * for affine bfloat16 weights at group size 64 and 4, 6 or 8 bits from 1024
 * rows, else 32. MLX_METAL_QMM_STOCK_TILE=1, read once per process, keeps 32.
 * Tests use it to show the 64-row tile ran.
 */
int mlx_qmm_t_block_rows(bool affine_bfloat16, int group_size, int bits, int m);

#ifdef __cplusplus
}
#endif

#endif
