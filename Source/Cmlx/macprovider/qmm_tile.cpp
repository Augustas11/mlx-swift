// Copyright © 2026 Malibu. macprovider fork addition (#2451).
//
// C entry point for the core's dense qmm row-tile choice. See
// include/mlx_qmm_tile.h. No upstream mlx or mlx-c file is modified.

#include "mlx/backend/metal/metal.h"
#include "mlx_qmm_tile.h"

extern "C" int
mlx_qmm_t_block_rows(bool affine_bfloat16, int group_size, int bits, int m) {
  return mlx::core::metal::qmm_t_block_rows(
      affine_bfloat16, group_size, bits, m);
}
