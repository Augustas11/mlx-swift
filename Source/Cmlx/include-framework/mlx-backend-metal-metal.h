#ifdef __cplusplus
// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <string>
#include <unordered_map>
#include <variant>

#include <Cmlx/mlx-api.h>

namespace mlx::core::metal {

/* Check if the Metal backend is available. */
MLX_API bool is_available();

/* Check if MLX kernels can use the GPU neural accelerators (NAX). */
MLX_API bool is_nax_available();

/** Capture a GPU trace, saving it to an absolute file `path` */
MLX_API void start_capture(std::string path = "");
MLX_API void stop_capture();

/** Get information about the GPU and system settings. */
MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info();

/* Set a custom path to mlx.metallib. Must be called before any MLX operation.
 */
MLX_API void set_metallib_path(const std::string& path);
MLX_API const std::string& get_metallib_path();

/* macprovider: internal. The number of K partial sums (WK) the small-row
 * qmv_mma kernels use on the given small-row route (see small_m_route.h), on
 * this process's GPU. quantized_matmul captures it per op. */
int qmv_mma_partial_sums(int route);

/* macprovider: internal. The row tile (BM) of the transposed dense qmm kernel
 * (affine_qmm_t, GPUs without NAX) for an M-row product: 64 for affine
 * bfloat16 weights at group size 64 and 4, 6 or 8 bits from 1024 rows, else
 * the stock 32 (see qmm in quantized.cpp). MLX_METAL_QMM_STOCK_TILE=1, read
 * once per process, keeps 32 for every product. */
MLX_API int
qmm_t_block_rows(bool affine_bfloat16, int group_size, int bits, int M);

} // namespace mlx::core::metal
#endif
