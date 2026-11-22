// Copyright © 2026 Apple Inc.

#if os(macOS)
import Cmlx
import Foundation
import MLX
import XCTest

/// macprovider (#2451): from 1024 rows the core's transposed dense qmm of affine bfloat16
/// weights at group size 64 and 4, 6 or 8 bits takes a 64-row tile instead of the stock 32
/// (`qmm_t_block_rows` in the core's quantized.cpp). The tile must not change any output bit:
/// every row of a 1024..4096-row product on the 64-row tile equals that row computed in 512-row
/// chunks, which stay on the stock tile. 3673 rows (a real prompt length) ends in a partial tile.
/// Shapes: GLM-5.3-Flash's dense MLP (N 4096, K 12288 and N 12288, K 4096), 8 and 4 bits.
/// The core's own doctest "test gpu qmm 64-row tile bit identity" checks the same cases.
class QmmTileTests: XCTestCase {

    /// Mirrors metal::is_nax_available() (the core's device.cpp): gen 17 and up (18 and up for
    /// "p" GPUs) on macOS 26.2 and up. NAX GPUs run qmm_nax, not this tile.
    private func naxGPU() -> Bool {
        let arch = GPU.deviceInfo().architecture
        guard let g = arch.range(of: "_g", options: .backwards) else { return false }
        let digits = arch[g.upperBound...].filter(\.isNumber)
        guard let gen = Int(String(digits)) else { return false }
        guard #available(macOS 26.2, *) else { return false }
        return gen >= (arch.hasSuffix("p") ? 18 : 17)
    }

    func testDenseTileIsBitIdenticalToStockChunks() throws {
        try XCTSkipIf(naxGPU(), "NAX GPU: the dense qmm tile does not run")
        let stock = ProcessInfo.processInfo.environment["MLX_METAL_QMM_STOCK_TILE"] ?? ""
        XCTAssertTrue(
            stock.isEmpty || stock == "0",
            "MLX_METAL_QMM_STOCK_TILE is set: unset it to test the 64-row tile")
        let chunk = 512
        let gs = 64
        let shapes = [(n: 4096, k: 12288), (n: 12288, k: 4096)]
        let rowCounts = [1024, 2048, 3673, 4096]

        MLXRandom.seed(0)
        for (n, k) in shapes {
            for bits in [8, 4] {
                let w = MLXRandom.normal([n, k]).asType(.bfloat16)
                let q = quantized(w, groupSize: gs, bits: bits)
                let xAll = (MLXRandom.normal([rowCounts.last!, k]) / Float(k).squareRoot())
                    .asType(.bfloat16)
                eval(q.wq, q.scales, q.biases!, xAll)
                func qmm(_ v: MLXArray) -> MLXArray {
                    quantizedMatmul(
                        v, q.wq, scales: q.scales, biases: q.biases, transpose: true,
                        groupSize: gs, bits: bits)
                }
                XCTAssertEqual(
                    mlx_qmm_t_block_rows(true, Int32(gs), Int32(bits), Int32(chunk)), 32)
                for m in rowCounts {
                    let what = "N \(n) K \(k) bits \(bits) M \(m)"
                    // The products below take the 64-row tile, so the comparison is not vacuous.
                    XCTAssertEqual(
                        mlx_qmm_t_block_rows(true, Int32(gs), Int32(bits), Int32(m)), 64, what)
                    let x = xAll[0 ..< m]
                    var parts = [MLXArray]()
                    for r in stride(from: 0, to: m, by: chunk) {
                        parts.append(qmm(x[r ..< min(r + chunk, m)]))
                    }
                    let expected = concatenated(parts, axis: 0)
                    let got = qmm(x)
                    // A [1, M, K] prefill batch takes the same tile (M rows, one batch).
                    let got3 = qmm(x.reshaped([1, m, k])).reshaped([m, n])
                    eval(expected, got, got3)
                    XCTAssertTrue(arrayEqual(got, expected).item(Bool.self), what)
                    XCTAssertTrue(arrayEqual(got3, expected).item(Bool.self), what + " [1, M, K]")
                }
            }
        }
    }
}
#endif
