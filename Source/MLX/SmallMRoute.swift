// Copyright © 2026 Apple Inc.

import Cmlx

/// The route affine quantized matmul takes for small row counts on non-NAX
/// GPUs, chosen per op.
///
/// The process route is read once from `MLX_METAL_SMALL_M_ROW_INVARIANT`
/// (`0` selects the upstream per-row kernels, `2` ``stockOrder``, anything
/// else ``rowInvariant``). ``withSmallMRoute(_:_:)`` overrides it for the ops
/// built on the current thread. Each quantized matmul captures the route when
/// it is added to the graph, and the route joins the compile cache key.
public enum SmallMRoute: Int32, Sendable {
    /// A row's output equals itself in any batch of 1...16 rows.
    case rowInvariant = 1
    /// A 2-row product equals each lone row; 3...16 rows equal each other.
    case stockOrder = 2
}

/// Build the ops in `body` with `route` as this thread's small-row route,
/// then restore the previous scope (or none).
///
/// The route is captured when each op is added to the graph, so `body` must
/// build its graph (or call its compiled function) synchronously; it does
/// not need to evaluate inside the scope. The closure is synchronous, so an
/// `await` inside it is impossible. Scopes nest: the inner route applies
/// inside, the outer one again after it returns or throws.
public func withSmallMRoute<R>(_ route: SmallMRoute, _ body: () throws -> R) rethrows -> R {
    let previous = smallMRouteScope()
    // route is 1 or 2 by construction, the only values the C entry accepts
    mlx_set_small_m_route_scope(route.rawValue)
    defer {
        if let previous {
            mlx_set_small_m_route_scope(previous.rawValue)
        } else {
            mlx_clear_small_m_route_scope()
        }
    }
    return try body()
}

/// The process route, read once from `MLX_METAL_SMALL_M_ROW_INVARIANT`;
/// `nil` when it selects the upstream per-row kernels (`0`), which no scope
/// can select.
public func processSmallMRoute() -> SmallMRoute? {
    var result: Int32 = 0
    mlx_process_small_m_route(&result)
    return SmallMRoute(rawValue: result)
}

/// The route ops built on this thread now capture: the innermost
/// ``withSmallMRoute(_:_:)`` scope, else ``processSmallMRoute()``.
public func currentSmallMRoute() -> SmallMRoute? {
    var result: Int32 = 0
    mlx_current_small_m_route(&result)
    return SmallMRoute(rawValue: result)
}

/// This thread's scope, `nil` when none is set.
func smallMRouteScope() -> SmallMRoute? {
    var result: Int32 = -1
    mlx_small_m_route_scope(&result)
    return SmallMRoute(rawValue: result)
}
