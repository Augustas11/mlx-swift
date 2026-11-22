// Copyright © 2026 Apple Inc.

import Cmlx
import MLX
import XCTest

/// Scope bookkeeping only: these tests build no arrays and touch no device.
class SmallMRouteTests: XCTestCase {

    private func scope() -> Int32 {
        var result: Int32 = 0
        XCTAssertEqual(mlx_small_m_route_scope(&result), 0)
        return result
    }

    func testNoScopeFollowsProcessRoute() {
        XCTAssertEqual(scope(), -1)
        XCTAssertEqual(currentSmallMRoute(), processSmallMRoute())
    }

    func testNestedScopesRestorePrevious() {
        let process = processSmallMRoute()
        withSmallMRoute(.stockOrder) {
            XCTAssertEqual(currentSmallMRoute(), .stockOrder)
            withSmallMRoute(.rowInvariant) {
                XCTAssertEqual(currentSmallMRoute(), .rowInvariant)
                withSmallMRoute(.rowInvariant) {
                    XCTAssertEqual(currentSmallMRoute(), .rowInvariant)
                }
                XCTAssertEqual(currentSmallMRoute(), .rowInvariant)
            }
            XCTAssertEqual(currentSmallMRoute(), .stockOrder)
            XCTAssertEqual(scope(), SmallMRoute.stockOrder.rawValue)
        }
        XCTAssertEqual(scope(), -1)
        XCTAssertEqual(currentSmallMRoute(), process)
        XCTAssertEqual(processSmallMRoute(), process)
    }

    func testScopeRestoredOnThrow() {
        struct Boom: Error {}
        withSmallMRoute(.rowInvariant) {
            do {
                try withSmallMRoute(.stockOrder) {
                    XCTAssertEqual(currentSmallMRoute(), .stockOrder)
                    throw Boom()
                }
                XCTFail("should throw")
            } catch {
                XCTAssertTrue(error is Boom)
            }
            XCTAssertEqual(currentSmallMRoute(), .rowInvariant)
        }
        XCTAssertEqual(scope(), -1)
    }

    func testReturnsBodyValue() {
        let value = withSmallMRoute(.stockOrder) { 42 }
        XCTAssertEqual(value, 42)
    }

    func testScopeIsPerThread() {
        withSmallMRoute(.stockOrder) {
            nonisolated(unsafe) var other: Int32 = 0
            let done = DispatchSemaphore(value: 0)
            Thread {
                mlx_small_m_route_scope(&other)
                done.signal()
            }.start()
            done.wait()
            XCTAssertEqual(other, -1)
            XCTAssertEqual(currentSmallMRoute(), .stockOrder)
        }
    }

    func testInvalidRouteIsAnErrorAndLeavesScope() {
        withSmallMRoute(.stockOrder) {
            for route: Int32 in [0, 3, -1] {
                var rc: Int32 = 0
                do {
                    try withError {
                        rc = mlx_set_small_m_route_scope(route)
                    }
                    XCTFail("route \(route) should be rejected")
                } catch {}
                XCTAssertNotEqual(rc, 0)
                XCTAssertEqual(currentSmallMRoute(), .stockOrder)
            }
        }
        XCTAssertEqual(scope(), -1)
    }

    func testRawValuesMatchCore() {
        XCTAssertEqual(SmallMRoute.rowInvariant.rawValue, 1)
        XCTAssertEqual(SmallMRoute.stockOrder.rawValue, 2)
        XCTAssertNil(SmallMRoute(rawValue: 0))
    }
}
