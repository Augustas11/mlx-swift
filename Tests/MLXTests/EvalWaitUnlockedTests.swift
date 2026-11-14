// Copyright © 2026 Malibu. macprovider fork addition (#2338).

import Foundation
import XCTest

@testable import MLX

// The path counters these tests read are debug-only.
#if DEBUG

    /// A synchronous eval holds `evalLock` while it schedules and releases it while
    /// the device runs (`evalReleasingLockForWait`).  These tests cover the
    /// concurrency contract of that split: independent graphs on two threads, one
    /// array shared by two threads while it is in flight, recursive entry under the
    /// lock, CPU streams, and the error path.  All arrays are small and run on the
    /// default (GPU) device and its one shared stream.
    class EvalWaitUnlockedTests: XCTestCase {

        final class Box<T>: @unchecked Sendable {
            private let lock = NSLock()
            private var _value: T
            init(_ value: T) { _value = value }
            var value: T {
                get { lock.withLock { _value } }
                set { lock.withLock { _value = newValue } }
            }
            func update(_ body: (inout T) -> Void) { lock.withLock { body(&_value) } }
        }

        /// Cyclic barrier for a fixed number of threads.
        final class Barrier: @unchecked Sendable {
            private let condition = NSCondition()
            private let parties: Int
            private var waiting = 0
            private var generation = 0

            init(_ parties: Int) { self.parties = parties }

            func arriveAndWait() {
                condition.lock()
                defer { condition.unlock() }
                let gen = generation
                waiting += 1
                if waiting == parties {
                    waiting = 0
                    generation += 1
                    condition.broadcast()
                    return
                }
                while gen == generation {
                    condition.wait()
                }
            }
        }

        /// Runs `body(0)` and `body(1)` on two dedicated threads and joins them.
        /// Fails (rather than hangs) if they do not finish within `timeout`.
        private func onTwoThreads(
            timeout: TimeInterval = 60, _ body: @escaping @Sendable (Int) -> Void
        ) {
            let done = DispatchSemaphore(value: 0)
            for index in 0 ..< 2 {
                let thread = Thread {
                    body(index)
                    done.signal()
                }
                thread.stackSize = 8 << 20
                thread.start()
            }
            for _ in 0 ..< 2 {
                XCTAssertEqual(
                    done.wait(timeout: .now() + timeout), .success,
                    "a thread did not finish: deadlock or starvation")
            }
        }

        /// A small deterministic GPU graph of a dozen ops.
        private static func graph(_ seed: Int) -> MLXArray {
            var x = MLXArray(0 ..< 256, [16, 16]).asType(.float32) / 256 + Float(seed % 97)
            for _ in 0 ..< 6 {
                x = tanh(matmul(x, x.T) / 16) + Float(seed % 13) * 0.01
            }
            return x.sum(axis: 1)
        }

        override func setUp() {
            super.setUp()
            EvalWaitPathCounters.reset()
        }

        // MARK: - (a) independent graphs on two threads

        func testTwoThreadsIndependentGraphsMatchSingleThreaded() {
            let iterations = 100
            var expected: [[[Float]]] = [[], []]
            for t in 0 ..< 2 {
                for i in 0 ..< iterations {
                    expected[t].append(Self.graph(t * 1000 + i).asArray(Float.self))
                }
            }
            EvalWaitPathCounters.reset()

            let results = Box([[[Float]]](repeating: [], count: 2))
            let firstDone = Box([UInt64](repeating: 0, count: 2))
            let lastDone = Box([UInt64](repeating: 0, count: 2))
            let start = Barrier(2)

            onTwoThreads { t in
                start.arriveAndWait()
                var mine: [[Float]] = []
                for i in 0 ..< iterations {
                    let y = Self.graph(t * 1000 + i)
                    eval(y)
                    mine.append(y.asArray(Float.self))
                    let now = DispatchTime.now().uptimeNanoseconds
                    if i == 0 { firstDone.update { $0[t] = now } }
                    lastDone.update { $0[t] = now }
                }
                results.update { $0[t] = mine }
            }

            XCTAssertEqual(results.value[0], expected[0])
            XCTAssertEqual(results.value[1], expected[1])
            // Both made progress while the other was still running.
            XCTAssertLessThan(firstDone.value[0], lastDone.value[1])
            XCTAssertLessThan(firstDone.value[1], lastDone.value[0])
            // The unlocked-wait path really ran (eval(y) on a fresh graph).
            XCTAssertGreaterThanOrEqual(
                EvalWaitPathCounters.counts.unlockedWaits, 2 * iterations)
            XCTAssertEqual(EvalWaitPathCounters.counts.recursiveStock, 0)
        }

        /// The lock is free while the device runs: a second thread takes
        /// `evalLock` and finishes a CPU eval (stock locked path) before a long
        /// GPU eval on the first thread returns.  With the lock held across the
        /// wait, as stock mlx-swift does, the second thread cannot get the lock
        /// until the first eval returns and this test fails.
        func testLockIsFreeWhileTheDeviceRuns() {
            let n = 2048
            let scheduled = DispatchSemaphore(value: 0)
            let gpuReturned = Box<UInt64>(0)
            let probeLocked = Box<UInt64>(0)
            let gpuOK = Box(false)
            let cpuOK = Box(false)

            onTwoThreads { t in
                if t == 0 {
                    let w = MLXArray.ones([n, n])
                    var x = MLXArray.ones([n, n])
                    for _ in 0 ..< 24 {
                        x = matmul(x, w) / Float(n)
                    }
                    scheduled.signal()
                    eval(x)
                    gpuReturned.value = DispatchTime.now().uptimeNanoseconds
                    gpuOK.value = (x[0, 0].item(Float.self) == 1) && (x.sum().item(Float.self) == Float(n * n))
                } else {
                    scheduled.wait()
                    // Thread 0 schedules a 48-op graph in well under this.
                    Thread.sleep(forTimeInterval: 0.003)
                    let a = MLXArray([Float(0), 1, 2, 3], [2, 2])
                    let y = matmul(a, a, stream: .cpu)
                    eval(y)
                    probeLocked.value = DispatchTime.now().uptimeNanoseconds
                    cpuOK.value = y.asArray(Float.self) == [2, 3, 6, 11]
                }
            }

            XCTAssertTrue(gpuOK.value)
            XCTAssertTrue(cpuOK.value)
            XCTAssertLessThan(
                probeLocked.value, gpuReturned.value,
                "the CPU eval waited for the GPU eval: evalLock was held across the wait")
        }

        // MARK: - (b) one array shared by two threads while in flight

        /// One lazy array `x` per iteration; both threads start together and
        /// either both eval `x`, or one evals `x` while the other evals a graph
        /// that takes `x` as an input.  This is the race of an unlocked
        /// `array::wait()` (detach_event / set_status) against `eval_impl`
        /// reading or detaching the same array's event under the lock.
        func testSharedArrayEvaluatedAndUsedAsInputConcurrently() {
            let iterations = 300
            var expectedX: [[Float]] = []
            var expectedY: [[Float]] = []
            for i in 0 ..< iterations {
                let x = Self.graph(i)
                let y = x * 2 + 1
                eval(x, y)
                expectedX.append(x.asArray(Float.self))
                expectedY.append(y.asArray(Float.self))
            }
            let ex = expectedX
            let ey = expectedY
            EvalWaitPathCounters.reset()

            let shared = Box<MLXArray?>(nil)
            let barrier = Barrier(2)
            let mismatches = Box(0)

            onTwoThreads { t in
                for i in 0 ..< iterations {
                    if t == 0 { shared.value = Self.graph(i) }
                    barrier.arriveAndWait()
                    let x = shared.value!
                    switch (i % 3, t) {
                    case (0, _), (1, 0), (2, 1):
                        eval(x)
                        if x.asArray(Float.self) != ex[i] { mismatches.update { $0 += 1 } }
                    default:
                        let y = x * 2 + 1
                        eval(y)
                        if y.asArray(Float.self) != ey[i] { mismatches.update { $0 += 1 } }
                        // x is now an evaluated input; reading it must be correct too
                        if x.asArray(Float.self) != ex[i] { mismatches.update { $0 += 1 } }
                    }
                    barrier.arriveAndWait()
                }
            }

            XCTAssertEqual(mismatches.value, 0)
            XCTAssertGreaterThan(EvalWaitPathCounters.counts.unlockedWaits, iterations)
        }

        // MARK: - (c) recursive entry

        func testRecursiveEvalUnderTheLockTakesTheStockPath() {
            let expected = Self.graph(7).asArray(Float.self)
            EvalWaitPathCounters.reset()

            let x = Self.graph(7)
            let got: [Float] = withEvalLock {
                XCTAssertEqual(EvalLockOwnership.depth, 1)
                eval(x)
                // asArray evals again (already evaluated): also recursive
                let v = x.asArray(Float.self)
                XCTAssertEqual(EvalLockOwnership.depth, 1)
                return v
            }
            XCTAssertEqual(got, expected)
            XCTAssertEqual(EvalLockOwnership.depth, 0)

            let counts = EvalWaitPathCounters.counts
            XCTAssertGreaterThanOrEqual(counts.recursiveStock, 2)
            XCTAssertEqual(counts.unlockedWaits, 0)

            // The lock is not leaked: another thread can eval on the unlocked path.
            let other = Box<[Float]>([])
            onTwoThreads(timeout: 30) { t in
                if t == 0 {
                    let y = Self.graph(7)
                    eval(y)
                    other.value = y.asArray(Float.self)
                }
            }
            XCTAssertEqual(other.value, expected)
            XCTAssertEqual(EvalWaitPathCounters.counts.unlockedWaits, 1)
        }

        /// One thread evaluates recursively under the lock while another runs
        /// unlocked-wait evals; neither deadlocks and both are correct.
        func testRecursiveAndUnlockedEvalsOnTwoThreads() {
            let iterations = 50
            var expected: [[Float]] = []
            for i in 0 ..< iterations { expected.append(Self.graph(i).asArray(Float.self)) }
            let ex = expected
            let mismatches = Box(0)

            onTwoThreads(timeout: 60) { t in
                for i in 0 ..< iterations {
                    let y = Self.graph(i)
                    let v: [Float]
                    if t == 0 {
                        v = withEvalLock {
                            eval(y)
                            return y.asArray(Float.self)
                        }
                    } else {
                        eval(y)
                        v = y.asArray(Float.self)
                    }
                    if v != ex[i] { mismatches.update { $0 += 1 } }
                }
            }

            XCTAssertEqual(mismatches.value, 0)
            let counts = EvalWaitPathCounters.counts
            XCTAssertGreaterThanOrEqual(counts.recursiveStock, iterations)
            XCTAssertGreaterThanOrEqual(counts.unlockedWaits, iterations)
        }

        // MARK: - CPU and already-evaluated fast paths

        func testCPUStreamAndAvailableArraysDoNotWaitUnlocked() {
            let a = MLXArray([Float(0), 1, 2, 3], [2, 2])
            let y = matmul(a, a, stream: .cpu)
            eval(y)
            XCTAssertEqual(y.asArray(Float.self), [2, 3, 6, 11])
            XCTAssertEqual(EvalWaitPathCounters.counts.unlockedWaits, 0)

            let g = Self.graph(3)
            eval(g)
            XCTAssertEqual(EvalWaitPathCounters.counts.unlockedWaits, 1)
            // already available: no second wait
            eval(g)
            _ = g.asArray(Float.self)
            XCTAssertEqual(EvalWaitPathCounters.counts.unlockedWaits, 1)
        }

        // MARK: - (d) error path

        /// A primitive that throws inside eval (a Metal kernel that does not
        /// compile).  The asynchronous schedule fails, the stock synchronous
        /// eval reruns and reports the error exactly once, the lock is released,
        /// and the next eval works.
        func testEvalErrorIsReportedOnceAndReleasesTheLock() {
            let expected = Self.graph(5).asArray(Float.self)
            EvalWaitPathCounters.reset()

            let messages = Box<[String]>([])
            withErrorHandler({ message in messages.update { $0.append(message) } }) {
                let kernel = MLXFast.metalKernel(
                    name: "eval_wait_broken",
                    inputNames: ["a"],
                    outputNames: ["out1"],
                    source: """
                            this is not metal;
                        """)
                let a = MLXArray.ones([4])
                let out = kernel(
                    [a],
                    grid: (4, 1, 1),
                    threadGroup: (4, 1, 1),
                    outputShapes: [[4]],
                    outputDTypes: [.float32])
                eval(out[0])
            }

            XCTAssertEqual(messages.value.count, 1, "\(messages.value)")
            XCTAssertEqual(EvalLockOwnership.depth, 0)
            XCTAssertEqual(EvalWaitPathCounters.counts.unlockedWaits, 0)

            // The lock is free for another thread, and the next eval is correct.
            let other = Box<[Float]>([])
            onTwoThreads(timeout: 30) { t in
                if t == 0 {
                    let y = Self.graph(5)
                    eval(y)
                    other.value = y.asArray(Float.self)
                }
            }
            XCTAssertEqual(other.value, expected)
        }
    }

#endif
