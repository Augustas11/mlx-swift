// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

/// lock to be held while doing any eval or asyncEval.  This is
/// a recursive lock to handle any cases where a closure might
/// call back into eval.
///
/// Acquire it through ``withEvalLock(_:)`` rather than directly: debug builds
/// record ownership there so that the `evalLock` → `CompiledFunction.lock`
/// ordering can be checked.
///
/// `evalLock` is the outermost lock in the library.  Nothing may be acquired
/// before it and then be waited on from inside it; see `CompiledFunction.call`.
let evalLock = NSRecursiveLock()

/// Per-thread `evalLock` ownership.
///
/// Explicit bookkeeping is required: `evalLock` is recursive, so a successful
/// `try()` cannot distinguish "this thread owns it" from "nobody owns it".  The
/// depth is thread-local, so it needs no lock of its own -- which matters,
/// because this runs while holding `evalLock`.
///
/// The depth is kept in every build: ``eval(_:)-(Collection<MLXArray>)`` reads
/// it to decide whether it may release the lock while the device runs.  Debug
/// builds also use it to check the `evalLock` → `CompiledFunction.lock`
/// ordering in `CompiledFunction.call`.
enum EvalLockOwnership {

    private static let depthKey: pthread_key_t = {
        var key = pthread_key_t()
        pthread_key_create(&key, nil)
        return key
    }()

    static var isHeldByCurrentThread: Bool {
        depth > 0
    }

    static var depth: Int {
        Int(bitPattern: pthread_getspecific(depthKey))
    }

    static func entered() {
        pthread_setspecific(depthKey, UnsafeRawPointer(bitPattern: depth + 1))
    }

    static func exited() {
        pthread_setspecific(depthKey, UnsafeRawPointer(bitPattern: depth - 1))
    }
}

#if DEBUG

    extension EvalLockOwnership {

        // Counters, so a test can assert both that the invariant held and that
        // it was actually exercised -- a check count of zero would mean the test
        // proved nothing.
        private static let counterLock = NSLock()
        nonisolated(unsafe) private static var checkCount = 0
        nonisolated(unsafe) private static var violationCount = 0
        nonisolated(unsafe) private static var compileEraseCount = 0

        /// Records, and in debug builds asserts, that `evalLock` is held by the
        /// current thread at a point where an inner lock is about to be taken.
        static func requireHeldByCurrentThread(_ message: @autoclosure () -> String) {
            let held = isHeldByCurrentThread
            counterLock.withLock {
                checkCount += 1
                if !held { violationCount += 1 }
            }
            assert(held, message())
        }

        /// Records a `CompiledFunction.deinit` erase of the compiler cache, so a
        /// test can tell "the erase ran under the lock" from "the deinit never
        /// fired".
        static func recordCompileErase() {
            counterLock.withLock { compileEraseCount += 1 }
        }

        /// (checks, violations) since ``resetCounters()``.
        static var counters: (checks: Int, violations: Int) {
            counterLock.withLock { (checkCount, violationCount) }
        }

        /// Compiler-cache erases since ``resetCounters()``.
        static var compileErases: Int {
            counterLock.withLock { compileEraseCount }
        }

        static func resetCounters() {
            counterLock.withLock {
                checkCount = 0
                violationCount = 0
                compileEraseCount = 0
            }
        }
    }

    /// Which branch of ``evalReleasingLockForWait(_:stock:)`` ran, so a test can
    /// show the path it means to cover was really taken.
    enum EvalWaitPathCounters {
        private static let lock = NSLock()
        nonisolated(unsafe) private static var unlockedWaitCount = 0
        nonisolated(unsafe) private static var recursiveStockCount = 0

        static func recordUnlockedWait() { lock.withLock { unlockedWaitCount += 1 } }
        static func recordRecursiveStock() { lock.withLock { recursiveStockCount += 1 } }

        /// (evals that waited unlocked, recursive evals that took the stock path)
        static var counts: (unlockedWaits: Int, recursiveStock: Int) {
            lock.withLock { (unlockedWaitCount, recursiveStockCount) }
        }

        static func reset() {
            lock.withLock {
                unlockedWaitCount = 0
                recursiveStockCount = 0
            }
        }
    }

#endif

/// Acquire ``evalLock`` for the duration of `body`.
///
/// This is the single choke point for ``evalLock``; it records per-thread
/// ownership so that a synchronous eval can tell a recursive entry, and debug
/// builds can check that `CompiledFunction`'s per-instance lock is never taken
/// outside it.
@inline(__always)
func withEvalLock<R>(_ body: () throws -> R) rethrows -> R {
    evalLock.lock()
    EvalLockOwnership.entered()
    defer {
        EvalLockOwnership.exited()
        evalLock.unlock()
    }
    return try body()
}

/// Synchronous eval of `arrays` that holds ``evalLock`` while the graph is
/// scheduled and releases it while the device runs (macprovider #2338).
///
/// Encoding and command submission stay under the lock. Only the wait for
/// completion runs unlocked, and it waits on copies of the completion events,
/// so it reads and writes no array: another thread may use the same arrays as
/// eval inputs meanwhile. The lock is taken again to mark the outputs
/// available, so every array status and event write stays under the lock.
///
/// `stock` runs instead, under the lock, when this thread already holds
/// ``evalLock`` (a recursive entry: the lock would not really be released, and
/// the caller may rely on the eval completing inside its locked region). The
/// shim itself falls back to the stock synchronous eval for CPU streams and
/// when the asynchronous schedule throws.
@inline(__always)
private func evalReleasingLockForWait(
    _ arrays: UnsafeBufferPointer<mlx_array>, stock: () -> Void
) {
    if EvalLockOwnership.isHeldByCurrentThread {
        #if DEBUG
            EvalWaitPathCounters.recordRecursiveStock()
        #endif
        withEvalLock { stock() }
        return
    }
    var pending = mlx_eval_wait(ctx: nil)
    let status = withEvalLock {
        mlx_eval_wait_begin(&pending, arrays.baseAddress, arrays.count)
    }
    guard status == 0, pending.ctx != nil else { return }
    #if DEBUG
        EvalWaitPathCounters.recordUnlockedWait()
    #endif
    _ = mlx_eval_wait_wait(pending)
    withEvalLock {
        mlx_eval_wait_finish(pending)
    }
    mlx_eval_wait_free(pending)
}

/// Synchronous eval of a single array context; see ``evalReleasingLockForWait(_:stock:)``.
func evalReleasingLockForWait(_ ctx: mlx_array) {
    withUnsafePointer(to: ctx) { p in
        evalReleasingLockForWait(UnsafeBufferPointer(start: p, count: 1)) {
            _ = mlx_array_eval(ctx)
        }
    }
}

/// Evaluate one or more `MLXArray`
///
/// ### See Also
/// - <doc:lazy-evaluation>
public func eval(_ arrays: MLXArray...) {
    eval(arrays)
}

/// Evaluate one or more `MLXArray`
///
/// ### See Also
/// - <doc:lazy-evaluation>
public func eval(_ arrays: some Collection<MLXArray>) {
    let ctxs = arrays.map { $0.ctx }
    withExtendedLifetime(arrays) {
        ctxs.withUnsafeBufferPointer { buffer in
            evalReleasingLockForWait(buffer) {
                let vector_array = mlx_vector_array_new_data(buffer.baseAddress, buffer.count)
                mlx_eval(vector_array)
                mlx_vector_array_free(vector_array)
            }
        }
    }
}

/// Evaluate one or more `MLXArray` asynchronously.
///
/// ### See Also
/// - <doc:lazy-evaluation>
/// - ``asyncEval(_:)-(Collection<MLXArray>)``
public func asyncEval(_ arrays: some Collection<MLXArray>) {
    let vector_array = new_mlx_vector_array(arrays)
    _ = withEvalLock {
        mlx_async_eval(vector_array)
    }
    mlx_vector_array_free(vector_array)
}

/// Evaluate one or more `MLXArray`.
///
/// This variant allows several structured types:
///
/// ```swift
/// let a: MLXArray
/// let b: [MLXArray]
/// let c: [String:MLXArray]
/// let d: [String:[MLXArray]]
/// let e: (MLXArray, MLXArray)
/// let f: [(String, MLXArray)]
/// let nested: [(MLXArray, [MLXArray])]
///
/// eval(a, b, c, d, e, f)
/// ```
///
/// Other structured types may be supported -- check the implementation.
///
/// ### See Also
/// - <doc:lazy-evaluation>
/// - ``asyncEval(_:)-(Collection<MLXArray>)``
public func eval(_ values: Any...) {
    var arrays = [MLXArray]()

    for item in values {
        collect(item, into: &arrays)
    }

    eval(arrays)
}

/// Evaluate one or more `MLXArray`.
///
/// See ``eval(_:)``
public func eval(_ values: some Sequence<Any>) {
    var arrays = [MLXArray]()

    for item in values {
        collect(item, into: &arrays)
    }

    eval(arrays)
}

/// Variant of ``eval(_:)-(Collection<MLXArray>)`` that checks for errors in MLX and throws.
///
/// ### See Also
/// - <doc:lazy-evaluation>
public func checkedEval(_ values: Any...) throws {
    var arrays = [MLXArray]()

    for item in values {
        collect(item, into: &arrays)
    }

    try withError {
        eval(arrays)
    }
}

/// Variant of ``eval(_:)-(MLXArray...)`` that checks for errors in MLX and throws.
///
/// ### See Also
/// - <doc:lazy-evaluation>
public func checkedEval(_ values: some Sequence<Any>) throws {
    var arrays = [MLXArray]()

    for item in values {
        collect(item, into: &arrays)
    }

    try withError {
        eval(arrays)
    }
}

/// Evaluate one or more `MLXArray` asynchronously.
///
/// This variant allows several structured types:
///
/// ```swift
/// let a: MLXArray
/// let b: [MLXArray]
/// let c: [String:MLXArray]
/// let d: [String:[MLXArray]]
/// let e: (MLXArray, MLXArray)
/// let f: [(String, MLXArray)]
/// let nested: [(MLXArray, [MLXArray])]
///
/// asyncEval(a, b, c, d, e, f)
/// ```
///
/// Other structured types may be supported -- check the implementation.
///
/// ### See Also
/// - <doc:lazy-evaluation>
public func asyncEval(_ values: Any...) {
    var arrays = [MLXArray]()

    for item in values {
        collect(item, into: &arrays)
    }

    asyncEval(arrays)
}

/// Evaluate one or more `MLXArray` asynchronously.
///
/// See ``asyncEval(_:)-(Collection<MLXArray>)``
public func asyncEval(_ values: some Sequence<Any>) {
    var arrays = [MLXArray]()

    for item in values {
        collect(item, into: &arrays)
    }

    asyncEval(arrays)
}

private func collect(_ item: Any, into arrays: inout [MLXArray]) {
    switch item {
    case let v as Evaluatable:
        arrays.append(contentsOf: v.innerState())

    case let v as NestedDictionary<String, MLXArray>:
        arrays.append(contentsOf: v.flattened().map { $0.1 })

    case let v as MLXArray:
        arrays.append(v)
    case let v as [MLXArray]:
        arrays.append(contentsOf: v)
    case let v as [Any]:
        for item in v {
            collect(item, into: &arrays)
        }
    case let v as [AnyHashable: Any]:
        for item in v.values {
            collect(item, into: &arrays)
        }
    case let v as (Any, Any):
        collect(v.0, into: &arrays)
        collect(v.1, into: &arrays)
    case let v as (Any, Any, Any):
        collect(v.0, into: &arrays)
        collect(v.1, into: &arrays)
        collect(v.2, into: &arrays)
    case let v as (Any, Any, Any, Any):
        collect(v.0, into: &arrays)
        collect(v.1, into: &arrays)
        collect(v.2, into: &arrays)
        collect(v.3, into: &arrays)
    case let v as (Any, Any, Any, Any, Any):
        collect(v.0, into: &arrays)
        collect(v.1, into: &arrays)
        collect(v.2, into: &arrays)
        collect(v.3, into: &arrays)
        collect(v.4, into: &arrays)
    case is String, is any BinaryInteger, is any BinaryFloatingPoint:
        // ignore, e.g. (String, MLXArray)
        break
    default:
        fatalError("Unable to extract MLXArray from \(item)")
    }
}
