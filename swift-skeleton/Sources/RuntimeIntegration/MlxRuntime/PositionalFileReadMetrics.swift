import Foundation

/**
 * An immutable point-in-time copy of the positional-read counters, safe to
 * hand across concurrency domains while the live metrics keep counting.
 */
public struct PositionalFileReadSnapshot: Equatable, Sendable {

    /// The highest number of reads that overlapped in time during the
    /// snapshot's lifetime.
    public let maximumConcurrentReadCount: Int

    /// How many measured read calls completed (successful or not).
    public let readCallCount: Int

    /// The sum of successfully read bytes.
    public let readByteCount: Int

    /// The summed wall-clock duration of all measured reads.
    public let totalReadElapsedNanoseconds: Int

    /// The single longest measured read.
    public let maximumReadElapsedNanoseconds: Int

    /// How many measured reads returned a failure.
    public let readFailureCount: Int

    public init(
        maximumConcurrentReadCount: Int,
        readCallCount: Int,
        readByteCount: Int,
        totalReadElapsedNanoseconds: Int,
        maximumReadElapsedNanoseconds: Int,
        readFailureCount: Int
    ) {
        self.maximumConcurrentReadCount = maximumConcurrentReadCount
        self.readCallCount = readCallCount
        self.readByteCount = readByteCount
        self.totalReadElapsedNanoseconds = totalReadElapsedNanoseconds
        self.maximumReadElapsedNanoseconds = maximumReadElapsedNanoseconds
        self.readFailureCount = readFailureCount
    }
}

/**
 * Lock-guarded instrumentation for positional file reads. Concurrency
 * journeys measure overlap (how many reads ran simultaneously), volume,
 * and latency so SSD (Solid State Drive) streaming behavior stays
 * attributable instead of guessed at.
 *
 * Marked `@unchecked Sendable`: every field is guarded by `stateLock`, so
 * cross-task sharing is sound by construction.
 */
public final class PositionalFileReadMetrics: @unchecked Sendable {

    private let stateLock: NSLock = NSLock()
    private var activeReadCount: Int = 0
    private var maximumConcurrentReadCount: Int = 0
    private var readCallCount: Int = 0
    private var readByteCount: Int = 0
    private var totalReadElapsedNanoseconds: Int = 0
    private var maximumReadElapsedNanoseconds: Int = 0
    private var readFailureCount: Int = 0

    public init() {
    }

    /**
     * Measures one read operation: overlap at entry, latency around it,
     * and volume or failure at exit.
     *
     * - Parameters:
     *   - byteCount: the number of bytes the operation intends to read.
     *   - operation: the read to run; returning `true` marks success.
     * - Returns: the operation's own success flag, unchanged.
     */
    public func measureRead(byteCount: Int, operation: () -> Bool) -> Bool {
        stateLock.lock()
        activeReadCount += 1
        if (activeReadCount > maximumConcurrentReadCount) {
            maximumConcurrentReadCount = activeReadCount
        }
        stateLock.unlock()

        let startedAtNanoseconds: UInt64 = DispatchTime.now().uptimeNanoseconds
        let didSucceed: Bool = operation()
        let elapsedNanoseconds: Int = Int(DispatchTime.now().uptimeNanoseconds - startedAtNanoseconds)

        stateLock.lock()
        activeReadCount -= 1
        readCallCount += 1
        totalReadElapsedNanoseconds += elapsedNanoseconds
        if (elapsedNanoseconds > maximumReadElapsedNanoseconds) {
            maximumReadElapsedNanoseconds = elapsedNanoseconds
        }
        if (didSucceed) {
            readByteCount += byteCount
        } else {
            readFailureCount += 1
        }
        stateLock.unlock()
        return didSucceed
    }

    /**
     * - Returns: an immutable copy of every counter at this instant.
     */
    public func snapshot() -> PositionalFileReadSnapshot {
        stateLock.lock()
        defer {
            stateLock.unlock()
        }
        return PositionalFileReadSnapshot(
            maximumConcurrentReadCount: maximumConcurrentReadCount,
            readCallCount: readCallCount,
            readByteCount: readByteCount,
            totalReadElapsedNanoseconds: totalReadElapsedNanoseconds,
            maximumReadElapsedNanoseconds: maximumReadElapsedNanoseconds,
            readFailureCount: readFailureCount)
    }
}
