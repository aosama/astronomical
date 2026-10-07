import Foundation

import AstronomicalConfig
import Testing

@testable import Supervisor

/// Shared fixtures for the attribution journey suites: bounded async waits,
/// injectable sinks and clocks, and the literary Romeo and Juliet payload the
/// Rust journeys use so oversized tool-call arguments exercise the bounding
/// path with source text rather than random tokens.
enum AttributionJourneySupport {

    static let TEST_REVISION: String = "0123456789abcdef0123456789abcdef01234567"

    static let JOURNEY_DEADLINE_SECONDS: Double = 5

    /// A literary payload used as oversized tool-call arguments.
    static func romeoArgumentsPayload(repeats: Int) -> String {
        let passage: String =
            "Juliet: O Romeo, Romeo! wherefore art thou Romeo? Deny thy father and refuse thy name."
        return String(repeating: passage, count: repeats)
    }

    /// One fresh temporary directory per journey; the caller owns cleanup.
    static func freshTemporaryDirectory(named directoryLabel: String) throws -> FilePath {
        let temporaryRoot: String = NSTemporaryDirectory()
            + "attribution-\(directoryLabel)-\(UUID().uuidString.prefix(8))"
        try FileManager.default.createDirectory(
            atPath: temporaryRoot,
            withIntermediateDirectories: true)
        return FilePath(string: temporaryRoot)
    }

    /// Awaits one operation under a hard deadline so a wedged collaborator
    /// fails the journey bounded instead of hanging the suite.
    static func awaitBounded<T: Sendable>(
        _ boundedOperation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        let boundedResult: T = try await withThrowingTaskGroup(
            of: T.self,
            returning: T.self,
            body: { (taskGroup: inout ThrowingTaskGroup<T, any Error>) -> T in
            taskGroup.addTask(operation: boundedOperation)
            taskGroup.addTask(operation: { () throws -> T in
                try await Task.sleep(
                    nanoseconds: UInt64(AttributionJourneySupport.JOURNEY_DEADLINE_SECONDS * 1_000_000_000))
                throw AttributionJourneySupport.JourneyDeadlineExceeded()
            })
            guard let firstFinished: T = try await taskGroup.next() else {
                throw AttributionJourneySupport.JourneyDeadlineExceeded()
            }
            taskGroup.cancelAll()
            return firstFinished
        })
        return boundedResult
    }

    struct JourneyDeadlineExceeded: Error, CustomStringConvertible {
        var description: String {
            return "attribution journey coverage must remain bounded"
        }
    }
}

/// Thread-safe in-memory attribution sink shared across concurrent measured
/// operations; the Rust journeys capture writes through an Arc<Mutex<Vec<u8>>>.
final class SharedMemoryAttributionSink: SupervisorAttributionWriterSink, @unchecked Sendable {

    private let stateLock: NSLock = NSLock()
    private var accumulatedBytes: Data = Data()

    func write(_ bytes: Data) throws {
        self.stateLock.lock()
        self.accumulatedBytes.append(bytes)
        self.stateLock.unlock()
    }

    func flush() throws {
    }

    func writtenBytes() -> Data {
        self.stateLock.lock()
        defer { self.stateLock.unlock(); }
        return self.accumulatedBytes
    }

    func recordCount() -> Int {
        let writtenBytes: Data = self.writtenBytes()
        return writtenBytes.split(separator: UInt8(ascii: "\n"))
            .filter({ (recordBytes: Data.SubSequence) -> Bool in
                return !recordBytes.isEmpty
            }).count
    }
}

/// Discards every write; the Rust journeys pass std::io::sink().
final class DiscardAttributionSink: SupervisorAttributionWriterSink {

    func write(_ bytes: Data) throws {
    }

    func flush() throws {
    }
}

/// Fails every write, but only after the measured operation completed; the
/// proof that attribution failures never preempt the operation.
final class CompletionCheckingAttributionSink: SupervisorAttributionWriterSink, @unchecked Sendable {

    private let operationCompleted: () -> Bool

    init(operationCompleted: @escaping () -> Bool) {
        self.operationCompleted = operationCompleted
    }

    func write(_ bytes: Data) throws {
        guard self.operationCompleted() else {
            throw JourneyAssertionFailure(problem: "the measured operation must complete before attribution writes")
        }
        throw IntentionalWriteFailure()
    }

    func flush() throws {
        throw IntentionalWriteFailure()
    }
}

struct IntentionalWriteFailure: Error, CustomStringConvertible {
    var description: String {
        return "intentional write failure"
    }
}

struct JourneyAssertionFailure: Error, CustomStringConvertible {
    let problem: String

    var description: String {
        return self.problem
    }
}

/// Thread-safe monotonic clock counter for deterministic attribution rows.
final class JourneyClockCounter: @unchecked Sendable {

    private let stateLock: NSLock = NSLock()
    private var callCount: Int = 0

    var totalCalls: Int {
        self.stateLock.lock()
        defer { self.stateLock.unlock(); }
        return self.callCount
    }

    func nextValue() -> UInt64 {
        self.stateLock.lock()
        let clockIndex: Int = self.callCount
        self.callCount = self.callCount + 1
        self.stateLock.unlock()
        return UInt64(1_000 + clockIndex)
    }
}

/// Thread-safe scripted clock returning one fixed value per call, in order;
/// exhausted values fail the journey instead of unrolling time.
final class JourneySequencedClock: @unchecked Sendable {

    private let stateLock: NSLock = NSLock()
    private var remainingValues: Array<UInt64>

    init(values: Array<UInt64>) {
        self.remainingValues = values
    }

    var remainingValueCount: Int {
        self.stateLock.lock()
        defer { self.stateLock.unlock(); }
        return self.remainingValues.count
    }

    func nextValue() throws -> UInt64 {
        self.stateLock.lock()
        defer { self.stateLock.unlock(); }
        guard let nextClockValue: UInt64 = self.remainingValues.first else {
            throw JourneyAssertionFailure(problem: "the journey clock ran out of scripted values")
        }
        self.remainingValues.removeFirst()
        return nextClockValue
    }
}

/// Lock-guarded boolean shared between a measured operation and its sink.
final class JourneyFlag: @unchecked Sendable {

    private let stateLock: NSLock = NSLock()
    private var flagValue: Bool = false

    var isSet: Bool {
        self.stateLock.lock()
        defer { self.stateLock.unlock(); }
        return self.flagValue
    }

    func set() -> Void {
        self.stateLock.lock()
        self.flagValue = true
        self.stateLock.unlock()
    }
}

/// An async one-shot gate coordinating concurrent measured operations in the
/// lock-freedom journey without blocking cooperative threads.
actor JourneyGate {

    private var isSignaled: Bool = false
    private var waitingContinuations: Array<CheckedContinuation<Void, Never>> = []

    func waitUntilSignaled() async -> Void {
        if self.isSignaled {
            return
        }
        await withCheckedContinuation({ (continuation: CheckedContinuation<Void, Never>) -> Void in
            self.waitingContinuations.append(continuation)
        })
    }

    func signal() -> Void {
        self.isSignaled = true
        for waitingContinuation: CheckedContinuation<Void, Never> in self.waitingContinuations {
            waitingContinuation.resume()
        }
        self.waitingContinuations = []
    }
}
