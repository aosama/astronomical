import Foundation

import AstronomicalConfig
import IpcProtocol
import os

/// Injectable append target for supervisor attribution rows; production uses
/// a file-backed sink, journeys use in-memory or failing sinks.
public protocol SupervisorAttributionWriterSink: AnyObject, Sendable {

    func write(_ bytes: Data) throws

    func flush() throws
}

/// File-backed append sink for the attribution JSONL file.
public final class FileAttributionWriterSink: SupervisorAttributionWriterSink {

    private let appendableFile: FileHandle

    public init(appendableFile: FileHandle) {
        self.appendableFile = appendableFile
    }

    public func write(_ bytes: Data) throws {
        try self.appendableFile.write(contentsOf: bytes)
    }

    public func flush() throws {
        // Rust's BufWriter flush only drains the userspace buffer into the
        // kernel; FileHandle.write already reached the kernel, so there is
        // nothing left to do and no per-record fsync is imposed.
    }
}

/**
 * Switchable start/end attribution for supervisor-owned operations,
 * mirroring the Rust log from
 * apps/supervisor/src/supervisor_performance_attribution.rs.
 *
 * Every supervisor-owned operation (library catalog load, download stages,
 * Qwen seed loading) is measured with explicit start/end wall-clock and
 * monotonic elapsed attribution and appended as one flushed JSON line to
 * `supervisor-performance-attribution.jsonl` when the diagnostics flag is
 * enabled. The disabled state owns neither a writer nor a clock, so a
 * disabled log never reads the clock.
 */
public final class SupervisorPerformanceAttributionLog: @unchecked Sendable {

    private static let SUPERVISOR_PERFORMANCE_ATTRIBUTION_FILE_NAME: String =
        "supervisor-performance-attribution.jsonl"

    private static let attributionLogger: Logger = Logger(
        subsystem: "dev.astronomical.supervisor",
        category: "performance-attribution")

    private let enabledSink: EnabledAttributionSink?

    private init(enabledSink: EnabledAttributionSink?) {
        self.enabledSink = enabledSink
    }

    /// Creates a no-overhead log for tests and application variants without diagnostics.
    public static func disabled() -> SupervisorPerformanceAttributionLog {
        return SupervisorPerformanceAttributionLog(enabledSink: nil)
    }

    /**
     * Opens the attribution log when the diagnostics flag is enabled.
     *
     * - Parameters:
     *   - logDirectory: the instance's logs directory; must already exist.
     *   - performanceAttributionEnabled: the resolved diagnostics flag.
     * - Throws: a Cocoa error when the file cannot be created or opened while enabled.
     */
    public static func open(
        logDirectory: FilePath,
        performanceAttributionEnabled: Bool
    ) throws -> SupervisorPerformanceAttributionLog {
        guard performanceAttributionEnabled else {
            return SupervisorPerformanceAttributionLog.disabled()
        }
        let attributionLogPath: FilePath = logDirectory.appending(
            component: SupervisorPerformanceAttributionLog.SUPERVISOR_PERFORMANCE_ATTRIBUTION_FILE_NAME)
        let attributionLogUrl: URL = URL(fileURLWithPath: attributionLogPath.string)
        if FileManager.default.fileExists(atPath: attributionLogUrl.path) == false {
            let isCreated: Bool = FileManager.default.createFile(
                atPath: attributionLogUrl.path,
                contents: nil)
            guard isCreated else {
                throw CocoaError(.fileNoSuchFile)
            }
        }
        guard let appendableFile: FileHandle = FileHandle(forWritingAtPath: attributionLogUrl.path) else {
            throw CocoaError(.fileNoSuchFile)
        }
        do {
            try appendableFile.seekToEnd()
        } catch {
            do {
                try appendableFile.close()
            } catch let closeError {
                SupervisorPerformanceAttributionLog.attributionLogger.warning(
                    "failed to close the supervisor attribution log after a rejected open: \(String(describing: closeError), privacy: .public)")
            }
            throw error
        }
        return SupervisorPerformanceAttributionLog(enabledSink: EnabledAttributionSink(
            writer: FileAttributionWriterSink(appendableFile: appendableFile),
            unixEpochMillis: { () throws -> UInt64 in
                return PerformanceLogClock.unixEpochMillis()
            }))
    }

    /// Builds an enabled log over an injectable writer, keeping writer
    /// failures deterministic without depending on filesystem behavior.
    public static func fromWriterAndClock(
        writer: SupervisorAttributionWriterSink,
        unixEpochMillis: @escaping () throws -> UInt64
    ) -> SupervisorPerformanceAttributionLog {
        return SupervisorPerformanceAttributionLog(enabledSink: EnabledAttributionSink(
            writer: writer,
            unixEpochMillis: unixEpochMillis))
    }

    /// Builds a log that discards the writer and clock unless enabled,
    /// proving the disabled branch drops test collaborators entirely.
    public static func fromWriterAndClockWhenEnabled(
        writer: SupervisorAttributionWriterSink,
        unixEpochMillis: @escaping () throws -> UInt64,
        performanceAttributionEnabled: Bool
    ) -> SupervisorPerformanceAttributionLog {
        guard performanceAttributionEnabled else {
            return SupervisorPerformanceAttributionLog.disabled()
        }
        return SupervisorPerformanceAttributionLog.fromWriterAndClock(
            writer: writer,
            unixEpochMillis: unixEpochMillis)
    }

    /// Whether the log records rows.
    public var isEnabled: Bool {
        return self.enabledSink != nil
    }

    /**
     * Measures a synchronous operation and appends one flushed JSON record
     * when enabled.
     *
     * - Throws: `SupervisorPerformanceAttributionError` when the clock, the
     *   operation/detail pairing, or the writer fails while enabled.
     */
    public func measureOperation<OperationOutput>(
        operation: SupervisorPerformanceOperation,
        measuredOperation: () -> OperationOutput,
        describeMeasurement: (OperationOutput) -> SupervisorPerformanceMeasurement
    ) throws -> OperationOutput {
        guard let enabledSink: EnabledAttributionSink = self.enabledSink else {
            return measuredOperation()
        }
        let startedAtUnixMillis: UInt64 = try SupervisorPerformanceAttributionLog.readClock(
            enabledSink: enabledSink)
        let startedAtMonotonic: DispatchTime = DispatchTime.now()
        let operationOutput: OperationOutput = measuredOperation()
        try self.finishMeasurement(
            enabledSink: enabledSink,
            operation: operation,
            startedAtUnixMillis: startedAtUnixMillis,
            startedAtMonotonic: startedAtMonotonic,
            measurement: describeMeasurement(operationOutput))
        return operationOutput
    }

    /**
     * Measures an async operation. The operation body runs without holding
     * the writer lock, so concurrent operations can complete and record
     * while another measured operation is still awaiting.
     *
     * - Throws: `SupervisorPerformanceAttributionError` when the clock, the
     *   operation/detail pairing, or the writer fails while enabled.
     */
    public func measureAsyncOperation<OperationOutput>(
        operation: SupervisorPerformanceOperation,
        measuredOperation: () async -> OperationOutput,
        describeMeasurement: (OperationOutput) -> SupervisorPerformanceMeasurement
    ) async throws -> OperationOutput {
        guard let enabledSink: EnabledAttributionSink = self.enabledSink else {
            return await measuredOperation()
        }
        let startedAtUnixMillis: UInt64 = try SupervisorPerformanceAttributionLog.readClock(
            enabledSink: enabledSink)
        let startedAtMonotonic: DispatchTime = DispatchTime.now()
        let operationOutput: OperationOutput = await measuredOperation()
        try self.recordCompletedMeasurement(
            enabledSink: enabledSink,
            operation: operation,
            startedAtUnixMillis: startedAtUnixMillis,
            startedAtMonotonic: startedAtMonotonic,
            measurement: describeMeasurement(operationOutput))
        return operationOutput
    }

    /**
     * Measures request-path work without allowing diagnostic failures to
     * alter its output: clock and writer failures warn and the operation
     * output is returned unchanged.
     */
    public func measureAsyncOperationBestEffort<OperationOutput>(
        operation: SupervisorPerformanceOperation,
        measuredOperation: () async -> OperationOutput,
        describeMeasurement: (OperationOutput) -> SupervisorPerformanceMeasurement
    ) async -> OperationOutput {
        guard let enabledSink: EnabledAttributionSink = self.enabledSink else {
            return await measuredOperation()
        }
        let startedAtUnixMillis: UInt64
        do {
            startedAtUnixMillis = try SupervisorPerformanceAttributionLog.readClock(enabledSink: enabledSink)
        } catch let attributionError {
            SupervisorPerformanceAttributionLog.attributionLogger.warning(
                "operation=\(operation.wireName, privacy: .public) supervisor performance attribution could not start: \(attributionError.description, privacy: .public)")
            return await measuredOperation()
        }
        let startedAtMonotonic: DispatchTime = DispatchTime.now()
        let operationOutput: OperationOutput = await measuredOperation()
        do {
            try self.recordCompletedMeasurement(
                enabledSink: enabledSink,
                operation: operation,
                startedAtUnixMillis: startedAtUnixMillis,
                startedAtMonotonic: startedAtMonotonic,
                measurement: describeMeasurement(operationOutput))
        } catch let attributionError {
            // Optional diagnostics cannot become a hidden model-steering dependency.
            SupervisorPerformanceAttributionLog.attributionLogger.warning(
                "operation=\(operation.wireName, privacy: .public) supervisor performance attribution could not finish: \(attributionError.description, privacy: .public)")
        }
        return operationOutput
    }

    private static func readClock(enabledSink: EnabledAttributionSink) throws(SupervisorPerformanceAttributionError) -> UInt64 {
        do {
            return try enabledSink.unixEpochMillis()
        } catch let clockError {
            throw SupervisorPerformanceAttributionError.clockFailure(
                problem: String(describing: clockError))
        }
    }

    private func finishMeasurement(
        enabledSink: EnabledAttributionSink,
        operation: SupervisorPerformanceOperation,
        startedAtUnixMillis: UInt64,
        startedAtMonotonic: DispatchTime,
        measurement: SupervisorPerformanceMeasurement
    ) throws(SupervisorPerformanceAttributionError) {
        let endedAtUnixMillis: UInt64 = try SupervisorPerformanceAttributionLog.readClock(
            enabledSink: enabledSink)
        try self.recordCompletedMeasurement(
            enabledSink: enabledSink,
            operation: operation,
            startedAtUnixMillis: startedAtUnixMillis,
            startedAtMonotonic: startedAtMonotonic,
            endedAtUnixMillis: endedAtUnixMillis,
            measurement: measurement)
    }

    private func recordCompletedMeasurement(
        enabledSink: EnabledAttributionSink,
        operation: SupervisorPerformanceOperation,
        startedAtUnixMillis: UInt64,
        startedAtMonotonic: DispatchTime,
        measurement: SupervisorPerformanceMeasurement
    ) throws(SupervisorPerformanceAttributionError) {
        let endedAtUnixMillis: UInt64 = try SupervisorPerformanceAttributionLog.readClock(
            enabledSink: enabledSink)
        try self.recordCompletedMeasurement(
            enabledSink: enabledSink,
            operation: operation,
            startedAtUnixMillis: startedAtUnixMillis,
            startedAtMonotonic: startedAtMonotonic,
            endedAtUnixMillis: endedAtUnixMillis,
            measurement: measurement)
    }

    private func recordCompletedMeasurement(
        enabledSink: EnabledAttributionSink,
        operation: SupervisorPerformanceOperation,
        startedAtUnixMillis: UInt64,
        startedAtMonotonic: DispatchTime,
        endedAtUnixMillis: UInt64,
        measurement: SupervisorPerformanceMeasurement
    ) throws(SupervisorPerformanceAttributionError) {
        guard measurement.matchesOperation(operation) else {
            throw SupervisorPerformanceAttributionError.mismatchedOperationDetail
        }
        let completedAtMonotonic: UInt64 = DispatchTime.now().uptimeNanoseconds
        let elapsedNanoseconds: UInt64 = completedAtMonotonic > startedAtMonotonic.uptimeNanoseconds
            ? completedAtMonotonic - startedAtMonotonic.uptimeNanoseconds
            : 0
        let attributionRecord: SupervisorPerformanceAttributionRecord = SupervisorPerformanceAttributionRecord(
            operationName: operation.wireName,
            startedAtUnixMillis: startedAtUnixMillis,
            endedAtUnixMillis: endedAtUnixMillis,
            elapsedNanoseconds: elapsedNanoseconds,
            outcomeName: measurement.outcome.wireName,
            catalogEntryCount: measurement.catalogEntryCount,
            downloadDetail: measurement.downloadDetail)
        try self.appendRecord(enabledSink: enabledSink, attributionRecord: attributionRecord)
    }

    private func appendRecord(
        enabledSink: EnabledAttributionSink,
        attributionRecord: SupervisorPerformanceAttributionRecord
    ) throws(SupervisorPerformanceAttributionError) {
        // Serialization stays outside the critical section so concurrent
        // operations serialize only the append required to keep each JSON
        // line intact.
        var wireWriter: JsonWireWriter = JsonWireWriter()
        do {
            try wireWriter.appendValue(attributionRecord.jsonlWireValue())
        } catch let serializationError {
            throw SupervisorPerformanceAttributionError.writerFailure(
                problem: String(describing: serializationError))
        }
        var recordBytes: Data = Data(wireWriter.serializedText.utf8)
        recordBytes.append(UInt8(ascii: "\n"))
        do {
            try enabledSink.appendSerializedLine(recordBytes)
        } catch let writeError {
            throw SupervisorPerformanceAttributionError.writerFailure(
                problem: String(describing: writeError))
        }
    }
}

/// The enabled log's shared writer and clock state; the writer append is
/// guarded so concurrent measured operations never interleave one row.
private final class EnabledAttributionSink: @unchecked Sendable {

    private let writerLock: NSLock

    private let writer: SupervisorAttributionWriterSink

    let unixEpochMillis: () throws -> UInt64

    init(
        writer: SupervisorAttributionWriterSink,
        unixEpochMillis: @escaping () throws -> UInt64
    ) {
        self.writerLock = NSLock()
        self.writer = writer
        self.unixEpochMillis = unixEpochMillis
    }

    func appendSerializedLine(_ recordBytes: Data) throws {
        self.writerLock.lock()
        defer { self.writerLock.unlock(); }
        try self.writer.write(recordBytes)
        try self.writer.flush()
    }
}
