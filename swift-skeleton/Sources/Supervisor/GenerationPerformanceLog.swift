import Foundation

import AstronomicalConfig
import IpcProtocol
import os

/**
 * Append-only JSONL writer for generation performance records, mirroring the
 * Rust log from apps/supervisor/src/generation_performance_log.rs.
 *
 * Opens `performance.jsonl` in the configured log directory at creation and
 * appends one line per completed generation. The volume is low (one line per
 * request), so a synchronous append-then-flush keeps records durable
 * immediately without ever blocking inference on a slow disk for long.
 * Write failures warn and drop the row: the serving path keeps running.
 */
public final class GenerationPerformanceLog {

    private static let performanceLogLogger: Logger = Logger(
        subsystem: "dev.astronomical.supervisor",
        category: "performance-log")

    private let appendableFile: FileHandle?

    private init(appendableFile: FileHandle?) {
        self.appendableFile = appendableFile
    }

    /**
     * Opens (or creates) the performance log file in the given directory.
     *
     * - Parameters:
     *   - logDirectory: the instance's logs directory; must already exist.
     * - Throws: a Cocoa error when the file cannot be created or opened for append.
     */
    public static func open(logDirectory: FilePath) throws -> GenerationPerformanceLog {
        let performanceLogPath: FilePath = logDirectory.appending(component: "performance.jsonl")
        let performanceLogUrl: URL = URL(fileURLWithPath: performanceLogPath.string)
        if FileManager.default.fileExists(atPath: performanceLogUrl.path) == false {
            let isCreated: Bool = FileManager.default.createFile(
                atPath: performanceLogUrl.path,
                contents: nil)
            guard isCreated else {
                throw CocoaError(.fileNoSuchFile)
            }
        }
        guard let appendableFile: FileHandle = FileHandle(forWritingAtPath: performanceLogUrl.path) else {
            throw CocoaError(.fileNoSuchFile)
        }
        do {
            try appendableFile.seekToEnd()
        } catch {
            do {
                try appendableFile.close()
            } catch let closeError {
                GenerationPerformanceLog.performanceLogLogger.warning(
                    "failed to close the performance log after a rejected open: \(String(describing: closeError), privacy: .public)")
            }
            throw error
        }
        return GenerationPerformanceLog(appendableFile: appendableFile)
    }

    /// Creates a log that drops every row; the handle that reports an
    /// unavailable worker never serves a generation to attribute.
    public static func disabled() -> GenerationPerformanceLog {
        return GenerationPerformanceLog(appendableFile: nil)
    }

    /**
     * Appends one performance record as a JSON line to the log file.
     *
     * The write hands the full line to the kernel in one call, which is the
     * durability contract of the Rust writeln!+flush pair. This is correct
     * for the low-volume performance log (one write per request) and avoids
     * data loss on an unclean shutdown.
     */
    public func record(_ performanceRecord: GenerationPerformanceRecord) -> Void {
        self.appendSerializedRecord(performanceRecord.jsonlWireValue())
    }

    /// Appends one finalized image-generation performance record.
    public func recordImage(_ imageRecord: ImageGenerationPerformanceRecord) -> Void {
        self.appendSerializedRecord(imageRecord.jsonlWireValue())
    }

    private func appendSerializedRecord(_ recordWireValue: JsonWireValue) -> Void {
        guard self.appendableFile != nil else {
            return;
        }
        let serializedLine: String;
        do {
            var wireWriter: JsonWireWriter = JsonWireWriter()
            try wireWriter.appendValue(recordWireValue)
            serializedLine = wireWriter.serializedText
        } catch let serializationError {
            GenerationPerformanceLog.performanceLogLogger.warning(
                "failed to serialize generation performance record: \(String(describing: serializationError), privacy: .public)")
            return
        }
        self.appendLine(serializedLine)
    }

    private func appendLine(_ serializedLine: String) -> Void {
        guard let appendableFile: FileHandle = self.appendableFile else {
            return
        }
        var lineBytes: Data = Data(serializedLine.utf8)
        lineBytes.append(UInt8(ascii: "\n"))
        do {
            // FileHandle.write hands the full line to the kernel in one call,
            // which is the durability contract of Rust's writeln!+flush pair.
            try appendableFile.write(contentsOf: lineBytes)
        } catch let writeError {
            GenerationPerformanceLog.performanceLogLogger.warning(
                "failed to write generation performance record: \(String(describing: writeError), privacy: .public)")
        }
    }
}

/// Wall-clock source shared by the performance and completion attribution logs.
public enum PerformanceLogClock {

    /// Returns the current time as milliseconds since the Unix epoch.
    public static func unixEpochMillis() -> UInt64 {
        let wallClockNanoseconds: Double = Date().timeIntervalSince1970 * 1_000_000_000
        guard wallClockNanoseconds > 0 else {
            return 0
        }
        return UInt64(wallClockNanoseconds) / 1_000_000
    }
}
