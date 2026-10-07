import Foundation

import AstronomicalConfig
import IpcProtocol
import os

/**
 * Append-only JSONL writer for completion attribution records, gated by
 * configuration, mirroring the Rust log from
 * apps/supervisor/src/completion_attribution_log.rs.
 *
 * The completion attribution log captures *what* a chat generation emitted —
 * each tool call's function name and arguments JSON plus the completion
 * reason — so tool-call argument pollution, foreign-dialect regressions, and
 * fail-closed retry loops are diagnosable instead of invisible. It is the
 * symmetric sibling of the generation performance log: performance
 * attribution answers "how well did it run?" (timing); completion
 * attribution answers "what did it emit?" (the tool calls and arguments).
 *
 * The log is off by default. An operator enables it through the
 * `diagnostics.completion_attribution_enabled` configuration flag. When
 * disabled, the sink is nil and `recordCompletion` is a no-op with no
 * allocation on the generation path.
 */
public final class CompletionAttributionLog {

    private static let completionLogLogger: Logger = Logger(
        subsystem: "dev.astronomical.supervisor",
        category: "completion-log")

    private let appendableFile: FileHandle?

    private init(appendableFile: FileHandle?) {
        self.appendableFile = appendableFile
    }

    /// Creates a no-overhead log for the disabled configuration.
    public static func disabled() -> CompletionAttributionLog {
        return CompletionAttributionLog(appendableFile: nil)
    }

    /**
     * Opens (or creates) the completion log when enabled.
     *
     * Returns a disabled log (no file, no writes) when the flag is false so
     * normal inference never touches the attribution path.
     *
     * - Parameters:
     *   - logDirectory: the instance's logs directory; must already exist.
     *   - completionAttributionEnabled: the resolved diagnostics flag.
     * - Throws: a Cocoa error when the file cannot be created or opened while enabled.
     */
    public static func open(
        logDirectory: FilePath,
        completionAttributionEnabled: Bool
    ) throws -> CompletionAttributionLog {
        guard completionAttributionEnabled else {
            return CompletionAttributionLog.disabled()
        }
        let completionLogPath: FilePath = logDirectory.appending(component: "completion.jsonl")
        let completionLogUrl: URL = URL(fileURLWithPath: completionLogPath.string)
        if FileManager.default.fileExists(atPath: completionLogUrl.path) == false {
            let isCreated: Bool = FileManager.default.createFile(
                atPath: completionLogUrl.path,
                contents: nil)
            guard isCreated else {
                throw CocoaError(.fileNoSuchFile)
            }
        }
        guard let appendableFile: FileHandle = FileHandle(forWritingAtPath: completionLogUrl.path) else {
            throw CocoaError(.fileNoSuchFile)
        }
        do {
            try appendableFile.seekToEnd()
        } catch {
            do {
                try appendableFile.close()
            } catch let closeError {
                CompletionAttributionLog.completionLogLogger.warning(
                    "failed to close the completion log after a rejected open: \(String(describing: closeError), privacy: .public)")
            }
            throw error
        }
        return CompletionAttributionLog(appendableFile: appendableFile)
    }

    /// Whether the log is enabled and writing rows.
    public var isEnabled: Bool {
        return self.appendableFile != nil
    }

    /**
     * Records one completion from the raw emitted tool calls.
     *
     * Binds the accumulated raw tool calls into bounded records and writes the
     * row when enabled. The caller supplies the timestamp so the completion
     * event owns request-correlation time, mirroring the performance log.
     */
    public func recordCompletion(
        timestampMillis: UInt64,
        requestId: UInt64,
        modelId: String,
        completionReason: String,
        completedToolCalls: Array<CompletedToolCall>
    ) -> Void {
        guard self.appendableFile != nil else {
            return
        }
        let boundedToolCalls: Array<CompletionToolCallRecord> = completedToolCalls.map(
            { (completedToolCall: CompletedToolCall) -> CompletionToolCallRecord in
                return CompletionToolCallRecord.fromArguments(
                    toolCallIndex: completedToolCall.toolCallIndex,
                    functionName: completedToolCall.functionName,
                    argumentsJson: completedToolCall.argumentsJson)
            })
        let attributionRecord: CompletionAttributionRecord = CompletionAttributionRecord(
            timestampMillis: timestampMillis,
            requestId: requestId,
            modelId: modelId,
            completionReason: completionReason,
            toolCalls: boundedToolCalls)
        self.record(attributionRecord)
    }

    /// Records one completion at the request's wall-clock time.
    public func recordCompletionAtNow(
        requestId: UInt64,
        modelId: String,
        completionReason: String,
        completedToolCalls: Array<CompletedToolCall>
    ) -> Void {
        self.recordCompletion(
            timestampMillis: PerformanceLogClock.unixEpochMillis(),
            requestId: requestId,
            modelId: modelId,
            completionReason: completionReason,
            completedToolCalls: completedToolCalls)
    }

    /**
     * Appends one completion record as a JSON line when enabled.
     *
     * No-op when disabled. The write hands the full line to the kernel in one
     * call so records are durable immediately, matching the generation
     * performance log contract.
     */
    public func record(_ attributionRecord: CompletionAttributionRecord) -> Void {
        guard let appendableFile: FileHandle = self.appendableFile else {
            return
        }
        let serializedLine: String
        do {
            var wireWriter: JsonWireWriter = JsonWireWriter()
            try wireWriter.appendValue(attributionRecord.jsonlWireValue())
            serializedLine = wireWriter.serializedText
        } catch let serializationError {
            CompletionAttributionLog.completionLogLogger.warning(
                "failed to serialize completion attribution record: \(String(describing: serializationError), privacy: .public)")
            return
        }
        var lineBytes: Data = Data(serializedLine.utf8)
        lineBytes.append(UInt8(ascii: "\n"))
        do {
            try appendableFile.write(contentsOf: lineBytes)
        } catch let writeError {
            CompletionAttributionLog.completionLogLogger.warning(
                "failed to write completion attribution record: \(String(describing: writeError), privacy: .public)")
        }
    }
}
