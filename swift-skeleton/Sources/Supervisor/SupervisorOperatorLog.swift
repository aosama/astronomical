import Foundation

import AstronomicalConfig

/**
 * The supervisor's own rotating operator log, migrating the file half of
 * apps/supervisor/src/main.rs's tracing appender: hour-stamped
 * `supervisor.<hour>.log` files inside the instance logging directory,
 * retention-bounded, written best-effort and lossy so a slow diagnostic
 * disk can never stall the serving path — the same contract the Rust
 * non-blocking writer gave. Worker-failure diagnostics persist here, which
 * is what makes an unexplained worker exit diagnosable after the fact.
 */
public final class SupervisorOperatorLog: @unchecked Sendable {

    private static let FILE_NAME_PREFIX: String = "supervisor."
    private static let FILE_NAME_SUFFIX: String = ".log"
    private static let DISABLED_LOG_DIRECTORY: String = "/dev/null-supervisor-operator-log"

    private let logDirectoryPath: String
    private let retainedLogFileCount: Int
    private let isRecordingEnabled: Bool
    private let logLock: NSLock
    private var currentHourStamp: String
    private var currentLogFilePath: String?

    private init(
        logDirectoryPath: String,
        retainedLogFileCount: Int,
        isRecordingEnabled: Bool
    ) {
        self.logDirectoryPath = logDirectoryPath
        self.retainedLogFileCount = max(1, retainedLogFileCount)
        self.isRecordingEnabled = isRecordingEnabled
        self.logLock = NSLock()
        self.currentHourStamp = ""
        self.currentLogFilePath = nil
    }

    /// A log that records nothing; every supervisor constructed without an
    /// explicit log directory gets this, so no call site needs a nil check.
    public static func disabled() -> SupervisorOperatorLog {
        return SupervisorOperatorLog(
            logDirectoryPath: SupervisorOperatorLog.DISABLED_LOG_DIRECTORY,
            retainedLogFileCount: 1,
            isRecordingEnabled: false)
    }

    /**
     * Opens the rotating log for one instance logging directory. The
     * directory is expected to exist (the daemon creates it before opening
     * any log); the first recorded line creates the current hour's file.
     */
    public static func open(
        logDirectory: FilePath,
        retainedLogFileCount: Int = LoggingConfig.defaultRetainedLogFiles
    ) -> SupervisorOperatorLog {
        return SupervisorOperatorLog(
            logDirectoryPath: logDirectory.string,
            retainedLogFileCount: retainedLogFileCount,
            isRecordingEnabled: true)
    }

    /// Appends one timestamped diagnostic line to the current hour's file,
    /// rotating on the hour boundary and pruning files beyond the retention
    /// bound. Every failure is swallowed: operator diagnostics must never
    /// take down or slow the serving path.
    public func recordDiagnosticLine(_ diagnosticLine: String) -> Void {
        if !self.isRecordingEnabled {
            return
        }
        self.logLock.lock()
        defer { self.logLock.unlock() }
        let recordedAt: Date = Date()
        let hourStamp: String = SupervisorOperatorLog.hourStampFormatter.string(from: recordedAt)
        var shouldPruneRotatedFiles: Bool = false
        if hourStamp != self.currentHourStamp || self.currentLogFilePath == nil {
            self.currentHourStamp = hourStamp
            self.currentLogFilePath = self.logDirectoryPath + "/"
                + SupervisorOperatorLog.FILE_NAME_PREFIX + hourStamp
                + SupervisorOperatorLog.FILE_NAME_SUFFIX
            shouldPruneRotatedFiles = true
        }
        guard let currentLogFilePath: String = self.currentLogFilePath else {
            return
        }
        let composedLine: String = SupervisorOperatorLog.lineStampFormatter.string(from: recordedAt)
            + " " + diagnosticLine + "\n"
        let lineData: Data = Data(composedLine.utf8)
        if !FileManager.default.fileExists(atPath: currentLogFilePath) {
            FileManager.default.createFile(atPath: currentLogFilePath, contents: lineData)
        } else {
            guard let logFileHandle: FileHandle = FileHandle(forWritingAtPath: currentLogFilePath) else {
                return
            }
            defer { try? logFileHandle.close() }
            _ = try? logFileHandle.seekToEnd()
            _ = try? logFileHandle.write(contentsOf: lineData)
        }
        if shouldPruneRotatedFiles {
            // Prune only after the current file exists, so the retention
            // bound counts the just-rotated file exactly like Rust's
            // max_log_files counts the active one.
            self.pruneRotatedFilesWhileLocked()
        }
    }

    /// Deletes the oldest rotated files beyond the retention bound; the
    /// caller holds the lock, and every failure is ignored as best-effort.
    private func pruneRotatedFilesWhileLocked() -> Void {
        let logFilePaths: [String] = (try? FileManager.default.contentsOfDirectory(atPath: self.logDirectoryPath))?
            .filter({ (fileName: String) -> Bool in
                return fileName.hasPrefix(SupervisorOperatorLog.FILE_NAME_PREFIX)
                    && fileName.hasSuffix(SupervisorOperatorLog.FILE_NAME_SUFFIX)
            })
            .sorted(by: { (firstFileName: String, secondFileName: String) -> Bool in
                return firstFileName > secondFileName
            })
            .map({ (fileName: String) -> String in
                return self.logDirectoryPath + "/" + fileName
            }) ?? []
        for excessLogFilePath: String in logFilePaths.dropFirst(self.retainedLogFileCount) {
            try? FileManager.default.removeItem(atPath: excessLogFilePath)
        }
    }

    private static let hourStampFormatter: DateFormatter = SupervisorOperatorLog.makeFormatter("yyyyMMdd-HH")
    private static let lineStampFormatter: DateFormatter = SupervisorOperatorLog.makeFormatter("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")

    private static func makeFormatter(_ format: String) -> DateFormatter {
        let formatter: DateFormatter = DateFormatter()
        formatter.dateFormat = format
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "UTC")
        return formatter
    }
}
