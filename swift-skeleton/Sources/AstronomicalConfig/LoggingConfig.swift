import Foundation;

/// Log-level vocabulary shared between the runtime logging surface and the
/// diagnostics config wire field, porting crates/config/src/logging_config.rs.
/// The wire field is a plain String, so the mapping helpers are the only way
/// level text enters or leaves this enum.
internal enum LogLevel: Equatable {

    case error;
    case warn;
    case info;
    case debug;
    case trace;

    internal static let defaultValue: LogLevel = .warn;

    internal static let wireNameError: String = "error";
    internal static let wireNameWarn: String = "warn";
    internal static let wireNameInfo: String = "info";
    internal static let wireNameDebug: String = "debug";
    internal static let wireNameTrace: String = "trace";

    internal func asStr() -> String {
        switch self {
        case .error: return LogLevel.wireNameError;
        case .warn: return LogLevel.wireNameWarn;
        case .info: return LogLevel.wireNameInfo;
        case .debug: return LogLevel.wireNameDebug;
        case .trace: return LogLevel.wireNameTrace;
        }
    }

    internal static func fromWireName(_ wireName: String) -> LogLevel? {
        switch wireName {
        case wireNameError: return .error;
        case wireNameWarn: return .warn;
        case wireNameInfo: return .info;
        case wireNameDebug: return .debug;
        case wireNameTrace: return .trace;
        default: return nil;
        }
    }
}

/// Runtime logging configuration, porting LoggingConfig from logging_config.rs.
internal struct LoggingConfig {

    /// Each buffered log line is flushed once the buffer reaches this many
    /// lines; the value comes straight from LOG_BUFFERED_LINE_LIMIT.
    internal static let logBufferedLineLimit: Int = 1024;

    /// The facade keeps this many rotated log files before pruning the rest.
    internal static let defaultRetainedLogFiles: Int = 7;

    internal let bufferedLineLimit: Int;
    internal let directory: FilePath;
    internal let level: LogLevel;
    internal let retainedFiles: Int;

    internal init(directory: FilePath, level: LogLevel, retainedFiles: Int) {
        self.bufferedLineLimit = LoggingConfig.logBufferedLineLimit;
        self.directory = directory;
        self.level = level;
        self.retainedFiles = retainedFiles;
    }
}
