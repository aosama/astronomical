import Darwin;
import Foundation;

/// Append-only writer for bounded attribution reports.
///
/// Logging stays switchable through configuration: opening with attribution
/// disabled never touches the filesystem, and every record call degrades to a
/// no-op, so code paths that do not request attribution allocate nothing.
public final class PerformanceAttributionLog {

    private var writer: FileHandle?;
    private var performanceAttributionLogPath: URL?;
    private var openedDeviceIdentifier: UInt64 = 0;
    private var openedInodeNumber: UInt64 = 0;

    /// Creates an inert writer for code paths that do not request attribution.
    public static func disabled() -> PerformanceAttributionLog {
        PerformanceAttributionLog(
            writer: nil,
            performanceAttributionLogPath: nil,
            openedDeviceIdentifier: 0,
            openedInodeNumber: 0);
    }

    /// Opens the report file only when attribution is enabled.
    public static func open(
        at performanceAttributionLogPath: URL,
        performanceAttributionEnabled: Bool
    ) throws -> PerformanceAttributionLog {
        if !performanceAttributionEnabled {
            return disabled();
        }
        let writer = try Self.openWriter(performanceAttributionLogPath);
        let identity = identity(ofOpenedFileAt: performanceAttributionLogPath);
        return PerformanceAttributionLog(
            writer: writer,
            performanceAttributionLogPath: performanceAttributionLogPath,
            openedDeviceIdentifier: identity?.deviceIdentifier ?? 0,
            openedInodeNumber: identity?.inodeNumber ?? 0);
    }

    private init(
        writer: FileHandle?,
        performanceAttributionLogPath: URL?,
        openedDeviceIdentifier: UInt64,
        openedInodeNumber: UInt64
    ) {
        self.writer = writer;
        self.performanceAttributionLogPath = performanceAttributionLogPath;
        self.openedDeviceIdentifier = openedDeviceIdentifier;
        self.openedInodeNumber = openedInodeNumber;
    }

    deinit {
        try? writer?.close();
    }

    /// Serializes and flushes one report, retrying a failed writer on the next report.
    public func record(_ performanceAttributionReport: PerformanceAttributionReport) throws
    -> Void {
        try reopenIfLogPathChanged();
        guard let performanceAttributionWriter = writer else {
            return;
        }
        let serializedReport: Data;
        do {
            serializedReport = try JSONEncoder().encode(performanceAttributionReport);
        } catch {
            writer = nil;
            throw error;
        }
        var reportLine = serializedReport;
        reportLine.append(0x0A);
        do {
            try performanceAttributionWriter.write(contentsOf: reportLine);
        } catch {
            writer = nil;
            throw error;
        }
    }

    /// Restores a writer whose file was rotated, renamed, or deleted underneath
    /// it, so log rotation can never silently drop later reports.
    private func reopenIfLogPathChanged() throws -> Void {
        guard let performanceAttributionLogPath = performanceAttributionLogPath else {
            return;
        }
        let shouldReopen: Bool;
        if writer == nil {
            shouldReopen = true;
        } else if let currentIdentity = Self.identity(ofOpenedFileAt: performanceAttributionLogPath) {
            shouldReopen = currentIdentity.deviceIdentifier != openedDeviceIdentifier
                || currentIdentity.inodeNumber != openedInodeNumber;
        } else {
            shouldReopen = true;
        }
        if shouldReopen {
            try writer?.close();
            let writer = try Self.openWriter(performanceAttributionLogPath);
            self.writer = writer;
            let reopenedFileIdentity = Self.identity(ofOpenedFileAt: performanceAttributionLogPath);
            openedDeviceIdentifier = reopenedFileIdentity?.deviceIdentifier ?? 0;
            openedInodeNumber = reopenedFileIdentity?.inodeNumber ?? 0;
        }
    }

    private static func openWriter(_ performanceAttributionLogPath: URL) throws -> FileHandle {
        if !FileManager.default.fileExists(atPath: performanceAttributionLogPath.path) {
            FileManager.default.createFile(
                atPath: performanceAttributionLogPath.path,
                contents: nil);
        }
        return try FileHandle(forWritingTo: performanceAttributionLogPath);
    }

    /// This toolchain's Foundation exposes `deviceIdentifier` but no
    /// `inodeNumber` convenience member, so the stable raw key string is used.
    private static let inodeNumberAttributeKey = FileAttributeKey(
        rawValue: "NSFileinodeNumber");

    private static func identity(
        ofOpenedFileAt performanceAttributionLogPath: URL
    ) -> (deviceIdentifier: UInt64, inodeNumber: UInt64)? {
        // stat() is used instead of FileManager.attributesOfItem because this
        // toolchain's Foundation returns attribute values that no longer cast
        // to NSNumber, which silently forced a reopen (and offset-0 overwrite)
        // on every record.
        var fileStatus = stat();
        guard stat(performanceAttributionLogPath.path, &fileStatus) == 0 else {
            return nil;
        }
        return (
            UInt64(fileStatus.st_dev),
            fileStatus.st_ino);
    }
}
