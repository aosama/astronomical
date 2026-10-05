import Foundation;
import Darwin;

/**
 * Size-bounded reads of discovery artifacts: a file must exist, be a regular
 * file, and be non-empty and within the caller's byte limit before any
 * content is buffered, mirroring `bounded_artifact_file.rs`.
 */
internal enum BoundedArtifactFile {
    internal enum FileReadError: Error, CustomStringConvertible {
        case unavailable(filePath: FilePath);
        case invalidSizeOrType(filePath: FilePath, actualBytes: UInt64, maximumBytes: UInt64);

        internal var description: String {
            switch (self) {
            case .unavailable(let filePath):
                return "artifact file is unavailable: \(filePath)";
            case .invalidSizeOrType(let filePath, let actualBytes, let maximumBytes):
                return "artifact file \(filePath) is \(actualBytes) bytes, which is empty or beyond the \(maximumBytes) byte limit";
            }
        }
    }

    internal enum DocumentReadError: Error, CustomStringConvertible {
        case file(BoundedArtifactFile.FileReadError);
        case malformedJson(filePath: FilePath, underlyingDescription: String);

        internal var description: String {
            switch (self) {
            case .file(let fileReadError): return "\(fileReadError)";
            case .malformedJson(let filePath, let underlyingDescription):
                return "artifact file \(filePath) is not valid JSON: \(underlyingDescription)";
            }
        }
    }

    internal static func readJsonDocument(fileAtPath: FilePath, maximumBytes: UInt64) throws -> Any {
        let documentBytes: Data = try self.readBoundedNonemptyFile(
            filePath: fileAtPath,
            maximumBytes: maximumBytes
        );
        do {
            return try DiscoveryStrictJsonDocument.parseDocument(bytes: documentBytes);
        } catch let parseError as DiscoveryStrictJsonDocument.ParseError {
            throw DocumentReadError.malformedJson(
                filePath: fileAtPath,
                underlyingDescription: parseError.description
            );
        }
    }

    internal static func readBoundedNonemptyFile(filePath: FilePath, maximumBytes: UInt64) throws -> Data {
        var fileStatus: stat = stat();
        guard Darwin.fstatat(Darwin.AT_FDCWD, filePath.string, &fileStatus, 0) == 0 else {
            throw FileReadError.unavailable(filePath: filePath);
        }
        let fileSizeBytes: UInt64 = UInt64(fileStatus.st_size);
        guard (fileStatus.st_mode & S_IFMT) == S_IFREG, fileSizeBytes > 0, fileSizeBytes <= maximumBytes else {
            throw FileReadError.invalidSizeOrType(
                filePath: filePath,
                actualBytes: fileSizeBytes,
                maximumBytes: maximumBytes
            );
        }
        let fileHandle: FileHandle;
        do {
            fileHandle = try FileHandle(forReadingFrom: URL(fileURLWithPath: filePath.string));
        } catch {
            throw FileReadError.unavailable(filePath: filePath);
        }
        defer {
            fileHandle.closeFile();
        }
        let fileBytes: Data;
        do {
            fileBytes = try fileHandle.read(upToCount: Int(maximumBytes) + 1) ?? Data();
        } catch {
            throw FileReadError.unavailable(filePath: filePath);
        }
        if fileBytes.isEmpty || fileBytes.count > Int(maximumBytes) {
            throw FileReadError.invalidSizeOrType(
                filePath: filePath,
                actualBytes: UInt64(fileBytes.count),
                maximumBytes: maximumBytes
            );
        }
        return fileBytes;
    }
}
