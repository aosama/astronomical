import Foundation;

import IpcProtocol;

/// One bounded failure while reading a persistent safetensors header, port
/// of the Rust `PersistentSafetensorsHeaderError`.
public enum PersistentSafetensorsHeaderError: Error, Equatable, Sendable {

    case readFileMetadata(filePath: String, problem: String);

    case readHeaderBytes(filePath: String, problem: String);

    case headerLengthTooLarge(
        filePath: String,
        headerLengthBytes: UInt64,
        maximumHeaderLengthBytes: UInt64);

    case truncatedFile(
        filePath: String,
        expectedMinimumBytes: UInt64,
        actualFileSizeBytes: UInt64);

    case invalidHeaderJson(filePath: String, problem: String);

    public var errorDescription: String? {
        switch self {
        case let .readFileMetadata(filePath, problem):
            return "failed to read persistent safetensors file metadata at \(filePath): \(problem)";
        case let .readHeaderBytes(filePath, problem):
            return "failed to read persistent safetensors header bytes at \(filePath): \(problem)";
        case let .headerLengthTooLarge(filePath, headerLengthBytes, maximumHeaderLengthBytes):
            return "persistent safetensors header at \(filePath) is \(headerLengthBytes) bytes, "
                + "maximum \(maximumHeaderLengthBytes)";
        case let .truncatedFile(filePath, expectedMinimumBytes, actualFileSizeBytes):
            return "persistent safetensors file at \(filePath) is truncated: expected "
                + "\(expectedMinimumBytes), got \(actualFileSizeBytes)";
        case let .invalidHeaderJson(filePath, problem):
            return "persistent safetensors header at \(filePath) is not valid JSON: \(problem)";
        }
    }
}

/// Parsed header data retained after bounded JSON parsing, port of the Rust
/// `PersistentSafetensorsHeader`.
public struct PersistentSafetensorsHeader {

    /// The 1 MiB bound mirrors the Rust persistent-cache header ceiling:
    /// startup scanning must never deserialize unbounded headers.
    static let MAXIMUM_HEADER_LENGTH_BYTES: UInt64 = 1024 * 1024;

    public let tensorViewsByName: [String: SafetensorsFraming.TensorView];

    public let metadata: [String: String];

    public let dataSectionStartBytes: UInt64;

    public let fileSizeBytes: UInt64;

    /// Reads and parses one bounded persistent safetensors header without
    /// touching payload bytes.
    public static func read(
        fileUrl: URL
    ) throws -> PersistentSafetensorsHeader {
        let filePath: String = fileUrl.path;
        let fileAttributes: [FileAttributeKey: Any];
        do {
            fileAttributes = try FileManager.default.attributesOfItem(atPath: filePath);
        } catch {
            throw PersistentSafetensorsHeaderError.readFileMetadata(
                filePath: filePath, problem: String(describing: error));
        }
        let fileSizeBytes: UInt64 = (fileAttributes[.size] as? NSNumber)?.uint64Value ?? 0;
        let boundedJsonHeader: SafetensorsFraming.BoundedJsonHeader;
        let fileHandle: FileHandle;
        do {
            fileHandle = try FileHandle(forReadingFrom: fileUrl);
        } catch {
            throw PersistentSafetensorsHeaderError.readHeaderBytes(
                filePath: filePath, problem: String(describing: error));
        }
        defer { try? fileHandle.close(); }
        do {
            boundedJsonHeader = try SafetensorsFraming.readBoundedJsonHeader(
                fileHandle: fileHandle,
                fileSizeBytes: fileSizeBytes,
                maximumHeaderLengthBytes: PersistentSafetensorsHeader.MAXIMUM_HEADER_LENGTH_BYTES);
        } catch let framingError as SafetensorsFraming.BoundedHeaderError {
            throw PersistentSafetensorsHeader.persistentHeaderError(
                framingError: framingError, filePath: filePath);
        } catch {
            throw PersistentSafetensorsHeaderError.readHeaderBytes(
                filePath: filePath, problem: String(describing: error));
        }
        var metadata: [String: String] = [:];
        if let metadataJsonValue: JsonWireValue = boundedJsonHeader.metadataJsonValue {
            do {
                let metadataObject: JsonWireObject = try JsonWireValue
                    .extractObject(metadataJsonValue);
                for metadataEntry: (key: String, value: JsonWireValue) in metadataObject.entries {
                    metadata[metadataEntry.key] = try JsonWireValue
                        .extractString(metadataEntry.value);
                }
            } catch let wireProblem as JsonWireProblem {
                throw PersistentSafetensorsHeaderError.invalidHeaderJson(
                    filePath: filePath, problem: wireProblem.description);
            } catch {
                throw PersistentSafetensorsHeaderError.invalidHeaderJson(
                    filePath: filePath, problem: String(describing: error));
            }
        }
        var tensorViewsByName: [String: SafetensorsFraming.TensorView] = [:];
        for tensorEntry: (tensorName: String, headerValue: JsonWireValue)
            in boundedJsonHeader.tensorJsonValues {
            let tensorView: SafetensorsFraming.TensorView;
            do {
                tensorView = try SafetensorsFraming.TensorView
                    .decoded(wireValue: tensorEntry.headerValue);
            } catch let wireProblem as JsonWireProblem {
                throw PersistentSafetensorsHeaderError.invalidHeaderJson(
                    filePath: filePath, problem: wireProblem.description);
            } catch {
                throw PersistentSafetensorsHeaderError.invalidHeaderJson(
                    filePath: filePath, problem: String(describing: error));
            }
            tensorViewsByName[tensorEntry.tensorName] = tensorView;
        }
        return PersistentSafetensorsHeader(
            tensorViewsByName: tensorViewsByName,
            metadata: metadata,
            dataSectionStartBytes: boundedJsonHeader.dataSectionStartBytes,
            fileSizeBytes: boundedJsonHeader.fileSizeBytes);
    }

    private static func persistentHeaderError(
        framingError: SafetensorsFraming.BoundedHeaderError,
        filePath: String
    ) -> PersistentSafetensorsHeaderError {
        switch framingError {
        case let .readLengthPrefix(problem):
            return .readHeaderBytes(filePath: filePath, problem: problem);
        case let .headerLengthTooLarge(headerLengthBytes, maximumHeaderLengthBytes):
            return .headerLengthTooLarge(
                filePath: filePath,
                headerLengthBytes: headerLengthBytes,
                maximumHeaderLengthBytes: maximumHeaderLengthBytes);
        case let .headerBeyondFile(headerEndOffsetBytes, fileSizeBytes):
            return .truncatedFile(
                filePath: filePath,
                expectedMinimumBytes: headerEndOffsetBytes,
                actualFileSizeBytes: fileSizeBytes);
        case let .readHeader(problem):
            return .readHeaderBytes(filePath: filePath, problem: problem);
        case let .invalidHeaderJson(problem):
            return .invalidHeaderJson(filePath: filePath, problem: problem);
        }
    }
}
