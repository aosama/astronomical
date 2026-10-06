import Foundation;
import IpcProtocol;

/// Shared bounded safetensors framing and raw JSON-header parsing, port of
/// crates/model-serving/src/safetensors/header.rs. The duplicate-key rejection
/// the Rust side implements with its duplicate-aware JSON visitor is inherent
/// in the shared wire parser, which rejects a repeated object key before any
/// consumer can classify metadata or apply family normalization.
public enum SafetensorsFraming {

    /// Size of the safetensors little-endian header-length prefix.
    public static let SAFETENSORS_HEADER_LENGTH_PREFIX_BYTES: UInt64 = 8;

    private static let MAXIMUM_DUPLICATE_HEADER_KEY_CHARACTERS: Int = 256;

    /// Raw safetensors header entries after a bounded file read and JSON parse.
    public struct BoundedJsonHeader {
        /// Raw per-tensor header entries in parse order (byte-sorted, since the
        /// strict wire parser emits sorted objects).
        public let tensorJsonValues: Array<(tensorName: String, headerValue: JsonWireValue)>;
        /// The `__metadata__` entry when the header declares one.
        public let metadataJsonValue: JsonWireValue?;
        /// Absolute byte offset where the payload data section begins.
        public let dataSectionStartBytes: UInt64;
        /// Total file size in bytes.
        public let fileSizeBytes: UInt64;
    }

    /// Framing or raw-JSON failures shared by safetensors consumers.
    public enum BoundedHeaderError: Error, Equatable {
        case readLengthPrefix(problem: String);
        case headerLengthTooLarge(headerLengthBytes: UInt64, maximumHeaderLengthBytes: UInt64);
        case headerBeyondFile(headerEndOffsetBytes: UInt64, fileSizeBytes: UInt64);
        case readHeader(problem: String);
        case invalidHeaderJson(problem: String);

        public var errorDescription: String? {
            switch self {
            case .readLengthPrefix(let problem):
                return "failed to read the safetensors header length prefix: \(problem)";
            case .headerLengthTooLarge(let headerLengthBytes, let maximumHeaderLengthBytes):
                return "safetensors header is \(headerLengthBytes) bytes, exceeding \(maximumHeaderLengthBytes)";
            case .headerBeyondFile(let headerEndOffsetBytes, let fileSizeBytes):
                return "safetensors header ends at \(headerEndOffsetBytes), beyond the file size \(fileSizeBytes)";
            case .readHeader(let problem):
                return "failed to read the safetensors header: \(problem)";
            case .invalidHeaderJson(let problem):
                return problem;
            }
        }
    }

    /// One raw tensor declaration shared by artifact and persistent-cache readers.
    public struct TensorView: Equatable {
        public let dtype: String;
        public let shape: Array<Int>;
        public let dataOffsets: Array<UInt64>;

        public init(dtype: String, shape: Array<Int>, dataOffsets: Array<UInt64>) {
            self.dtype = dtype;
            self.shape = shape;
            self.dataOffsets = dataOffsets;
        }

        public static func decoded(wireValue: JsonWireValue) throws -> SafetensorsFraming.TensorView {
            let tensorObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
            try tensorObject.rejectUnknownFields(allowedFieldNames: ["dtype", "shape", "data_offsets"]);
            return SafetensorsFraming.TensorView(
                dtype: try tensorObject.decodeString(fieldName: "dtype"),
                shape: try tensorObject.decodeArray(
                    fieldName: "shape",
                    mappedElement: { (dimensionWireValue: JsonWireValue) throws -> Int in
                        let dimensionValue: UInt64 = try JsonWireValue.extractUInt64(dimensionWireValue);
                        guard dimensionValue <= UInt64(Int.max) else {
                            throw JsonWireProblem.invalidType(
                                expectedTypeName: "usize", found: "integer `\(dimensionValue)`");
                        }
                        return Int(dimensionValue);
                    }),
                dataOffsets: try tensorObject.decodeArray(
                    fieldName: "data_offsets",
                    mappedElement: { (offsetWireValue: JsonWireValue) throws -> UInt64 in
                        return try JsonWireValue.extractUInt64(offsetWireValue);
                    }));
        }

        public func dataStartOffset() -> UInt64 {
            return self.dataOffsets[0];
        }

        public func dataEndOffset() -> UInt64 {
            return self.dataOffsets[1];
        }
    }

    /// Reads one bounded safetensors header without touching payload bytes.
    public static func readBoundedJsonHeader(
        fileHandle: FileHandle,
        fileSizeBytes: UInt64,
        maximumHeaderLengthBytes: UInt64) throws -> BoundedJsonHeader {
        guard let lengthPrefixData: Data = try Self.readExactly(
            fileHandle: fileHandle, fromOffset: 0, byteCount: Int(SAFETENSORS_HEADER_LENGTH_PREFIX_BYTES)) else {
            throw BoundedHeaderError.readLengthPrefix(
                problem: "the file ended before the 8-byte header length prefix");
        }
        let headerLengthBytes: UInt64 = lengthPrefixData.withUnsafeBytes { (rawBuffer: UnsafeRawBufferPointer) -> UInt64 in
            var littleEndianValue: UInt64 = 0;
            withUnsafeMutableBytes(of: &littleEndianValue) { (valueBuffer: UnsafeMutableRawBufferPointer) -> Void in
                valueBuffer.copyBytes(from: rawBuffer.prefix(MemoryLayout<UInt64>.size));
            };
            return UInt64(littleEndian: littleEndianValue);
        };
        if headerLengthBytes > maximumHeaderLengthBytes {
            throw BoundedHeaderError.headerLengthTooLarge(
                headerLengthBytes: headerLengthBytes, maximumHeaderLengthBytes: maximumHeaderLengthBytes);
        }
        let (dataSectionStartBytes, startOverflow) = SAFETENSORS_HEADER_LENGTH_PREFIX_BYTES
            .addingReportingOverflow(headerLengthBytes);
        if startOverflow {
            throw BoundedHeaderError.headerLengthTooLarge(
                headerLengthBytes: headerLengthBytes, maximumHeaderLengthBytes: maximumHeaderLengthBytes);
        }
        if dataSectionStartBytes > fileSizeBytes {
            throw BoundedHeaderError.headerBeyondFile(
                headerEndOffsetBytes: dataSectionStartBytes, fileSizeBytes: fileSizeBytes);
        }
        let headerJsonData: Data;
        do {
            guard let fullyReadHeaderData: Data = try Self.readExactly(
                fileHandle: fileHandle, fromOffset: UInt64(SAFETENSORS_HEADER_LENGTH_PREFIX_BYTES),
                byteCount: Int(headerLengthBytes)) else {
                throw BoundedHeaderError.readHeader(
                    problem: "the file ended before the full header");
            }
            headerJsonData = fullyReadHeaderData;
        } catch let readError as BoundedHeaderError {
            throw readError;
        } catch {
            throw BoundedHeaderError.readHeader(problem: String(describing: error));
        }
        let headerWireValue: JsonWireValue;
        do {
            headerWireValue = try JsonWireParser.parseDocument(documentBytes: headerJsonData);
        } catch let jsonWireProblem as JsonWireProblem {
            throw BoundedHeaderError.invalidHeaderJson(problem: jsonWireProblem.description);
        } catch {
            throw BoundedHeaderError.invalidHeaderJson(problem: "malformed header JSON");
        }
        guard case let .object(headerObject) = headerWireValue else {
            throw BoundedHeaderError.invalidHeaderJson(
                problem: JsonWireProblem.expectedObject(found: headerWireValue.foundDescription).description);
        }
        var tensorJsonValues: Array<(tensorName: String, headerValue: JsonWireValue)> = Array();
        var metadataJsonValue: JsonWireValue? = nil;
        for headerEntry: (key: String, value: JsonWireValue) in headerObject.entries {
            if headerEntry.key == "__metadata__" {
                metadataJsonValue = headerEntry.value;
                continue;
            }
            tensorJsonValues.append((headerEntry.key, headerEntry.value));
        }
        return BoundedJsonHeader(
            tensorJsonValues: tensorJsonValues,
            metadataJsonValue: metadataJsonValue,
            dataSectionStartBytes: dataSectionStartBytes,
            fileSizeBytes: fileSizeBytes);
    }

    /// Positioned bounded read mirroring the Rust `read_exact_at` loop: nil
    /// when the file ends before the requested bytes are complete.
    private static func readExactly(
        fileHandle: FileHandle, fromOffset startOffsetBytes: UInt64,
        byteCount: Int) throws -> Data? {
        fileHandle.seek(toFileOffset: startOffsetBytes);
        var completedBytes: Data = Data();
        while completedBytes.count < byteCount {
            guard let chunkData: Data = try fileHandle.read(upToCount: byteCount - completedBytes.count),
                chunkData.isEmpty == false else {
                return nil;
            }
            completedBytes.append(chunkData);
        }
        return completedBytes;
    }
}
