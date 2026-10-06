import XCTest;
import ModelServing;
import IpcProtocol;

/// Behavioral journeys for the bounded safetensors header reader, mirroring
/// the framing contract of crates/model-serving/src/safetensors/header.rs:
/// little-endian length prefix, header-size bound, header-within-file bound,
/// duplicate-key rejection, and `__metadata__` extraction.
final class SafetensorsHeaderTests: XCTestCase {

    func testShouldParseABoundedHeaderWithOneTensorAndMetadata() throws {
        let (fileHandle, fileSizeBytes): (FileHandle, UInt64) = try Self.openFramedFile(
            headerJsonText: """
            {"__metadata__":{"format":"pt"},"tensor.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}
            """, payloadByteCount: 12);
        defer { fileHandle.closeFile(); }
        let parsedHeader: SafetensorsFraming.BoundedJsonHeader = try SafetensorsFraming.readBoundedJsonHeader(
            fileHandle: fileHandle, fileSizeBytes: fileSizeBytes,
            maximumHeaderLengthBytes: 1024 * 1024);
        XCTAssertEqual(parsedHeader.dataSectionStartBytes, UInt64(fileSizeBytes) - 12);
        XCTAssertEqual(parsedHeader.fileSizeBytes, fileSizeBytes);
        XCTAssertEqual(parsedHeader.tensorJsonValues.count, 1);
        XCTAssertEqual(parsedHeader.tensorJsonValues[0].tensorName, "tensor.weight");
        let tensorView: SafetensorsFraming.TensorView = try SafetensorsFraming.TensorView.decoded(
            wireValue: parsedHeader.tensorJsonValues[0].headerValue);
        XCTAssertEqual(tensorView.dtype, "BF16");
        XCTAssertEqual(tensorView.shape, [2, 3]);
        XCTAssertEqual(tensorView.dataStartOffset(), 0);
        XCTAssertEqual(tensorView.dataEndOffset(), 12);
        let metadataValue: JsonWireValue = try XCTUnwrap(parsedHeader.metadataJsonValue);
        guard case let .object(metadataObject) = metadataValue else {
            XCTFail("expected a metadata object");
            return;
        }
        XCTAssertEqual(metadataObject.value(forKey: "format"), .string("pt"));
    }

    func testShouldRejectDuplicateHeaderKeysBeforeAnyConsumerCanClassifyThem() throws {
        let rawDuplicateHeaderBytes: Array<UInt8> = Array(#"{"tensor.weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"tensor.weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#.utf8);
        let framedFileBytes: Array<UInt8> = Self.frameBytes(
            headerJsonBytes: rawDuplicateHeaderBytes, payloadByteCount: 2);
        let framedFilePath: URL = try Self.writeTemporaryFile(framedFileBytes);
        defer { try? FileManager.default.removeItem(at: framedFilePath); }
        let fileHandle: FileHandle = try FileHandle(forReadingFrom: framedFilePath);
        defer { fileHandle.closeFile(); }
        do {
            _ = try SafetensorsFraming.readBoundedJsonHeader(
                fileHandle: fileHandle, fileSizeBytes: UInt64(framedFileBytes.count),
                maximumHeaderLengthBytes: 1024 * 1024);
            XCTFail("duplicate safetensors header keys must be rejected");
            return;
        } catch let headerError as SafetensorsFraming.BoundedHeaderError {
            guard case .invalidHeaderJson(let problem) = headerError else {
                XCTFail("expected InvalidHeaderJson, got \(headerError)");
                return;
            }
            XCTAssertTrue(
                problem.lowercased().contains("duplicate"),
                "the duplicate-key rejection must name the duplication, got: \(problem)");
        }
    }

    func testShouldRejectAHeaderLongerThanTheDeclaredBound() throws {
        let (fileHandle, fileSizeBytes): (FileHandle, UInt64) = try Self.openFramedFile(
            headerJsonText: #"{"tensor.weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
            payloadByteCount: 1);
        defer { fileHandle.closeFile(); }
        do {
            _ = try SafetensorsFraming.readBoundedJsonHeader(
                fileHandle: fileHandle, fileSizeBytes: fileSizeBytes,
                maximumHeaderLengthBytes: 8);
            XCTFail("a header beyond the declared bound must fail closed");
            return;
        } catch let headerError as SafetensorsFraming.BoundedHeaderError {
            guard case .headerLengthTooLarge = headerError else {
                XCTFail("expected HeaderLengthTooLarge, got \(headerError)");
                return;
            }
        }
    }

    func testShouldRejectAHeaderThatReachesBeyondTheFileEnd() throws {
        let (fileHandle, _) : (FileHandle, UInt64) = try Self.openFramedFile(
            headerJsonText: #"{"tensor.weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
            payloadByteCount: 1);
        defer { fileHandle.closeFile(); }
        do {
            _ = try SafetensorsFraming.readBoundedJsonHeader(
                fileHandle: fileHandle, fileSizeBytes: 4,
                maximumHeaderLengthBytes: 1024 * 1024);
            XCTFail("a header reaching past the file end must fail closed");
            return;
        } catch let headerError as SafetensorsFraming.BoundedHeaderError {
            guard case .headerBeyondFile(let headerEndOffsetBytes, let declaredFileSizeBytes) = headerError else {
                XCTFail("expected HeaderBeyondFile, got \(headerError)");
                return;
            }
            XCTAssertGreaterThan(headerEndOffsetBytes, declaredFileSizeBytes);
        }
    }

    private static func openFramedFile(
        headerJsonText: String, payloadByteCount: Int) throws -> (FileHandle, UInt64) {
        let framedFileBytes: Array<UInt8> = frameBytes(
            headerJsonBytes: Array(headerJsonText.utf8), payloadByteCount: payloadByteCount);
        let framedFilePath: URL = try writeTemporaryFile(framedFileBytes);
        let fileHandle: FileHandle = try FileHandle(forReadingFrom: framedFilePath);
        return (fileHandle, UInt64(framedFileBytes.count));
    }

    private static func frameBytes(
        headerJsonBytes: Array<UInt8>, payloadByteCount: Int) -> Array<UInt8> {
        var framedBytes: Array<UInt8> = Array();
        var littleEndianHeaderLength: UInt64 = UInt64(headerJsonBytes.count);
        withUnsafeBytes(of: &littleEndianHeaderLength) { (valueBuffer: UnsafeRawBufferPointer) -> Void in
            framedBytes.append(contentsOf: Array(valueBuffer));
        };
        framedBytes.append(contentsOf: headerJsonBytes);
        framedBytes.append(contentsOf: Array<UInt8>(repeating: 0, count: payloadByteCount));
        return framedBytes;
    }

    private static func writeTemporaryFile(_ fileBytes: Array<UInt8>) throws -> URL {
        let temporaryDirectoryURL: URL = URL(fileURLWithPath: NSTemporaryDirectory());
        let framedFileURL: URL = temporaryDirectoryURL
            .appendingPathComponent("safetensors-header-\(UUID().uuidString).safetensors");
        try Data(fileBytes).write(to: framedFileURL);
        return framedFileURL;
    }
}
