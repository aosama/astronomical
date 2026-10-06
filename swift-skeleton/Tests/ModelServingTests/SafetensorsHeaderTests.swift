import Foundation;
import ModelServing;
import IpcProtocol;
import Testing;
import JourneyCategories;

/**
 * Behavioral journeys for the bounded safetensors header reader, mirroring
 * the framing contract of crates/model-serving/src/safetensors/header.rs:
 * little-endian length prefix, header-size bound, header-within-file bound,
 * duplicate-key rejection, and `__metadata__` extraction.
 */
@Suite(.tags(.hermeticJourney))
final class SafetensorsHeaderTests {

    @Test
    func should_parse_a_bounded_header_with_one_tensor_and_metadata() throws {
        let (fileHandle, fileSizeBytes): (FileHandle, UInt64) = try Self.openFramedFile(
            headerJsonText: """
            {"__metadata__":{"format":"pt"},"tensor.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}
            """, payloadByteCount: 12);
        defer { fileHandle.closeFile(); }
        let parsedHeader: SafetensorsFraming.BoundedJsonHeader = try SafetensorsFraming.readBoundedJsonHeader(
            fileHandle: fileHandle, fileSizeBytes: fileSizeBytes,
            maximumHeaderLengthBytes: 1024 * 1024);
        #expect(parsedHeader.dataSectionStartBytes == UInt64(fileSizeBytes) - 12);
        #expect(parsedHeader.fileSizeBytes == fileSizeBytes);
        #expect(parsedHeader.tensorJsonValues.count == 1);
        #expect(parsedHeader.tensorJsonValues[0].tensorName == "tensor.weight");
        let tensorView: SafetensorsFraming.TensorView = try SafetensorsFraming.TensorView.decoded(
            wireValue: parsedHeader.tensorJsonValues[0].headerValue);
        #expect(tensorView.dtype == "BF16");
        #expect(tensorView.shape == [2, 3]);
        #expect(tensorView.dataStartOffset() == 0);
        #expect(tensorView.dataEndOffset() == 12);
        let metadataValue: JsonWireValue = try #require(parsedHeader.metadataJsonValue);
        guard case let .object(metadataObject) = metadataValue else {
            Issue.record("expected a metadata object");
            return;
        }
        #expect(metadataObject.value(forKey: "format") == JsonWireValue.string("pt"));
    }

    @Test
    func should_reject_duplicate_header_keys_before_any_consumer_can_classify_them() throws {
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
            Issue.record("duplicate safetensors header keys must be rejected");
        } catch let headerError as SafetensorsFraming.BoundedHeaderError {
            guard case .invalidHeaderJson(let problem) = headerError else {
                Issue.record("expected InvalidHeaderJson, got \(headerError)");
                return;
            }
            #expect(
                problem.lowercased().contains("duplicate"),
                "the duplicate-key rejection must name the duplication, got: \(problem)");
        }
    }

    @Test
    func should_reject_a_header_longer_than_the_declared_bound() throws {
        let (fileHandle, fileSizeBytes): (FileHandle, UInt64) = try Self.openFramedFile(
            headerJsonText: #"{"tensor.weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
            payloadByteCount: 1);
        defer { fileHandle.closeFile(); }
        do {
            _ = try SafetensorsFraming.readBoundedJsonHeader(
                fileHandle: fileHandle, fileSizeBytes: fileSizeBytes,
                maximumHeaderLengthBytes: 8);
            Issue.record("a header beyond the declared bound must fail closed");
        } catch let headerError as SafetensorsFraming.BoundedHeaderError {
            guard case .headerLengthTooLarge = headerError else {
                Issue.record("expected HeaderLengthTooLarge, got \(headerError)");
                return;
            }
        }
    }

    @Test
    func should_reject_a_header_that_reaches_beyond_the_file_end() throws {
        let (fileHandle, _): (FileHandle, UInt64) = try Self.openFramedFile(
            headerJsonText: #"{"tensor.weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
            payloadByteCount: 1);
        defer { fileHandle.closeFile(); }
        do {
            _ = try SafetensorsFraming.readBoundedJsonHeader(
                fileHandle: fileHandle, fileSizeBytes: 4,
                maximumHeaderLengthBytes: 1024 * 1024);
            Issue.record("a header reaching past the file end must fail closed");
        } catch let headerError as SafetensorsFraming.BoundedHeaderError {
            guard case .headerBeyondFile(let headerEndOffsetBytes, let declaredFileSizeBytes) = headerError else {
                Issue.record("expected HeaderBeyondFile, got \(headerError)");
                return;
            }
            #expect(headerEndOffsetBytes > declaredFileSizeBytes);
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
