import Foundation

import MLX
import Testing
import JourneyCategories
import ModelServingTestSupport
import RuntimeIntegration

/**
 * Hermetic SafeTensors reader journeys, continuing the Rust
 * `direct_mlx::safetensors_reader` tests at the Swift boundary: a retained
 * whole-file load through an open descriptor (with the path already
 * unlinked, matching the Rust descriptor-ownership contract) and a
 * bounded-range (expert-paged) load of two tensors from scattered source
 * ranges with read instrumentation attached.
 */
@Suite(.serialized, .tags(.hermeticMlxJourney))
final class SafetensorsReaderTests {

    init() {
        signal(SIGPIPE, SIG_IGN)
        MLXMetallibLocator.overrideMetallibPathIfNecessary()
    }

    @Test(.timeLimit(.minutes(1)))
    func should_load_a_retained_weights_file_and_report_missing_tensors() throws {
        let headerJson: String = "{\"model.embed_tokens.weight\":{\"dtype\":\"F32\",\"shape\":[1,4],\"data_offsets\":[0,16]}}"
        let headerJsonBytes: [UInt8] = Array(headerJson.utf8)
        var fixtureBytes: [UInt8] = SafetensorsFixtureSupport.littleEndianLengthPrefix(of: UInt64(headerJsonBytes.count))
        fixtureBytes.append(contentsOf: headerJsonBytes)
        fixtureBytes.append(contentsOf: SafetensorsFixtureSupport.littleEndianBytes(of: [1, 2, 3, 4]))

        let weightsFileUrl: URL = SafetensorsFixtureSupport.temporaryFileUrl("reader-retained.safetensors")
        FileManager.default.createFile(atPath: weightsFileUrl.path, contents: Data(fixtureBytes))
        defer {
            try? FileManager.default.removeItem(at: weightsFileUrl)
        }

        let weightsFileHandle: FileHandle = try FileHandle(forReadingFrom: weightsFileUrl)
        defer {
            weightsFileHandle.closeFile()
        }
        // Descriptor-ownership parity with Rust: once the reader holds the
        // descriptor, the path itself can already be gone.
        try? FileManager.default.removeItem(at: weightsFileUrl)

        let weightsFile: SafetensorsFile = try MlxRuntime.loadSafetensors(weightsFile: weightsFileHandle)

        do {
            let missingTensor: MLXArray = try weightsFile.tensor("missing.weight")
            MLX.asyncEval([missingTensor])
            Issue.record("looking up a tensor the header never named must fail")
        } catch MlxRuntimeError.tensorLookupFailed(let tensorName) {
            #expect(tensorName == "missing.weight")
        }

        var embedTensorHolder: MLXArray? = nil
        do {
            let embedTensor: MLXArray = try weightsFile.tensor("model.embed_tokens.weight")
            MLX.asyncEval([embedTensor])
            embedTensorHolder = embedTensor
        }
        guard let embedTensor: MLXArray = embedTensorHolder else {
            Issue.record("the embedded-tokens tensor must load under its header name")
            return
        }
        #expect(embedTensor.shape == [1, 4])
        #expect(embedTensor.dtype == .float32)
        #expect(embedTensor.size == 4)
        #expect(embedTensor.nbytes == 16)
        MLX.eval([embedTensor])
        #expect(embedTensor.asArray(Float.self) == [1, 2, 3, 4])
    }

    @Test(.timeLimit(.minutes(1)))
    func should_load_two_tensors_from_bounded_read_intervals_with_metrics() throws {
        let headerJson: String = "{\"first.weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},\"second.weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[8,16]}}"
        let headerJsonBytes: [UInt8] = Array(headerJson.utf8)
        var syntheticHeaderBytes: [UInt8] = SafetensorsFixtureSupport.littleEndianLengthPrefix(of: UInt64(headerJsonBytes.count))
        syntheticHeaderBytes.append(contentsOf: headerJsonBytes)

        var fixtureBytes: [UInt8] = Array(repeating: UInt8(0), count: 40)
        fixtureBytes.replaceSubrange(4..<12, with: SafetensorsFixtureSupport.littleEndianBytes(of: [1, 2]))
        fixtureBytes.replaceSubrange(28..<36, with: SafetensorsFixtureSupport.littleEndianBytes(of: [3, 4]))

        let weightsFileUrl: URL = SafetensorsFixtureSupport.temporaryFileUrl("reader-bounded.safetensors")
        FileManager.default.createFile(atPath: weightsFileUrl.path, contents: Data(fixtureBytes))
        defer {
            try? FileManager.default.removeItem(at: weightsFileUrl)
        }

        let sourceFileHandle: FileHandle = try FileHandle(forReadingFrom: weightsFileUrl)
        defer {
            sourceFileHandle.closeFile()
        }

        let readIntervals: [BoundedReadInterval] = [
            BoundedReadInterval(virtualPayloadOffset: 0, sourceFileOffset: 4, sourceByteCount: 8),
            BoundedReadInterval(virtualPayloadOffset: 8, sourceFileOffset: 28, sourceByteCount: 8),
        ]
        let positionalFileReadMetrics: PositionalFileReadMetrics = PositionalFileReadMetrics()
        let weightsFile: SafetensorsFile = try MlxRuntime.loadSafetensorsFromBoundedRanges(
            sourceFile: sourceFileHandle,
            syntheticHeaderBytes: Data(syntheticHeaderBytes),
            intervals: readIntervals,
            totalPayloadBytes: 16,
            expertFileReadMetrics: positionalFileReadMetrics)

        var firstTensorHolder: MLXArray? = nil
        var secondTensorHolder: MLXArray? = nil
        do {
            let firstTensor: MLXArray = try weightsFile.tensor("first.weight")
            MLX.asyncEval([firstTensor])
            firstTensorHolder = firstTensor
            let secondTensor: MLXArray = try weightsFile.tensor("second.weight")
            MLX.asyncEval([secondTensor])
            secondTensorHolder = secondTensor
        }
        guard let firstTensor: MLXArray = firstTensorHolder,
              let secondTensor: MLXArray = secondTensorHolder else {
            Issue.record("both bounded-range tensors must load under their header names")
            return
        }
        MLX.eval([firstTensor, secondTensor])
        #expect(firstTensor.asArray(Float.self) == [1, 2])
        #expect(secondTensor.asArray(Float.self) == [3, 4])

        let readSnapshot: PositionalFileReadSnapshot = positionalFileReadMetrics.snapshot()
        #expect(readSnapshot.readCallCount == 2)
        #expect(readSnapshot.readByteCount == 16)
        #expect(readSnapshot.totalReadElapsedNanoseconds > 0)
        #expect(readSnapshot.maximumReadElapsedNanoseconds > 0)
        #expect(readSnapshot.readFailureCount == 0)
    }
}
