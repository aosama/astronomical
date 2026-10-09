import Foundation

import MLX
import Testing
import JourneyCategories
import ModelServingTestSupport
import RuntimeIntegration

/**
 * Hermetic SafeTensors writer journeys, continuing the Rust
 * `direct_mlx::safetensors_writer` tests at the Swift boundary: file
 * round-trips through the stock MLX saver, the descriptor-I/O failure
 * class for unwritable destinations, in-memory serialization round-trips,
 * and the bounded serializer's byte-ceiling refusal.
 */
extension RuntimeIntegrationMlxJourneyContainer {

    @Suite(.serialized, .tags(.hermeticMlxJourney))
    final class SafetensorsWriterTests {

        init() {
            signal(SIGPIPE, SIG_IGN)
            MLXMetallibLocator.overrideMetallibPathIfNecessary()
        }

        @Test(.timeLimit(.minutes(1)))
        func should_round_trip_named_arrays_through_the_file_writer() throws {
            let destinationUrl: URL = SafetensorsFixtureSupport.temporaryFileUrl("writer-roundtrip.safetensors")
            FileManager.default.createFile(atPath: destinationUrl.path, contents: Data())
            defer {
                try? FileManager.default.removeItem(at: destinationUrl)
            }

            let recurrentValues: [Float] = [1, 2, 3, 4]
            let arraysByName: [String: MLXArray] = [
                "layer_0_recurrent": MLXArray(recurrentValues, [1, 4]),
                "layer_1_keys": MLX.zeros([2, 3], dtype: .bfloat16),
            ]
            let metadata: [String: String] = ["format_version": "1", "token_count": "2048"]
            let writeOutcome: SafetensorsWriteOutcome = try MlxRuntime.saveSafetensors(
                arrays: arraysByName,
                metadata: metadata,
                to: destinationUrl)
            let onDiskByteCount: Int = try #require(
                try FileManager.default.attributesOfItem(atPath: destinationUrl.path)[.size] as? Int)
            #expect(writeOutcome.writtenByteCount == onDiskByteCount)
            #expect(writeOutcome.writtenByteCount > 0)

            let weightsFileHandle: FileHandle = try FileHandle(forReadingFrom: destinationUrl)
            defer {
                weightsFileHandle.closeFile()
            }
            let weightsFile: SafetensorsFile = try MlxRuntime.loadSafetensors(weightsFile: weightsFileHandle)

            var recurrentTensorHolder: MLXArray? = nil
            do {
                let recurrentTensor: MLXArray = try weightsFile.tensor("layer_0_recurrent")
                MLX.asyncEval([recurrentTensor])
                recurrentTensorHolder = recurrentTensor
            }
            guard let recurrentTensor: MLXArray = recurrentTensorHolder else {
                Issue.record("the recurrent tensor must survive the write-read round trip")
                return
            }
            #expect(recurrentTensor.shape == [1, 4])
            MLX.eval([recurrentTensor])
            #expect(recurrentTensor.asArray(Float.self) == [1, 2, 3, 4])

            var keysTensorHolder: MLXArray? = nil
            do {
                let keysTensor: MLXArray = try weightsFile.tensor("layer_1_keys")
                MLX.asyncEval([keysTensor])
                keysTensorHolder = keysTensor
            }
            guard let keysTensor: MLXArray = keysTensorHolder else {
                Issue.record("the keys tensor must survive the write-read round trip")
                return
            }
            #expect(keysTensor.shape == [2, 3])
            #expect(keysTensor.dtype == .bfloat16)
        }

        @Test(.timeLimit(.minutes(1)))
        func should_report_descriptor_io_failure_for_a_read_only_destination() throws {
            let destinationUrl: URL = SafetensorsFixtureSupport.temporaryFileUrl("writer-readonly.safetensors")
            FileManager.default.createFile(atPath: destinationUrl.path, contents: Data())
            try FileManager.default.setAttributes([.posixPermissions: 0o444], ofItemAtPath: destinationUrl.path)
            defer {
                try? FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: destinationUrl.path)
                try? FileManager.default.removeItem(at: destinationUrl)
            }

            do {
                _ = try MlxRuntime.saveSafetensors(
                    arrays: ["probe": MLXArray([Float(1)], [1])],
                    metadata: [:],
                    to: destinationUrl)
                Issue.record("saving to a read-only destination must fail")
            } catch SafetensorsWriterError.descriptorIo(let description) {
                #expect(!description.isEmpty)
            } catch let writerError as SafetensorsWriterError {
                Issue.record("unexpected writer error: \(writerError)")
            } catch {
                Issue.record("unexpected error type: \(error)")
            }
        }

        @Test(.timeLimit(.minutes(1)))
        func should_serialize_to_memory_and_reload_through_native_loadarrays() throws {
            let tensorValues: [Float] = [2, 4, 6, 8]
            let serializedBytes: Data = try MlxRuntime.serializeSafetensors(
                arrays: ["t": MLXArray(tensorValues, [2, 2])],
                metadata: ["format_version": "test"])
            let reloadedArraysByName: [String: MLXArray] = try MLX.loadArrays(data: serializedBytes)

            guard let reloadedTensor: MLXArray = reloadedArraysByName["t"] else {
                Issue.record("the in-memory serialized tensor must reload under its name")
                return
            }
            MLX.asyncEval([reloadedTensor])
            MLX.eval([reloadedTensor])
            #expect(reloadedTensor.shape == [2, 2])
            #expect(reloadedTensor.asArray(Float.self) == [2, 4, 6, 8])
        }

        @Test(.timeLimit(.minutes(1)))
        func should_refuse_serialization_above_the_caller_byte_ceiling() throws {
            let tensorValues: [Float] = [1, 3, 5, 7]
            let arraysByName: [String: MLXArray] = ["t": MLXArray(tensorValues, [2, 2])]
            let metadata: [String: String] = [
                "format_version": "11",
                "storage_contract_fingerprint": "0123456789abcdef0123456789abcdef",
            ]
            let serializedBytes: Data = try MlxRuntime.serializeSafetensors(arrays: arraysByName, metadata: metadata)
            let serializedBytesAgain: Data = try MlxRuntime.serializeSafetensors(arrays: arraysByName, metadata: metadata)
            #expect(serializedBytes == serializedBytesAgain)
            let serializedByteCount: Int = serializedBytes.count
            let maximumByteCount: Int = serializedByteCount - 1

            do {
                _ = try MlxRuntime.serializeSafetensors(
                    arrays: arraysByName,
                    metadata: metadata,
                    maximumByteCount: maximumByteCount)
                Issue.record("serialization must refuse byte counts above the caller's ceiling")
            } catch MlxRuntimeError.safetensorsSerializationLimitExceeded(
                let attemptedByteCount,
                let permittedByteCount) {
                #expect(attemptedByteCount == serializedByteCount)
                #expect(permittedByteCount == maximumByteCount)
            } catch {
                Issue.record("unexpected error type: \(error)")
            }
        }
    }
}
