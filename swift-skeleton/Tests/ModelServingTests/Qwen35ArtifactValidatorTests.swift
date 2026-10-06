import CryptoKit;
import Foundation;
import IpcProtocol;
import ModelServing;
import ModelServingTestSupport;
import Testing;
import JourneyCategories;

/**
 * Hermetic journeys for the Qwen3.5 artifact validator: required-file
 * discovery, config and OptiQ metadata binding, shard-index inventory
 * resolution, and descriptor-backed shard source transfer. CPU only — no
 * model load.
 *
 * Every journey synthesizes a complete model directory from one tiny dense
 * config; tensor inventory comes from the config itself, so shard headers
 * carry exactly the tensors the artifact contract demands and assertions
 * stay structural (counts, sums, hashes) instead of golden-master bytes.
 */
@Suite(.tags(.hermeticJourney))
final class Qwen35ArtifactValidatorTests {

    private static let MAX_OUTPUT_TOKENS: UInt32 = 256;

    /**
     * A complete shard-layout artifact with a matching measured OptiQ
     * metadata document must validate into a descriptor-backed artifact
     * whose structural facts all derive from the config: shard count,
     * payload sum, tensor inventory size, model identity, and the
     * config-hash revision.
     */
    @Test
    func should_validate_a_complete_shard_layout_artifact_into_a_descriptor_backed_artifact() throws {
        let (modelDirectoryUrl, modelLayout): (URL, TinyDenseArtifactFixture.SynthesizedLayout) =
            try TinyDenseArtifactFixture.writeModelDirectory(includeOptiQMetadata: true);
        defer { try? FileManager.default.removeItem(at: modelDirectoryUrl); }

        let validatedArtifact: ValidatedQwen35Artifact = try Qwen35ArtifactValidator()
            .validate(modelDirectory: modelDirectoryUrl.path, maxOutputTokens: Self.MAX_OUTPUT_TOKENS);

        #expect(validatedArtifact.shardCount() == 1);
        #expect(validatedArtifact.totalPayloadBytes() == modelLayout.totalPayloadBytes);
        #expect(validatedArtifact.tensorInventory().tensorCount() == modelLayout.tensorProfileCount);
        #expect(validatedArtifact.modelId() == modelDirectoryUrl.lastPathComponent);
        #expect(validatedArtifact.revision() == Self.configRevisionHex(configBytes: modelLayout.configBytes));
        #expect(validatedArtifact.maxOutputTokens() == Self.MAX_OUTPUT_TOKENS);
        #expect(validatedArtifact.tokenizerBytes() == Data(TinyDenseArtifactFixture.PLACEHOLDER_TOKENIZER_BYTES));
        #expect(validatedArtifact.supportsImageInput() == false);
        #expect(validatedArtifact.hasSeparateVisionSidecar() == false);
        #expect(validatedArtifact.config().layerCount() == 1);
    }

    /**
     * The OptiQ artifact mapping binds the measured bit map to the
     * quantization config: a measured profile that differs from the config
     * override fails validation with the group-size mismatch and a public
     * failure reason naming both sides.
     */
    @Test
    func should_bind_measured_optiq_metadata_profiles_to_the_quantization_config_during_validation() throws {
        let (modelDirectoryUrl, _): (URL, TinyDenseArtifactFixture.SynthesizedLayout) =
            try TinyDenseArtifactFixture.writeModelDirectory(
                includeOptiQMetadata: true, measuredGroupSizeOverride: 32);
        defer { try? FileManager.default.removeItem(at: modelDirectoryUrl); }

        do {
            _ = try Qwen35ArtifactValidator().validate(
                modelDirectory: modelDirectoryUrl.path, maxOutputTokens: Self.MAX_OUTPUT_TOKENS);
            Issue.record("metadata group size 32 against config group size 64 must fail validation");
        } catch let validationError as Qwen35ArtifactValidationError {
            guard case .optiQMetadata(.configGroupSizeMismatch(let moduleName, let configGroupSize, let metadataGroupSize)) = validationError else {
                Issue.record("expected an OptiQ metadata group-size mismatch, got \(validationError)");
                return;
            }
            #expect(moduleName == "language_model.model.layers.0.mlp.down_proj");
            #expect(configGroupSize == 64);
            #expect(metadataGroupSize == 32);
            let publicFailureReason: String = validationError.publicFailureReason();
            #expect(publicFailureReason.contains("OptiQ metadata validation failed"));
            #expect(publicFailureReason.contains("group size 64 in config and 32 in metadata"));
        }
    }

    /**
     * Modules whose scales and biases are absent from the shard inventory
     * resolve to unquantized bfloat16 storage: the artifact still validates
     * and the resolved config carries the zero-bit profile.
     */
    @Test
    func should_resolve_unquantized_modules_from_the_shard_inventory_while_validating() throws {
        let (modelDirectoryUrl, modelLayout): (URL, TinyDenseArtifactFixture.SynthesizedLayout) =
            try TinyDenseArtifactFixture.writeModelDirectory(omitLmHeadScaleTensors: true);
        defer { try? FileManager.default.removeItem(at: modelDirectoryUrl); }

        let validatedArtifact: ValidatedQwen35Artifact = try Qwen35ArtifactValidator()
            .validate(modelDirectory: modelDirectoryUrl.path, maxOutputTokens: Self.MAX_OUTPUT_TOKENS);

        let lmHeadProfile: OptiQQuantizationProfile = validatedArtifact.config()
            .quantizationProfile(forModule: "language_model.lm_head");
        #expect(lmHeadProfile.isUnquantized());
        #expect(validatedArtifact.totalPayloadBytes() == modelLayout.totalPayloadBytes);
    }

    /**
     * An artifact without optiq_metadata.json carries no measured bit map,
     * so validation succeeds on the config and shard evidence alone.
     */
    @Test
    func should_accept_an_artifact_without_optiq_metadata() throws {
        let (modelDirectoryUrl, _): (URL, TinyDenseArtifactFixture.SynthesizedLayout) =
            try TinyDenseArtifactFixture.writeModelDirectory();
        defer { try? FileManager.default.removeItem(at: modelDirectoryUrl); }

        let validatedArtifact: ValidatedQwen35Artifact = try Qwen35ArtifactValidator()
            .validate(modelDirectory: modelDirectoryUrl.path, maxOutputTokens: Self.MAX_OUTPUT_TOKENS);

        #expect(validatedArtifact.shardCount() == 1);
    }

    /**
     * A model directory that does not exist fails closed with the typed
     * directory-not-found artifact error.
     */
    @Test
    func should_reject_a_model_directory_that_does_not_exist() throws {
        let missingDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("absent-qwen35-\(UUID().uuidString)");

        do {
            _ = try Qwen35ArtifactValidator().validate(
                modelDirectory: missingDirectoryUrl.path, maxOutputTokens: Self.MAX_OUTPUT_TOKENS);
            Issue.record("a missing model directory must fail validation");
        } catch let validationError as Qwen35ArtifactValidationError {
            guard case .artifact(.modelDirectoryNotFound(let modelDirectory)) = validationError else {
                Issue.record("expected modelDirectoryNotFound, got \(validationError)");
                return;
            }
            #expect(modelDirectory == missingDirectoryUrl.path);
        }
    }

    /**
     * An artifact missing a required file fails validation by naming the
     * absent file instead of the bytes behind it.
     */
    @Test
    func should_reject_an_artifact_missing_a_required_file() throws {
        let (modelDirectoryUrl, _): (URL, TinyDenseArtifactFixture.SynthesizedLayout) =
            try TinyDenseArtifactFixture.writeModelDirectory(omitTokenizerFile: true);
        defer { try? FileManager.default.removeItem(at: modelDirectoryUrl); }

        do {
            _ = try Qwen35ArtifactValidator().validate(
                modelDirectory: modelDirectoryUrl.path, maxOutputTokens: Self.MAX_OUTPUT_TOKENS);
            Issue.record("an artifact without tokenizer.json must fail validation");
        } catch let validationError as Qwen35ArtifactValidationError {
            guard case .artifact(.inspectRequiredFile(let missingFileName, _)) = validationError else {
                Issue.record("expected inspectRequiredFile, got \(validationError)");
                return;
            }
            #expect(missingFileName == "tokenizer.json");
        }
    }

    /**
     * Shard sources transfer exactly once by resolved validated identity:
     * a resolved file name answers its source id, a taken source cannot be
     * taken again, and an unknown file name resolves to a typed error.
     */
    @Test
    func should_transfer_validated_shard_sources_exactly_once_by_resolved_identity() throws {
        let (modelDirectoryUrl, _): (URL, TinyDenseArtifactFixture.SynthesizedLayout) =
            try TinyDenseArtifactFixture.writeModelDirectory();
        defer { try? FileManager.default.removeItem(at: modelDirectoryUrl); }
        let validatedArtifact: ValidatedQwen35Artifact = try Qwen35ArtifactValidator()
            .validate(modelDirectory: modelDirectoryUrl.path, maxOutputTokens: Self.MAX_OUTPUT_TOKENS);

        let resolvedSourceId: TensorSourceId = try validatedArtifact
            .sourceIdForFileName(TinyDenseArtifactFixture.SHARD_FILE_NAME);
        let transferredWeightsFile: ValidatedWeightsFile = try validatedArtifact
            .takeSafetensorsSource(resolvedSourceId);
        #expect(transferredWeightsFile.validatedRequiredFile.fileName == TinyDenseArtifactFixture.SHARD_FILE_NAME);

        do {
            _ = try validatedArtifact.takeSafetensorsSource(resolvedSourceId);
            Issue.record("a second transfer of one source id must fail");
        } catch let validationError as Qwen35ArtifactValidationError {
            guard case .artifact(.profileMissingRequiredFile) = validationError else {
                Issue.record("expected profileMissingRequiredFile on the second transfer, got \(validationError)");
                return;
            }
        }
        do {
            _ = try validatedArtifact.sourceIdForFileName("absent-shard.safetensors");
            Issue.record("an unknown shard file name must fail to resolve");
        } catch let validationError as Qwen35ArtifactValidationError {
            guard case .artifact(.profileMissingRequiredFile(let absentFileName)) = validationError else {
                Issue.record("expected profileMissingRequiredFile for the unknown file, got \(validationError)");
                return;
            }
            #expect(absentFileName == "absent-shard.safetensors");
        }
    }

        /**
     * The validator derives the artifact revision from the SHA-256 of the
     * config bytes; the journey derives the same value independently.
     */
    private static func configRevisionHex(configBytes: Array<UInt8>) -> String {
        let configDigest: SHA256Digest = SHA256.hash(data: Data(configBytes));
        let digestBytes: Array<UInt8> = Array(configDigest);
        var leadingQuadWord: UInt64 = 0;
        for digestByte: UInt8 in digestBytes[0..<8] {
            leadingQuadWord = (leadingQuadWord << 8) | UInt64(digestByte);
        }
        return String(format: "%012llx", leadingQuadWord);
    }

    private struct FramedShardBytes {
        var fileBytes: Data;
        var payloadBytes: UInt64;
    }
}
