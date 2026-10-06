import CryptoKit;
import Foundation;
import IpcProtocol;
import ModelServing;
import Testing;

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
@Suite
final class Qwen35ArtifactValidatorTests {

    private static let MODEL_DIRECTORY_LEAF_NAME: String = "example-tiny-dense-qwen35";
    private static let SHARD_FILE_NAME: String = "model-00001-of-00001.safetensors";
    private static let TOKENIZER_BYTES: Array<UInt8> = Array("{\"model\":{\"vocab\":{}}}".utf8);
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
        let modelLayout: SynthesizedModelLayout = try Self.synthesizedShardModelLayout(
            omitLmHeadScaleTensors: false, includeOptiQMetadata: true);
        let modelDirectoryUrl: URL = try Self.writeModelDirectory(modelLayout: modelLayout);
        defer { try? FileManager.default.removeItem(at: modelDirectoryUrl); }

        let validatedArtifact: ValidatedQwen35Artifact = try Qwen35ArtifactValidator()
            .validate(modelDirectory: modelDirectoryUrl.path, maxOutputTokens: Self.MAX_OUTPUT_TOKENS);

        #expect(validatedArtifact.shardCount() == 1);
        #expect(validatedArtifact.totalPayloadBytes() == modelLayout.totalPayloadBytes);
        #expect(validatedArtifact.tensorInventory().tensorCount() == modelLayout.tensorProfileCount);
        #expect(validatedArtifact.modelId() == modelDirectoryUrl.lastPathComponent);
        #expect(validatedArtifact.revision() == Self.configRevisionHex(configBytes: modelLayout.configBytes));
        #expect(validatedArtifact.maxOutputTokens() == Self.MAX_OUTPUT_TOKENS);
        #expect(validatedArtifact.tokenizerBytes() == Data(Self.TOKENIZER_BYTES));
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
        let modelLayout: SynthesizedModelLayout = try Self.synthesizedShardModelLayout(
            omitLmHeadScaleTensors: false, includeOptiQMetadata: true,
            measuredGroupSizeOverride: 32);
        let modelDirectoryUrl: URL = try Self.writeModelDirectory(modelLayout: modelLayout);
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
        let modelLayout: SynthesizedModelLayout = try Self.synthesizedShardModelLayout(
            omitLmHeadScaleTensors: true, includeOptiQMetadata: false);
        let modelDirectoryUrl: URL = try Self.writeModelDirectory(modelLayout: modelLayout);
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
        let modelLayout: SynthesizedModelLayout = try Self.synthesizedShardModelLayout(
            omitLmHeadScaleTensors: false, includeOptiQMetadata: false);
        let modelDirectoryUrl: URL = try Self.writeModelDirectory(modelLayout: modelLayout);
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
        let modelLayout: SynthesizedModelLayout = try Self.synthesizedShardModelLayout(
            omitLmHeadScaleTensors: false, includeOptiQMetadata: false);
        let modelDirectoryUrl: URL = try Self.writeModelDirectory(
            modelLayout: modelLayout, omitTokenizerFile: true);
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
        let modelLayout: SynthesizedModelLayout = try Self.synthesizedShardModelLayout(
            omitLmHeadScaleTensors: false, includeOptiQMetadata: false);
        let modelDirectoryUrl: URL = try Self.writeModelDirectory(modelLayout: modelLayout);
        defer { try? FileManager.default.removeItem(at: modelDirectoryUrl); }
        let validatedArtifact: ValidatedQwen35Artifact = try Qwen35ArtifactValidator()
            .validate(modelDirectory: modelDirectoryUrl.path, maxOutputTokens: Self.MAX_OUTPUT_TOKENS);

        let resolvedSourceId: TensorSourceId = try validatedArtifact
            .sourceIdForFileName(Self.SHARD_FILE_NAME);
        let transferredWeightsFile: ValidatedWeightsFile = try validatedArtifact
            .takeSafetensorsSource(resolvedSourceId);
        #expect(transferredWeightsFile.validatedRequiredFile.fileName == Self.SHARD_FILE_NAME);

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

    // MARK: - Synthesized model fixtures

    struct SynthesizedModelLayout {
        var configBytes: Array<UInt8>;
        var indexBytes: Array<UInt8>;
        var shardFileBytes: Data;
        var optiQMetadataBytes: Array<UInt8>?;
        var totalPayloadBytes: UInt64;
        var tensorProfileCount: Int;
    }

    /**
     * Builds the full layout of a tiny one-layer dense artifact: the config
     * derives every tensor profile, the shard mirrors those profiles as a
     * real safetensors file, and the optional measured metadata carries one
     * entry per quantized module override except the embeddings.
     */
    private static func synthesizedShardModelLayout(
        omitLmHeadScaleTensors: Bool, includeOptiQMetadata: Bool,
        measuredGroupSizeOverride: UInt32? = nil) throws -> SynthesizedModelLayout {
        let configBytes: Array<UInt8> = try denseSingleLayerConfigBytes();
        var config: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        let unfilteredProfiles: Array<TensorProfile> = Qwen3_5TensorSpec
            .qwen3_5LanguageTensorProfiles(qwen3_5Config: config);
        var shardTensorNames: Set<String> = Set(unfilteredProfiles.map { (tensorProfile: TensorProfile) -> String in
            return tensorProfile.name;
        });
        if omitLmHeadScaleTensors {
            shardTensorNames.remove("language_model.lm_head.scales");
            shardTensorNames.remove("language_model.lm_head.biases");
        }
        config.resolveUnquantizedModulesFromShardIndex(shardTensorNames: shardTensorNames);
        let tensorProfiles: Array<TensorProfile> = Qwen3_5TensorSpec
            .qwen3_5LanguageTensorProfiles(qwen3_5Config: config);

        let framedShardBytes: FramedShardBytes = safetensorsFileBytes(tensorProfiles: tensorProfiles);
        let indexBytes: Array<UInt8> = weightMapIndexBytes(
            tensorNames: shardTensorNames, totalPayloadBytes: framedShardBytes.payloadBytes);
        let optiQMetadataBytes: Array<UInt8>? = includeOptiQMetadata
            ? try measuredOptiQMetadataBytes(config: config, measuredGroupSizeOverride: measuredGroupSizeOverride)
            : nil;
        return SynthesizedModelLayout(
            configBytes: configBytes,
            indexBytes: indexBytes,
            shardFileBytes: framedShardBytes.fileBytes,
            optiQMetadataBytes: optiQMetadataBytes,
            totalPayloadBytes: framedShardBytes.payloadBytes,
            tensorProfileCount: tensorProfiles.count);
    }

    private static func writeModelDirectory(
        modelLayout: SynthesizedModelLayout, omitTokenizerFile: Bool = false) throws -> URL {
        let modelDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(MODEL_DIRECTORY_LEAF_NAME)-\(UUID().uuidString)");
        try FileManager.default.createDirectory(at: modelDirectoryUrl, withIntermediateDirectories: true);
        try Data(modelLayout.configBytes).write(
            to: modelDirectoryUrl.appendingPathComponent("config.json"));
        if omitTokenizerFile == false {
            try Data(TOKENIZER_BYTES).write(to: modelDirectoryUrl.appendingPathComponent("tokenizer.json"));
        }
        try Data(modelLayout.indexBytes).write(
            to: modelDirectoryUrl.appendingPathComponent("model.safetensors.index.json"));
        try modelLayout.shardFileBytes.write(
            to: modelDirectoryUrl.appendingPathComponent(SHARD_FILE_NAME));
        if let optiQMetadataBytes: Array<UInt8> = modelLayout.optiQMetadataBytes {
            try Data(optiQMetadataBytes).write(
                to: modelDirectoryUrl.appendingPathComponent("optiq_metadata.json"));
        }
        return modelDirectoryUrl;
    }

    /**
     * The tiny dense config: the standard minimal fixture shrunk to one
     * full-attention decoder layer whose every dimension stays divisible by
     * the affine group size, so generated profiles carry non-empty shapes.
     */
    private static func denseSingleLayerConfigBytes() throws -> Array<UInt8> {
        var configValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.minimalValidConfigJson();
        let architecturesValue: JsonWireValue = .stringArray(["Qwen3_5ForConditionalGeneration"]);
        configValue = configValue.settingObjectKey(path: ["architectures"], newValue: architecturesValue);
        configValue = configValue.settingObjectKey(path: ["model_type"], newValue: .string("qwen3_5"));
        let eosTokenIdsValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue("[3, 4]");
        configValue = configValue.settingObjectKey(path: ["eos_token_id"], newValue: eosTokenIdsValue);
        configValue = configValue.settingObjectKey(path: ["pad_token_id"], newValue: .unsignedInteger(4));
        let tinyTextConfig: [(fieldName: String, fieldValue: JsonWireValue)] = [
            ("model_type", .string("qwen3_5_text")),
            ("hidden_size", .unsignedInteger(64)),
            ("num_hidden_layers", .unsignedInteger(1)),
            ("num_attention_heads", .unsignedInteger(1)),
            ("num_key_value_heads", .unsignedInteger(1)),
            ("head_dim", .unsignedInteger(64)),
            ("vocab_size", .unsignedInteger(256)),
            ("intermediate_size", .unsignedInteger(64)),
            ("num_experts", .unsignedInteger(0)),
            ("num_experts_per_tok", .unsignedInteger(0)),
            ("moe_intermediate_size", .unsignedInteger(0)),
            ("shared_expert_intermediate_size", .unsignedInteger(0)),
            ("linear_num_key_heads", .unsignedInteger(1)),
            ("linear_num_value_heads", .unsignedInteger(1)),
            ("linear_key_head_dim", .unsignedInteger(64)),
            ("linear_value_head_dim", .unsignedInteger(64)),
        ];
        for textConfigField: (fieldName: String, fieldValue: JsonWireValue) in tinyTextConfig {
            configValue = configValue.settingObjectKey(
                path: ["text_config", textConfigField.fieldName], newValue: textConfigField.fieldValue);
        }
        configValue = configValue.settingObjectKey(
            path: ["text_config", "layer_types"], newValue: .stringArray(["full_attention"]));
        return try Qwen3_5MoeConfigFixtures.serializedBytes(configValue);
    }

    private static func measuredOptiQMetadataBytes(
        config: Qwen3_5Config, measuredGroupSizeOverride: UInt32?) throws -> Array<UInt8> {
        var measuredModuleEntries: Array<String> = Array();
        for moduleEntry: (moduleName: String, profile: OptiQQuantizationProfile)
            in config.quantizedModuleProfiles().entries {
            // Only quantized modules publish measurements: unquantized
            // modules (norms) carry no bit map entries, and the embedding
            // pair stays unmeasured by the frozen artifact contract.
            if moduleEntry.profile.isUnquantized()
                || moduleEntry.moduleName == "language_model.model.embed_tokens"
                || moduleEntry.moduleName == "language_model.lm_head" {
                continue;
            }
            let measuredGroupSize: UInt32 = measuredGroupSizeOverride ?? moduleEntry.profile.groupSize;
            measuredModuleEntries.append(
                "\"\(moduleEntry.moduleName)\": {\"bits\": \(moduleEntry.profile.bits), \"group_size\": \(measuredGroupSize)}");
        }
        let perLayerJsonText: String = measuredModuleEntries.joined(separator: ", ");
        let metadataJsonText: String = """
    {"method": "static_mixed_precision", "base_model": "example/example-parent", "reference": "structural_rules", "target_bpw": 4.0, "achieved_bpw": 4.0, "n_high_bits": 0, "n_low_bits": \(measuredModuleEntries.count), "threshold": 0.0, "per_layer": {\(perLayerJsonText)}}
    """;
        return Array(metadataJsonText.utf8);
    }

    private static func safetensorsFileBytes(
        tensorProfiles: Array<TensorProfile>) -> FramedShardBytes {
        var headerEntries: Array<String> = Array();
        var payloadBytes: Data = Data();
        var payloadOffsetBytes: UInt64 = 0;
        for tensorProfile: TensorProfile in tensorProfiles {
            let dtypeCanonicalName: String = self.safetensorsDtypeName(tensorDtype: tensorProfile.dtype);
            let elementCount: UInt64 = tensorProfile.shape.reduce(UInt64(1), { (productBytes: UInt64, dimensionValue: Int) -> UInt64 in
                return productBytes * UInt64(dimensionValue);
            });
            let tensorPayloadBytes: UInt64 = elementCount * self.dtypeBitsPerElement(tensorDtype: tensorProfile.dtype) / 8;
            let tensorEndOffsetBytes: UInt64 = payloadOffsetBytes + tensorPayloadBytes;
            headerEntries.append(
                "\"\(tensorProfile.name)\": {\"dtype\": \"\(dtypeCanonicalName)\", \"shape\": [\(tensorProfile.shape.map(String.init).joined(separator: ", "))], \"data_offsets\": [\(payloadOffsetBytes), \(tensorEndOffsetBytes)]}");
            payloadBytes.append(Data(count: Int(tensorPayloadBytes)));
            payloadOffsetBytes = tensorEndOffsetBytes;
        }
        let headerText: String = "{" + headerEntries.joined(separator: ", ") + "}";
        var framedBytes: Array<UInt8> = Array();
        var littleEndianHeaderLength: UInt64 = UInt64(headerText.utf8.count);
        withUnsafeBytes(of: &littleEndianHeaderLength) { (valueBuffer: UnsafeRawBufferPointer) -> Void in
            framedBytes.append(contentsOf: Array(valueBuffer));
        };
        framedBytes.append(contentsOf: Array(headerText.utf8));
        framedBytes.append(contentsOf: Array(payloadBytes));
        return FramedShardBytes(fileBytes: Data(framedBytes), payloadBytes: payloadOffsetBytes);
    }

    private static func weightMapIndexBytes(
        tensorNames: Set<String>, totalPayloadBytes: UInt64) -> Array<UInt8> {
        let weightMapEntries: Array<String> = tensorNames.sorted(by: { (leftName: String, rightName: String) -> Bool in
            return Array(leftName.utf8).lexicographicallyPrecedes(Array(rightName.utf8));
        }).map { (tensorName: String) -> String in
            return "\"\(tensorName)\": \"\(SHARD_FILE_NAME)\"";
        };
        let indexJsonText: String = """
    {"metadata": {"total_size": \(totalPayloadBytes)}, "weight_map": {\(weightMapEntries.joined(separator: ", "))}}
    """;
        return Array(indexJsonText.utf8);
    }

    private static func safetensorsDtypeName(tensorDtype: TensorDtype) -> String {
        switch tensorDtype {
        case .affineQuantizationFloat, .float32: return "F32";
        case .modelFloat, .bfloat16: return "BF16";
        case .uint32: return "U32";
        }
    }

    private static func dtypeBitsPerElement(tensorDtype: TensorDtype) -> UInt64 {
        switch tensorDtype {
        case .affineQuantizationFloat, .float32, .uint32: return 32;
        case .modelFloat, .bfloat16: return 16;
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
