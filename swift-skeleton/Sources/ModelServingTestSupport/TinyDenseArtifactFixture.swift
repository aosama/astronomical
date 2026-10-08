import Foundation;

import ModelServing;

/// Synthesizes a complete tiny dense Qwen3.5 artifact on disk: one
/// full-attention decoder layer whose every dimension stays divisible by
/// the affine group size, a real safetensors shard mirroring the
/// config-derived tensor profiles, the shard index, and — on request — a
/// working tiny tokenizer pair.
///
/// Payload bytes are deterministic and position-varying: float elements
/// carry a pinned safe exponent byte with a varying mantissa and integer
/// elements vary freely, so every directory is a real distinct model whose
/// forward pass is numerically safe, and two directories with different
/// variant salts produce measurably different weight values.
///
/// Shared by the artifact-validation journeys and the artifact-weight
/// streaming journeys; all assertions stay structural (counts, sums,
/// config-derived facts), never golden-master bytes.
public enum TinyDenseArtifactFixture {

    public static let MODEL_DIRECTORY_LEAF_NAME: String = "example-tiny-dense-qwen35";
    public static let SHARD_FILE_NAME: String = "model-00001-of-00001.safetensors";
    public static let PLACEHOLDER_TOKENIZER_BYTES: Array<UInt8> = Array(
        "{\"model\":{\"vocab\":{}}}".utf8);
    public static let DEFAULT_WEIGHT_VARIANT_SALT: UInt8 = 0;
    public static let ALTERNATE_WEIGHT_VARIANT_SALT: UInt8 = 1;
    public static let MAX_OUTPUT_TOKENS: UInt32 = 256;

    public struct SynthesizedLayout {
        public var configBytes: Array<UInt8>;
        public var totalPayloadBytes: UInt64;
        public var tensorProfileCount: Int;
    }

    /// Writes the full artifact directory and returns the layout facts the
    /// journeys assert against.
    ///
    /// - Parameters:
    ///   - weightVariantSalt: rotates every payload byte pattern; changing
    ///     it changes the model's weight values.
    ///   - omitLmHeadScaleTensors: drops the lm_head scales and biases so
    ///     the validator resolves that module as unquantized.
    ///   - includeOptiQMetadata: writes a measured `optiq_metadata.json`
    ///     bound to the config's quantization overrides.
    ///   - measuredGroupSizeOverride: substitutes every measured group size
    ///     in the metadata document.
    ///   - includeTokenizerFiles: writes the tiny working tokenizer pair in
    ///     place of the placeholder `tokenizer.json` bytes.
    ///   - omitTokenizerFile: skips `tokenizer.json` entirely.
    public static func writeModelDirectory(
        weightVariantSalt: UInt8 = TinyDenseArtifactFixture.DEFAULT_WEIGHT_VARIANT_SALT,
        omitLmHeadScaleTensors: Bool = false,
        includeOptiQMetadata: Bool = false,
        measuredGroupSizeOverride: UInt32? = nil,
        includeTokenizerFiles: Bool = false,
        omitTokenizerFile: Bool = false
    ) throws -> (modelDirectoryUrl: URL, layout: SynthesizedLayout) {
        let configBytes: Array<UInt8> = try TinyDenseArtifactFixture.denseSingleLayerConfigBytes();
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

        let framedShardBytes: SynthesizedShardBytes = TinyDenseArtifactFixture.safetensorsFileBytes(
            tensorProfiles: tensorProfiles, weightVariantSalt: weightVariantSalt);
        let indexBytes: Array<UInt8> = TinyDenseArtifactFixture.weightMapIndexBytes(
            tensorNames: shardTensorNames, totalPayloadBytes: framedShardBytes.payloadBytes);
        let optiQMetadataBytes: Array<UInt8>? = includeOptiQMetadata
            ? try TinyDenseArtifactFixture.measuredOptiQMetadataBytes(
                config: config, measuredGroupSizeOverride: measuredGroupSizeOverride)
            : nil;

        let modelDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "\(TinyDenseArtifactFixture.MODEL_DIRECTORY_LEAF_NAME)-\(UUID().uuidString)");
        try FileManager.default.createDirectory(at: modelDirectoryUrl, withIntermediateDirectories: true);
        try Data(configBytes).write(to: modelDirectoryUrl.appendingPathComponent("config.json"));
        if omitTokenizerFile == false {
            if includeTokenizerFiles {
                try TinyTokenizerFixture.writeFiles(modelDirectoryUrl: modelDirectoryUrl);
            } else {
                try Data(TinyDenseArtifactFixture.PLACEHOLDER_TOKENIZER_BYTES).write(
                    to: modelDirectoryUrl.appendingPathComponent("tokenizer.json"));
            }
        }
        try Data(indexBytes).write(
            to: modelDirectoryUrl.appendingPathComponent("model.safetensors.index.json"));
        try framedShardBytes.fileBytes.write(
            to: modelDirectoryUrl.appendingPathComponent(TinyDenseArtifactFixture.SHARD_FILE_NAME));
        if let optiQMetadataBytes: Array<UInt8> = optiQMetadataBytes {
            try Data(optiQMetadataBytes).write(
                to: modelDirectoryUrl.appendingPathComponent("optiq_metadata.json"));
        }
        return (
            modelDirectoryUrl: modelDirectoryUrl,
            layout: SynthesizedLayout(
                configBytes: configBytes,
                totalPayloadBytes: framedShardBytes.payloadBytes,
                tensorProfileCount: tensorProfiles.count)
        );
    }

    /**
     * The tiny dense config: one full-attention layer whose dimensions stay
     * divisible by the affine group size, the M-RoPE parameters the dense
     * text model expects (partial rotary 1.0 so the section sum matches the
     * rotary dimension), and the mixed 4/8-bit OptiQ quantization contract.
     */
    private static func denseSingleLayerConfigBytes() throws -> Array<UInt8> {
        let configJsonText: String = """
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "dtype": "bfloat16",
                "eos_token_id": [3, 4],
                "pad_token_id": 4,
                "tie_word_embeddings": false,
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "hidden_act": "silu",
                    "hidden_size": 64,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "head_dim": 64,
                    "rms_norm_eps": 0.000001,
                    "attention_bias": false,
                    "mlp_bias": false,
                    "norm_topk_prob": true,
                    "output_router_logits": false,
                    "vocab_size": 256,
                    "intermediate_size": 64,
                    "max_position_embeddings": 4096,
                    "full_attention_interval": 1,
                    "linear_conv_kernel_dim": 4,
                    "linear_num_key_heads": 1,
                    "linear_num_value_heads": 1,
                    "linear_key_head_dim": 64,
                    "linear_value_head_dim": 64,
                    "num_experts": 0,
                    "num_experts_per_tok": 0,
                    "moe_intermediate_size": 0,
                    "shared_expert_intermediate_size": 0,
                    "rope_parameters": {
                        "type": "default",
                        "mrope_interleaved": true,
                        "mrope_section": [11, 11, 10],
                        "rope_theta": 100000.0,
                        "partial_rotary_factor": 1.0
                    },
                    "layer_types": ["full_attention"]
                },
                "quantization": {"group_size": 64, "bits": 4, "mode": "affine", "language_model.model.embed_tokens": {"group_size": 64, "bits": 8}, "language_model.lm_head": {"group_size": 64, "bits": 8}},
                "quantization_config": {"group_size": 64, "bits": 4, "mode": "affine", "language_model.model.embed_tokens": {"group_size": 64, "bits": 8}, "language_model.lm_head": {"group_size": 64, "bits": 8}}
            }
            """;
        return Array(configJsonText.utf8);
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

    /// Frames one real safetensors file whose payload bytes vary by tensor
    /// and position, at the default weight variant.
    public static func synthesizedShardBytes(
        tensorProfiles: Array<TensorProfile>
    ) -> SynthesizedShardBytes {
        return TinyDenseArtifactFixture.safetensorsFileBytes(
            tensorProfiles: tensorProfiles,
            weightVariantSalt: TinyDenseArtifactFixture.DEFAULT_WEIGHT_VARIANT_SALT);
    }

    /// Frames the shard index whose weight map names every tensor's shard.
    public static func synthesizedIndexBytes(
        tensorNames: Set<String>, totalPayloadBytes: UInt64
    ) -> Array<UInt8> {
        return TinyDenseArtifactFixture.weightMapIndexBytes(
            tensorNames: tensorNames, totalPayloadBytes: totalPayloadBytes);
    }

    /// Frames one real safetensors file whose payload bytes vary by tensor
    /// and position. Float elements pin their exponent byte (0x3C or 0x3D
    /// per variant) and vary the mantissa bytes, so values stay small,
    /// finite, and distinct; packed-integer elements vary every byte.
    private static func safetensorsFileBytes(
        tensorProfiles: Array<TensorProfile>, weightVariantSalt: UInt8) -> SynthesizedShardBytes {
        var headerEntries: Array<String> = Array();
        var payloadBytes: Data = Data();
        var payloadOffsetBytes: UInt64 = 0;
        for (tensorOrdinal, tensorProfile): (Int, TensorProfile) in tensorProfiles.enumerated() {
            let dtypeCanonicalName: String = TinyDenseArtifactFixture.safetensorsDtypeName(
                tensorDtype: tensorProfile.dtype);
            let elementCount: UInt64 = tensorProfile.shape.reduce(UInt64(1), { (productBytes: UInt64, dimensionValue: Int) -> UInt64 in
                return productBytes * UInt64(dimensionValue);
            });
            let bitsPerElement: UInt64 = TinyDenseArtifactFixture.dtypeBitsPerElement(
                tensorDtype: tensorProfile.dtype);
            let tensorPayloadBytes: UInt64 = elementCount * bitsPerElement / 8;
            let tensorEndOffsetBytes: UInt64 = payloadOffsetBytes + tensorPayloadBytes;
            headerEntries.append(
                "\"\(tensorProfile.name)\": {\"dtype\": \"\(dtypeCanonicalName)\", \"shape\": [\(tensorProfile.shape.map(String.init).joined(separator: ", "))], \"data_offsets\": [\(payloadOffsetBytes), \(tensorEndOffsetBytes)]}");
            let bytesPerElement: Int = Int(bitsPerElement / 8);
            let isFloatElement: Bool = tensorProfile.dtype != .uint32;
            let floatExponentByte: UInt8 = 0x3C + (weightVariantSalt & 0x01);
            var tensorPayload: Array<UInt8> = Array(repeating: UInt8(0), count: Int(tensorPayloadBytes));
            for bytePosition: Int in 0..<tensorPayload.count {
                let varyingByte: UInt8 = UInt8(
                    (tensorOrdinal * 101 + bytePosition * 37 + Int(weightVariantSalt)) & 0xFF);
                if isFloatElement && bytePosition % bytesPerElement == bytesPerElement - 1 {
                    tensorPayload[bytePosition] = floatExponentByte;
                } else {
                    tensorPayload[bytePosition] = varyingByte;
                }
            }
            payloadBytes.append(Data(tensorPayload));
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
        return SynthesizedShardBytes(fileBytes: Data(framedBytes), payloadBytes: payloadOffsetBytes);
    }

    private static func weightMapIndexBytes(
        tensorNames: Set<String>, totalPayloadBytes: UInt64) -> Array<UInt8> {
        let weightMapEntries: Array<String> = tensorNames.sorted(by: { (leftName: String, rightName: String) -> Bool in
            return Array(leftName.utf8).lexicographicallyPrecedes(Array(rightName.utf8));
        }).map { (tensorName: String) -> String in
            return "\"\(tensorName)\": \"\(TinyDenseArtifactFixture.SHARD_FILE_NAME)\"";
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

    public struct SynthesizedShardBytes {
        public var fileBytes: Data;
        public var payloadBytes: UInt64;
    }
}
