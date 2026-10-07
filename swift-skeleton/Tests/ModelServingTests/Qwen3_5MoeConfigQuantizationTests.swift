import Foundation;
import ModelServing;
import IpcProtocol;
import Testing;
import JourneyCategories;

/// Ported from crates/model-serving/tests/qwen3_5_moe_hermetic/config/quantization.rs.
@Suite(.tags(.hermeticJourney))
final class Qwen3_5MoeConfigQuantizationTests {

    @Test
    func should_reject_a_router_gate_quantization_override_with_invalid_bits() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let invalidGateOverrideQuantization: JsonWireValue = frozenConfigValue.objectValue(forKey: "quantization")!
            .settingObjectKey(
                path: ["language_model.model.layers.0.mlp.gate", "bits"],
                newValue: .unsignedInteger(7));
        let modifiedConfigValue: JsonWireValue = frozenConfigValue
            .settingObjectKey(path: ["quantization"], newValue: invalidGateOverrideQuantization)
            .settingObjectKey(path: ["quantization_config"], newValue: invalidGateOverrideQuantization);
        let invalidConfigBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(modifiedConfigValue);
        do {
            _ = try Qwen3_5Config.fromJsonBytes(configBytes: invalidConfigBytes);
            Issue.record("a router-gate quantization override with MLX-unsupported bits must fail");
        } catch let configError as Qwen3_5ConfigError {
            #expect(
                configError == .unsupportedQuantizationOverrideBits(
                    moduleName: "language_model.model.layers.0.mlp.gate", actualValue: 7));
        }
    }

    @Test
    func should_parse_a_sparse_mixture_of_experts_quantization_config_with_mixed_group_sizes() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let sparseQuantization: JsonWireValue = try Qwen3_5MoeConfigFixtures
            .wireValue(#"{"group_size": 64, "bits": 6, "mode": "affine"}"#)
            .settingObjectKey(
                path: ["language_model.model.embed_tokens"],
                newValue: try Qwen3_5MoeConfigFixtures.wireValue(
                    #"{"group_size": 64, "bits": 8, "mode": "affine"}"#))
            .settingObjectKey(
                path: ["language_model.lm_head"],
                newValue: try Qwen3_5MoeConfigFixtures.wireValue(
                    #"{"group_size": 64, "bits": 8, "mode": "affine"}"#))
            .settingObjectKey(
                path: ["language_model.model.layers.0.mlp.shared_expert_gate"],
                newValue: try Qwen3_5MoeConfigFixtures.wireValue(
                    #"{"group_size": 64, "bits": 8, "mode": "affine"}"#))
            .settingObjectKey(
                path: ["language_model.model.layers.0.mlp.shared_expert.down_proj"],
                newValue: try Qwen3_5MoeConfigFixtures.wireValue(
                    #"{"group_size": 128, "bits": 8, "mode": "affine"}"#))
            .settingObjectKey(
                path: ["language_model.model.layers.0.linear_attn.out_proj"],
                newValue: try Qwen3_5MoeConfigFixtures.wireValue(
                    #"{"group_size": 128, "bits": 6, "mode": "affine"}"#))
            .settingObjectKey(
                path: ["language_model.model.layers.0.linear_attn.in_proj_qkv"],
                newValue: try Qwen3_5MoeConfigFixtures.wireValue(
                    #"{"group_size": 64, "bits": 8, "mode": "affine"}"#));
        let modifiedConfigValue: JsonWireValue = frozenConfigValue
            .settingObjectKey(path: ["quantization"], newValue: sparseQuantization)
            .settingObjectKey(path: ["quantization_config"], newValue: sparseQuantization);
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(modifiedConfigValue);
        let config: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);

        #expect(config.defaultQuantizationBits() == 6);
        #expect(config.defaultQuantizationGroupSize() == 64);
        let embedProfile: OptiQQuantizationProfile = config.quantizationProfile(
            forModule: "language_model.model.embed_tokens");
        #expect(embedProfile.bits == 8);
        #expect(embedProfile.groupSize == 64);
        let sharedDownProfile: OptiQQuantizationProfile = config.quantizationProfile(
            forModule: "language_model.model.layers.0.mlp.shared_expert.down_proj");
        #expect(sharedDownProfile.bits == 8);
        #expect(sharedDownProfile.groupSize == 128);
        let outProjProfile: OptiQQuantizationProfile = config.quantizationProfile(
            forModule: "language_model.model.layers.0.linear_attn.out_proj");
        #expect(outProjProfile.bits == 6);
        #expect(outProjProfile.groupSize == 128);
        let switchMlpProfile: OptiQQuantizationProfile = config.quantizationProfile(
            forModule: "language_model.model.layers.0.mlp.switch_mlp.gate_proj");
        #expect(switchMlpProfile.bits == 6);
        #expect(switchMlpProfile.groupSize == 64);
    }

    @Test
    func should_resolve_unquantized_modules_from_shard_index_when_scales_are_absent() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let sparseQuantization: JsonWireValue = try Qwen3_5MoeConfigFixtures
            .wireValue(#"{"group_size": 64, "bits": 6, "mode": "affine"}"#)
            .settingObjectKey(
                path: ["language_model.model.embed_tokens"],
                newValue: try Qwen3_5MoeConfigFixtures.wireValue(
                    #"{"group_size": 64, "bits": 8, "mode": "affine"}"#))
            .settingObjectKey(
                path: ["language_model.lm_head"],
                newValue: try Qwen3_5MoeConfigFixtures.wireValue(
                    #"{"group_size": 64, "bits": 8, "mode": "affine"}"#));
        let modifiedConfigValue: JsonWireValue = frozenConfigValue
            .settingObjectKey(path: ["quantization"], newValue: sparseQuantization)
            .settingObjectKey(path: ["quantization_config"], newValue: sparseQuantization);
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(modifiedConfigValue);
        var config: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        let gateBeforeProfile: OptiQQuantizationProfile = config.quantizationProfile(
            forModule: "language_model.model.layers.0.mlp.gate");
        #expect(gateBeforeProfile.bits == 6);

        var shardTensorNames: Set<String> = [];
        shardTensorNames.insert("language_model.model.layers.0.mlp.gate.weight");
        shardTensorNames.insert("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight");
        shardTensorNames.insert("language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales");
        config.resolveUnquantizedModulesFromShardIndex(shardTensorNames: shardTensorNames);

        let gateAfterProfile: OptiQQuantizationProfile = config.quantizationProfile(
            forModule: "language_model.model.layers.0.mlp.gate");
        #expect(gateAfterProfile.isUnquantized());
        let switchMlpProfile: OptiQQuantizationProfile = config.quantizationProfile(
            forModule: "language_model.model.layers.0.mlp.switch_mlp.gate_proj");
        #expect(switchMlpProfile.bits == 6);
    }

    @Test
    func should_not_resolve_gates_as_unquantized_when_scales_are_present() throws {
        let optiqConfigBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.frozenOrnith10OptiQConfigBytes();
        var config: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: optiqConfigBytes);

        var shardTensorNames: Set<String> = [];
        shardTensorNames.insert("language_model.model.layers.0.mlp.gate.weight");
        shardTensorNames.insert("language_model.model.layers.0.mlp.gate.scales");
        shardTensorNames.insert("language_model.model.layers.0.mlp.gate.biases");
        config.resolveUnquantizedModulesFromShardIndex(shardTensorNames: shardTensorNames);

        let gateProfile: OptiQQuantizationProfile = config.quantizationProfile(
            forModule: "language_model.model.layers.0.mlp.gate");
        #expect(gateProfile.isUnquantized() == false);
        #expect(gateProfile.bits == 4);
    }

    @Test
    func should_not_hide_missing_scales_when_affine_biases_are_present() throws {
        let optiqConfigBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.frozenOrnith10OptiQConfigBytes();
        var config: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: optiqConfigBytes);

        var shardTensorNames: Set<String> = [];
        shardTensorNames.insert("language_model.model.layers.0.mlp.gate.weight");
        shardTensorNames.insert("language_model.model.layers.0.mlp.gate.biases");
        config.resolveUnquantizedModulesFromShardIndex(shardTensorNames: shardTensorNames);

        let gateProfile: OptiQQuantizationProfile = config.quantizationProfile(
            forModule: "language_model.model.layers.0.mlp.gate");
        #expect(gateProfile.isUnquantized() == false);
    }

    @Test
    func should_parse_the_sparse_mixed_precision_quantization_config() throws {
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.frozenSparseMixedPrecisionConfigBytes();
        let config: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);

        #expect(config.defaultQuantizationBits() == 6);
        #expect(config.defaultQuantizationGroupSize() == 64);
        let moduleProfileExpectations: Array<(
            moduleName: String, expectedBits: UInt32, expectedGroupSize: UInt32
        )> = [
            ("language_model.model.embed_tokens", 8, 64),
            ("language_model.lm_head", 8, 64),
            ("language_model.model.layers.0.mlp.shared_expert.gate_proj", 8, 128),
            ("language_model.model.layers.0.mlp.shared_expert_gate", 8, 64),
            ("language_model.model.layers.0.linear_attn.out_proj", 6, 128),
            ("language_model.model.layers.0.linear_attn.in_proj_qkv", 8, 64),
            ("language_model.model.layers.1.linear_attn.in_proj_qkv", 6, 64),
            ("language_model.model.layers.3.self_attn.q_proj", 8, 64),
            ("language_model.model.layers.11.self_attn.q_proj", 6, 64),
            ("language_model.model.layers.0.mlp.switch_mlp.gate_proj", 6, 64),
        ];
        for moduleProfileExpectation in moduleProfileExpectations {
            let moduleProfile: OptiQQuantizationProfile = config.quantizationProfile(
                forModule: moduleProfileExpectation.moduleName);
            #expect(
                moduleProfile.bits == moduleProfileExpectation.expectedBits,
                "bits for \(moduleProfileExpectation.moduleName)");
            #expect(
                moduleProfile.groupSize == moduleProfileExpectation.expectedGroupSize,
                "group size for \(moduleProfileExpectation.moduleName)");
        }
    }

    @Test
    func should_resolve_the_sparse_override_router_as_unquantized() throws {
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.frozenSparseMixedPrecisionConfigBytes();
        var config: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);

        // The artifact stores the router gates and the normalization weights as
        // plain native tensors with no scales or biases.
        var shardTensorNames: Set<String> = [];
        for decoderLayerIndex: Int in 0..<40 {
            shardTensorNames.insert(
                "language_model.model.layers.\(decoderLayerIndex).mlp.gate.weight");
        }
        shardTensorNames.insert("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight");
        shardTensorNames.insert("language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales");
        shardTensorNames.insert("language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases");
        config.resolveUnquantizedModulesFromShardIndex(shardTensorNames: shardTensorNames);

        for unquantizedModuleName: String in [
            "language_model.model.layers.0.mlp.gate",
            "language_model.model.layers.39.mlp.gate",
        ] {
            let moduleProfile: OptiQQuantizationProfile = config.quantizationProfile(
                forModule: unquantizedModuleName);
            #expect(
                moduleProfile.isUnquantized(),
                "\(unquantizedModuleName) should resolve as native floating point");
        }
        let switchMlpProfile: OptiQQuantizationProfile = config.quantizationProfile(
            forModule: "language_model.model.layers.0.mlp.switch_mlp.gate_proj");
        #expect(switchMlpProfile.bits == 6);
        #expect(switchMlpProfile.groupSize == 64);
    }
}
