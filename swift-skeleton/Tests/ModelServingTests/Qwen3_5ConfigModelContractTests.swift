import Foundation;
import ModelServing;
import IpcProtocol;
import Testing;
import JourneyCategories;

/// Ported from crates/model-serving/tests/qwen3_5_hermetic/config/model_contract.rs.
@Suite(.tags(.hermeticJourney))
final class Qwen3_5ConfigModelContractTests {

    @Test
    func should_parse_the_frozen_qwen35_moe_text_core_config() throws {
        let ornithConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes());
        #expect(ornithConfig.hiddenSize() == 2_048);
        #expect(ornithConfig.layerCount() == 40);
        #expect(ornithConfig.vocabularySize() == 248_320);
        #expect(ornithConfig.torchDtype() == "bfloat16");
        #expect(ornithConfig.hiddenActivation() == "silu");
        #expect(ornithConfig.rmsNormEpsilonBits() == Float(1e-6).bitPattern);
        #expect(ornithConfig.ropeThetaBits() == Float(10_000_000).bitPattern);
        #expect(ornithConfig.partialRotaryFactorBits() == Float(0.25).bitPattern);
        #expect(ornithConfig.endOfSequenceTokenIds() == [248_046, 248_044]);
        #expect(ornithConfig.hasAttentionBias() == false);
        #expect(ornithConfig.hasMlpBias() == false);
        #expect(ornithConfig.hasTiedEmbeddings() == false);
        #expect(ornithConfig.normalizesTopKProbabilities());
        #expect(ornithConfig.modelWeightStorage() == .affineQuantized);
        #expect(ornithConfig.defaultQuantizationBits() == 6);
        #expect(ornithConfig.defaultQuantizationGroupSize() == 64);
        #expect(
            ornithConfig.quantizationProfile(forModule: "language_model.model.embed_tokens").bits == 8);
        #expect(
            ornithConfig.quantizationProfile(forModule: "language_model.lm_head").bits == 8);
        #expect(ornithConfig.expertCount() == 256);
        #expect(ornithConfig.expertsPerToken() == 8);
        #expect(ornithConfig.expertIntermediateSize() == 512);
        #expect(ornithConfig.sharedExpertIntermediateSize() == 512);
        #expect(
            ornithConfig.contextMemoryReservationBytes(contextTokenCount: 1) == 20_480,
            "request admission should reserve only context-growing full-attention KV state, not all 40 decoder-layer activations");
    }

    @Test
    func should_parse_a_native_bfloat16_config_without_quantization_metadata() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let nativeBfloat16ConfigDocument: JsonWireValue = frozenConfigValue
            .removingObjectKey(path: ["quantization"])
            .removingObjectKey(path: ["quantization_config"]);
        let nativeBfloat16ConfigBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(
            nativeBfloat16ConfigDocument);
        let nativeBfloat16Config: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: nativeBfloat16ConfigBytes);
        #expect(nativeBfloat16Config.activationDtype() == "bfloat16");
        #expect(
            nativeBfloat16Config.modelWeightStorage() == .nativeBfloat16,
            "an absent quantization contract must explicitly identify native BF16 storage");
        let nativeBfloat16TensorProfiles: Array<TensorProfile> = Qwen3_5TensorSpec.qwen3_5LanguageTensorProfiles(
            qwen3_5Config: nativeBfloat16Config);
        #expect(
            nativeBfloat16TensorProfiles.contains(where: { (tensorProfile: TensorProfile) -> Bool in tensorProfile.name == "language_model.lm_head.weight" }),
            "native BF16 artifacts must retain their dense weight tensor");
        #expect(
            nativeBfloat16TensorProfiles.contains(where: { (tensorProfile: TensorProfile) -> Bool in tensorProfile.name == "language_model.lm_head.scales" }) == false,
            "native BF16 artifacts must not require affine scales");
    }

    @Test
    func should_parse_a_config_with_only_one_quantization_document() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let singleQuantizationDocument: JsonWireValue = frozenConfigValue
            .removingObjectKey(path: ["quantization_config"]);
        let singleQuantizationBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(
            singleQuantizationDocument);
        let parsedConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: singleQuantizationBytes);
        #expect(parsedConfig.modelWeightStorage() == .affineQuantized);
        #expect(parsedConfig.defaultQuantizationBits() == 6);
        #expect(parsedConfig.defaultQuantizationGroupSize() == 64);
    }

    @Test
    func should_estimate_long_context_memory_from_full_attention_key_value_state_only() throws {
        let ornithConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes());
        #expect(
            ornithConfig.contextMemoryReservationBytes(contextTokenCount: 179_350) == 3_673_088_000,
            "the 179k-token OpenCode request should reserve about 3.4 GiB of KV state, not the inflated 29.4 GB all-layer activation estimate");
    }

    @Test
    func should_reject_a_context_memory_reservation_that_overflows_the_platform_range() throws {
        let ornithConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes());
        #expect(ornithConfig.contextMemoryReservationBytes(contextTokenCount: Int.max) == nil);
    }

    @Test
    func should_use_the_declared_attention_type_for_each_decoder_layer() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let modifiedConfigValue: JsonWireValue = frozenConfigValue
            .settingObjectKey(
                path: ["text_config", "layer_types", "0"], newValue: .string("full_attention"))
            .settingObjectKey(
                path: ["text_config", "layer_types", "3"], newValue: .string("linear_attention"));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(modifiedConfigValue);
        let config: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        let tensorProfiles: Array<TensorProfile> = Qwen3_5TensorSpec.qwen3_5LanguageTensorProfiles(
            qwen3_5Config: config);
        let tensorNames: Set<String> = Set(tensorProfiles.map({ (tensorProfile: TensorProfile) -> String in tensorProfile.name }));
        #expect(tensorNames.contains("language_model.model.layers.0.self_attn.q_proj.weight"));
        #expect(tensorNames.contains("language_model.model.layers.3.linear_attn.in_proj_qkv.weight"));
    }

    @Test
    func should_reject_an_ornith_layer_schedule_with_the_wrong_number_of_layers() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let modifiedConfigValue: JsonWireValue = frozenConfigValue.settingObjectKey(
            path: ["text_config", "layer_types"], newValue: .array(Array()));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(modifiedConfigValue);
        do {
            _ = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
            Issue.record("a layer schedule of the wrong length must fail");
        } catch let configError as Qwen3_5ConfigError {
            #expect(
                configError == Qwen3_5ConfigError.layerTypeCountMismatch(
                    actualLayerTypeCount: 0, expectedLayerTypeCount: 40));
        }
    }

    @Test
    func should_reject_linear_attention_dimensions_that_exceed_the_mlx_shape_range() throws {
        var configValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.minimalValidConfigJson();
        configValue = configValue.settingObjectKey(
            path: ["text_config", "linear_num_key_heads"],
            newValue: .unsignedInteger(UInt64(UInt32.max)));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(configValue);
        do {
            _ = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
            Issue.record("linear-attention dimensions beyond the MLX shape range must fail");
        } catch let configError as Qwen3_5ConfigError {
            switch configError {
            case .invalidConfigValue, .invalidConfigValueDynamic:
                return;
            default:
                Issue.record("expected an invalid-config rejection, got \(configError)");
            }
        }
    }
}
