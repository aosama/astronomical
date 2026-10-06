import XCTest;
import ModelServing;
import IpcProtocol;

/// Ported from crates/model-serving/tests/qwen3_5_hermetic/config/model_contract.rs.
final class Qwen3_5ConfigModelContractTests: XCTestCase {

    func testShouldParseTheFrozenQwen35MoeTextCoreConfig() throws {
        let ornithConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes());
        XCTAssertEqual(ornithConfig.hiddenSize(), 2_048);
        XCTAssertEqual(ornithConfig.layerCount(), 40);
        XCTAssertEqual(ornithConfig.vocabularySize(), 248_320);
        XCTAssertEqual(ornithConfig.mtpLayerCount(), 1);
        XCTAssertEqual(ornithConfig.torchDtype(), "bfloat16");
        XCTAssertEqual(ornithConfig.hiddenActivation(), "silu");
        XCTAssertEqual(ornithConfig.rmsNormEpsilonBits(), Float(1e-6).bitPattern);
        XCTAssertEqual(ornithConfig.ropeThetaBits(), Float(10_000_000).bitPattern);
        XCTAssertEqual(ornithConfig.partialRotaryFactorBits(), Float(0.25).bitPattern);
        XCTAssertEqual(ornithConfig.endOfSequenceTokenIds(), [248_046, 248_044]);
        XCTAssertFalse(ornithConfig.hasAttentionBias());
        XCTAssertFalse(ornithConfig.hasMlpBias());
        XCTAssertFalse(ornithConfig.hasTiedEmbeddings());
        XCTAssertTrue(ornithConfig.normalizesTopKProbabilities());
        XCTAssertEqual(ornithConfig.modelWeightStorage(), .affineQuantized);
        XCTAssertEqual(ornithConfig.defaultQuantizationBits(), 6);
        XCTAssertEqual(ornithConfig.defaultQuantizationGroupSize(), 64);
        XCTAssertEqual(
            ornithConfig.quantizationProfile(forModule: "language_model.model.embed_tokens").bits, 8);
        XCTAssertEqual(
            ornithConfig.quantizationProfile(forModule: "language_model.lm_head").bits, 8);
        XCTAssertEqual(ornithConfig.expertCount(), 256);
        XCTAssertEqual(ornithConfig.expertsPerToken(), 8);
        XCTAssertEqual(ornithConfig.expertIntermediateSize(), 512);
        XCTAssertEqual(ornithConfig.sharedExpertIntermediateSize(), 512);
        XCTAssertEqual(
            ornithConfig.contextMemoryReservationBytes(contextTokenCount: 1),
            20_480,
            "request admission should reserve only context-growing full-attention KV state, not all 40 decoder-layer activations");
    }

    func testShouldParseANativeBfloat16ConfigWithoutQuantizationMetadata() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let nativeBfloat16ConfigDocument: JsonWireValue = frozenConfigValue
            .removingObjectKey(path: ["quantization"])
            .removingObjectKey(path: ["quantization_config"]);
        let nativeBfloat16ConfigBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(
            nativeBfloat16ConfigDocument);
        let nativeBfloat16Config: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: nativeBfloat16ConfigBytes);
        XCTAssertEqual(nativeBfloat16Config.activationDtype(), "bfloat16");
        XCTAssertEqual(
            nativeBfloat16Config.modelWeightStorage(),
            .nativeBfloat16,
            "an absent quantization contract must explicitly identify native BF16 storage");
        let nativeBfloat16TensorProfiles: Array<TensorProfile> = Qwen3_5TensorSpec.qwen3_5LanguageTensorProfiles(
            qwen3_5Config: nativeBfloat16Config);
        XCTAssertTrue(
            nativeBfloat16TensorProfiles.contains(where: { (tensorProfile: TensorProfile) -> Bool in tensorProfile.name == "language_model.lm_head.weight" }),
            "native BF16 artifacts must retain their dense weight tensor");
        XCTAssertFalse(
            nativeBfloat16TensorProfiles.contains(where: { (tensorProfile: TensorProfile) -> Bool in tensorProfile.name == "language_model.lm_head.scales" }),
            "native BF16 artifacts must not require affine scales");
    }

    func testShouldParseAConfigWithOnlyOneQuantizationDocument() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let singleQuantizationDocument: JsonWireValue = frozenConfigValue
            .removingObjectKey(path: ["quantization_config"]);
        let singleQuantizationBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(
            singleQuantizationDocument);
        let parsedConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: singleQuantizationBytes);
        XCTAssertEqual(parsedConfig.modelWeightStorage(), .affineQuantized);
        XCTAssertEqual(parsedConfig.defaultQuantizationBits(), 6);
        XCTAssertEqual(parsedConfig.defaultQuantizationGroupSize(), 64);
    }

    func testShouldEstimateLongContextMemoryFromFullAttentionKeyValueStateOnly() throws {
        let ornithConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes());
        XCTAssertEqual(
            ornithConfig.contextMemoryReservationBytes(contextTokenCount: 179_350),
            3_673_088_000,
            "the 179k-token OpenCode request should reserve about 3.4 GiB of KV state, not the inflated 29.4 GB all-layer activation estimate");
    }

    func testShouldRejectAContextMemoryReservationThatOverflowsThePlatformRange() throws {
        let ornithConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes());
        XCTAssertNil(ornithConfig.contextMemoryReservationBytes(contextTokenCount: Int.max));
    }

    func testShouldUseTheDeclaredAttentionTypeForEachDecoderLayer() throws {
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
        XCTAssertTrue(tensorNames.contains("language_model.model.layers.0.self_attn.q_proj.weight"));
        XCTAssertTrue(tensorNames.contains("language_model.model.layers.3.linear_attn.in_proj_qkv.weight"));
    }

    func testShouldRejectAnOrnithLayerScheduleWithTheWrongNumberOfLayers() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let modifiedConfigValue: JsonWireValue = frozenConfigValue.settingObjectKey(
            path: ["text_config", "layer_types"], newValue: .array(Array()));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(modifiedConfigValue);
        do {
            _ = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
            XCTFail("a layer schedule of the wrong length must fail");
        } catch let configError as Qwen3_5ConfigError {
            XCTAssertEqual(
                configError,
                .layerTypeCountMismatch(actualLayerTypeCount: 0, expectedLayerTypeCount: 40));
        }
    }

    func testShouldRejectLinearAttentionDimensionsThatExceedTheMlxShapeRange() throws {
        var configValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.minimalValidConfigJson();
        configValue = configValue.settingObjectKey(
            path: ["text_config", "linear_num_key_heads"],
            newValue: .unsignedInteger(UInt64(UInt32.max)));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(configValue);
        do {
            _ = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
            XCTFail("linear-attention dimensions beyond the MLX shape range must fail");
        } catch let configError as Qwen3_5ConfigError {
            switch configError {
            case .invalidConfigValue, .invalidConfigValueDynamic:
                return;
            default:
                XCTFail("expected an invalid-config rejection, got \(configError)");
            }
        }
    }
}
