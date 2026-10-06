import XCTest;
import ModelServing;
import IpcProtocol;

/// Ported from crates/model-serving/tests/qwen3_5_hermetic/config/quantization.rs.
final class Qwen3_5ConfigQuantizationTests: XCTestCase {

    func testShouldAcceptEveryAffineQuantizationBitWidthSupportedByMlx() throws {
        for quantizationBits: UInt32 in [2, 3, 4, 5, 6, 8] {
            let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
                String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
            let modifiedConfigValue: JsonWireValue = frozenConfigValue
                .settingObjectKey(
                    path: ["quantization", "bits"], newValue: .unsignedInteger(UInt64(quantizationBits)))
                .settingObjectKey(path: ["quantization_config"], newValue: frozenConfigValue.objectValue(forKey: "quantization")!
                    .settingObjectKey(path: ["bits"], newValue: .unsignedInteger(UInt64(quantizationBits))));
            let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(modifiedConfigValue);
            let config: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
            XCTAssertEqual(config.defaultQuantizationBits(), quantizationBits);
        }
    }

    func testShouldAcceptEveryAffineQuantizationGroupSizeSupportedByMlx() throws {
        for quantizationGroupSize: UInt32 in [32, 64, 128] {
            let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
                String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
            let modifiedConfigValue: JsonWireValue = frozenConfigValue
                .settingObjectKey(
                    path: ["quantization", "group_size"],
                    newValue: .unsignedInteger(UInt64(quantizationGroupSize)))
                .settingObjectKey(
                    path: ["quantization_config"],
                    newValue: frozenConfigValue.objectValue(forKey: "quantization")!
                        .settingObjectKey(path: ["group_size"], newValue: .unsignedInteger(UInt64(quantizationGroupSize))));
            let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(modifiedConfigValue);
            let config: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
            XCTAssertEqual(config.defaultQuantizationGroupSize(), quantizationGroupSize);
        }
    }

    func testShouldRejectTheAffineQuantizationBitWidthUnsupportedByMlx() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let unsupportedBitsQuantization: JsonWireValue = frozenConfigValue.objectValue(forKey: "quantization")!
            .settingObjectKey(path: ["bits"], newValue: .unsignedInteger(7));
        let modifiedConfigValue: JsonWireValue = frozenConfigValue
            .settingObjectKey(path: ["quantization"], newValue: unsupportedBitsQuantization)
            .settingObjectKey(path: ["quantization_config"], newValue: unsupportedBitsQuantization);
        let invalidConfigBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(modifiedConfigValue);
        do {
            _ = try Qwen3_5Config.fromJsonBytes(configBytes: invalidConfigBytes);
            XCTFail("MLX-unsupported affine bits must fail");
        } catch let configError as Qwen3_5ConfigError {
            guard case .invalidConfigValueDynamic = configError else {
                XCTFail("expected InvalidConfigValueDynamic, got \(configError)");
                return;
            }
        }
    }

    func testShouldParseAStandardSixBitConfigWithoutHighBitEmbeddingOverrides() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let trimmedQuantization: JsonWireValue = frozenConfigValue
            .objectValue(forKey: "quantization")!
            .removingObjectKey(path: ["language_model.model.embed_tokens"])
            .removingObjectKey(path: ["language_model.lm_head"]);
        let modifiedConfigValue: JsonWireValue = frozenConfigValue
            .settingObjectKey(path: ["quantization"], newValue: trimmedQuantization)
            .settingObjectKey(path: ["quantization_config"], newValue: trimmedQuantization);
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(modifiedConfigValue);
        let config: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        XCTAssertEqual(
            config.quantizationProfile(forModule: "language_model.model.embed_tokens").bits, 6);
        XCTAssertEqual(config.quantizationProfile(forModule: "language_model.lm_head").bits, 6);
    }

    func testShouldCreateAnUnquantizedQuantizationProfile() {
        let profile: OptiQQuantizationProfile = .unquantized();
        XCTAssertEqual(profile.bits, 0);
        XCTAssertEqual(profile.groupSize, 0);
        XCTAssertTrue(profile.isUnquantized());
    }
}
