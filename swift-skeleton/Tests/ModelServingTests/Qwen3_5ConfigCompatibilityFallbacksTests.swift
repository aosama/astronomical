import XCTest;
import ModelServing;
import IpcProtocol;

/// Ported from crates/model-serving/tests/qwen3_5_hermetic/config/compatibility_fallbacks.rs.
final class Qwen3_5ConfigCompatibilityFallbacksTests: XCTestCase {

    func testShouldAcceptASingleElementEosTokenIdArray() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let modifiedConfigValue: JsonWireValue = frozenConfigValue.settingObjectKey(
            path: ["eos_token_id"], newValue: .array([.unsignedInteger(248046)]));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(modifiedConfigValue);
        let ornithConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        XCTAssertEqual(ornithConfig.endOfSequenceTokenIds()[0], 248046);
        XCTAssertEqual(ornithConfig.endOfSequenceTokenIds()[1], 248044);
    }

    func testShouldAcceptASingleEosTokenIdAndUsePadTokenIdAsSecond() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let modifiedConfigValue: JsonWireValue = frozenConfigValue.settingObjectKey(
            path: ["eos_token_id"], newValue: .unsignedInteger(248046));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(modifiedConfigValue);
        let ornithConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        XCTAssertEqual(ornithConfig.endOfSequenceTokenIds()[0], 248046);
        XCTAssertEqual(ornithConfig.endOfSequenceTokenIds()[1], 248044);
    }

    func testShouldAcceptAnOrnithConfigWithADifferentRopeBase() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let modifiedConfigValue: JsonWireValue = frozenConfigValue.settingObjectKey(
            path: ["text_config", "rope_theta"], newValue: .double(100000.0));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(modifiedConfigValue);
        _ = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
    }

    func testShouldAcceptEveryDeclaredEndOfSequenceTokenId() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let modifiedConfigValue: JsonWireValue = frozenConfigValue.settingObjectKey(
            path: ["eos_token_id"],
            newValue: .array([.unsignedInteger(248046), .unsignedInteger(248044), .unsignedInteger(248043)]));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(modifiedConfigValue);
        let config: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        XCTAssertEqual(config.endOfSequenceTokenIds(), [248_046, 248_044, 248_043]);
    }

    func testShouldAcceptRouterLogitsAsAnIgnoredGenerationOutputPreference() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let modifiedConfigValue: JsonWireValue = frozenConfigValue.settingObjectKey(
            path: ["text_config", "output_router_logits"], newValue: .boolean(true));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(modifiedConfigValue);
        _ = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
    }

    func testShouldParseAgentsA1StyleConfigWithMultipleMissingTopLevelFields() throws {
        var configValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.minimalValidConfigJson();
        // Simulate the exact Agents-A1-OptiQ-4bit config pattern:
        // - No top-level eos_token_id
        // - No top-level pad_token_id
        // - No top-level dtype (falls back to text_config.dtype)
        // - No text_config.output_router_logits
        // - No text_config.partial_rotary_factor (in rope_parameters only)
        configValue = configValue
            .removingObjectKey(path: ["eos_token_id"])
            .removingObjectKey(path: ["dtype"])
            .removingObjectKey(path: ["text_config", "output_router_logits"])
            .removingObjectKey(path: ["text_config", "partial_rotary_factor"])
            .settingObjectKey(path: ["text_config", "eos_token_id"], newValue: .unsignedInteger(248044))
            .settingObjectKey(path: ["text_config", "dtype"], newValue: .string("bfloat16"));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(configValue);
        let parsedConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        XCTAssertEqual(
            parsedConfig.endOfSequenceTokenIds(),
            [248044, 248046],
            "eos_token_ids should resolve from text_config with chat token appended");
        XCTAssertEqual(
            parsedConfig.activationDtype(),
            "bfloat16",
            "activation dtype should fall back to text_config.dtype");
        XCTAssertEqual(
            parsedConfig.partialRotaryFactorBits(),
            Float(0.25).bitPattern,
            "partial_rotary_factor should fall back to rope_parameters");
    }

    func testShouldParseConfigWhenRopeParametersUsesRopeTypeAlias() throws {
        var configValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.minimalValidConfigJson();
        // Some Qwen models use "rope_type" instead of "type" in rope_parameters.
        // Qwen configurations use both spellings, and serde's alias should accept both.
        configValue = configValue
            .removingObjectKey(path: ["text_config", "rope_parameters", "type"])
            .settingObjectKey(
                path: ["text_config", "rope_parameters", "rope_type"], newValue: .string("default"));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(configValue);
        let parsedConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        XCTAssertEqual(
            parsedConfig.ropeThetaBits(),
            Float(10_000_000).bitPattern,
            "rope_theta should still be parsed correctly with 'rope_type' alias");
    }

    func testShouldParseConfigWhenTextConfigOutputRouterLogitsIsAbsent() throws {
        let configValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.minimalValidConfigJson()
            .removingObjectKey(path: ["text_config", "output_router_logits"]);
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(configValue);
        _ = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
    }

    func testShouldParseConfigWhenTopLevelEosTokenIdIsAbsentButTextConfigHasIt() throws {
        var configValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.minimalValidConfigJson();
        // Remove top-level eos_token_id — the parser should fall back to
        // text_config.eos_token_id and add the Qwen chat EOS token (248046).
        configValue = configValue
            .removingObjectKey(path: ["eos_token_id"])
            .settingObjectKey(path: ["text_config", "eos_token_id"], newValue: .unsignedInteger(248044));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(configValue);
        let parsedConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        XCTAssertEqual(
            parsedConfig.endOfSequenceTokenIds(),
            [248044, 248046],
            "eos_token_ids should resolve to [text_config_eos, QWEN_CHAT_EOS] when top-level is absent");
    }

    func testShouldParseConfigWhenTopLevelPartialRotaryFactorIsAbsentButRopeParametersHasIt() throws {
        let configValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.minimalValidConfigJson()
            .removingObjectKey(path: ["text_config", "partial_rotary_factor"]);
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(configValue);
        let parsedConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        XCTAssertEqual(
            parsedConfig.partialRotaryFactorBits(),
            Float(0.25).bitPattern,
            "partial_rotary_factor should fall back to rope_parameters.partial_rotary_factor when absent at text_config level");
    }

    func testShouldAcceptAnOrnithConfigWithTiedEmbeddings() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let modifiedConfigValue: JsonWireValue = frozenConfigValue.settingObjectKey(
            path: ["tie_word_embeddings"], newValue: .boolean(true));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(modifiedConfigValue);
        let parsedConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        XCTAssertTrue(parsedConfig.hasTiedEmbeddings());
    }

    func testShouldResolveEosTokenIdsFromTextConfigAndAddChatEosToken() throws {
        var configValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.minimalValidConfigJson();
        // Remove top-level eos_token_id and pad_token_id — exactly the Agents-A1 pattern.
        configValue = configValue
            .removingObjectKey(path: ["eos_token_id"])
            .removingObjectKey(path: ["pad_token_id"])
            .settingObjectKey(path: ["text_config", "eos_token_id"], newValue: .unsignedInteger(248044));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(configValue);
        let parsedConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        XCTAssertEqual(
            parsedConfig.endOfSequenceTokenIds(),
            [248044, 248046],
            "eos_token_ids should be [text_config_eos, QWEN_CHAT_EOS_TOKEN_ID] when top-level is absent");
    }

    func testShouldResolveEosTokenIdsFromTextConfigArrayAndAddChatToken() throws {
        var configValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.minimalValidConfigJson();
        configValue = configValue
            .removingObjectKey(path: ["eos_token_id"])
            .removingObjectKey(path: ["pad_token_id"])
            .settingObjectKey(
                path: ["text_config", "eos_token_id"], newValue: .array([.unsignedInteger(248044)]));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(configValue);
        let parsedConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        XCTAssertEqual(
            parsedConfig.endOfSequenceTokenIds(),
            [248044, 248046],
            "eos_token_ids should be [text_config_eos, QWEN_CHAT_EOS] when text_config has single-element array");
    }
}
