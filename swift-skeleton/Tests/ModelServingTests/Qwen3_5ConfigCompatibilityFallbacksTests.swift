import Foundation;
import ModelServing;
import IpcProtocol;
import Testing;
import JourneyCategories;

/// Ported from crates/model-serving/tests/qwen3_5_hermetic/config/compatibility_fallbacks.rs.
@Suite(.tags(.hermeticJourney))
final class Qwen3_5ConfigCompatibilityFallbacksTests {

    @Test
    func should_accept_a_single_element_eos_token_id_array() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let modifiedConfigValue: JsonWireValue = frozenConfigValue.settingObjectKey(
            path: ["eos_token_id"], newValue: .array([.unsignedInteger(248046)]));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(modifiedConfigValue);
        let ornithConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        #expect(ornithConfig.endOfSequenceTokenIds()[0] == 248046);
        #expect(ornithConfig.endOfSequenceTokenIds()[1] == 248044);
    }

    @Test
    func should_accept_a_single_eos_token_id_and_use_pad_token_id_as_second() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let modifiedConfigValue: JsonWireValue = frozenConfigValue.settingObjectKey(
            path: ["eos_token_id"], newValue: .unsignedInteger(248046));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(modifiedConfigValue);
        let ornithConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        #expect(ornithConfig.endOfSequenceTokenIds()[0] == 248046);
        #expect(ornithConfig.endOfSequenceTokenIds()[1] == 248044);
    }

    @Test
    func should_accept_an_ornith_config_with_a_different_rope_base() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let modifiedConfigValue: JsonWireValue = frozenConfigValue.settingObjectKey(
            path: ["text_config", "rope_theta"], newValue: .double(100000.0));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(modifiedConfigValue);
        _ = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
    }

    @Test
    func should_accept_every_declared_end_of_sequence_token_id() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let modifiedConfigValue: JsonWireValue = frozenConfigValue.settingObjectKey(
            path: ["eos_token_id"],
            newValue: .array([.unsignedInteger(248046), .unsignedInteger(248044), .unsignedInteger(248043)]));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(modifiedConfigValue);
        let config: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        #expect(config.endOfSequenceTokenIds() == [248_046, 248_044, 248_043]);
    }

    @Test
    func should_accept_router_logits_as_an_ignored_generation_output_preference() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let modifiedConfigValue: JsonWireValue = frozenConfigValue.settingObjectKey(
            path: ["text_config", "output_router_logits"], newValue: .boolean(true));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(modifiedConfigValue);
        _ = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
    }

    @Test
    func should_parse_agents_a1_style_config_with_multiple_missing_top_level_fields() throws {
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
        #expect(
            parsedConfig.endOfSequenceTokenIds() == [248044, 248046],
            "eos_token_ids should resolve from text_config with chat token appended");
        #expect(
            parsedConfig.activationDtype() == "bfloat16",
            "activation dtype should fall back to text_config.dtype");
        #expect(
            parsedConfig.partialRotaryFactorBits() == Float(0.25).bitPattern,
            "partial_rotary_factor should fall back to rope_parameters");
    }

    @Test
    func should_parse_config_when_rope_parameters_uses_rope_type_alias() throws {
        var configValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.minimalValidConfigJson();
        // Some Qwen models use "rope_type" instead of "type" in rope_parameters.
        // Qwen configurations use both spellings, and serde's alias should accept both.
        configValue = configValue
            .removingObjectKey(path: ["text_config", "rope_parameters", "type"])
            .settingObjectKey(
                path: ["text_config", "rope_parameters", "rope_type"], newValue: .string("default"));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(configValue);
        let parsedConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        #expect(
            parsedConfig.ropeThetaBits() == Float(10_000_000).bitPattern,
            "rope_theta should still be parsed correctly with 'rope_type' alias");
    }

    @Test
    func should_parse_config_when_text_config_output_router_logits_is_absent() throws {
        let configValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.minimalValidConfigJson()
            .removingObjectKey(path: ["text_config", "output_router_logits"]);
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(configValue);
        _ = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
    }

    @Test
    func should_parse_config_when_top_level_eos_token_id_is_absent_but_text_config_has_it() throws {
        var configValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.minimalValidConfigJson();
        // Remove top-level eos_token_id — the parser should fall back to
        // text_config.eos_token_id and add the Qwen chat EOS token (248046).
        configValue = configValue
            .removingObjectKey(path: ["eos_token_id"])
            .settingObjectKey(path: ["text_config", "eos_token_id"], newValue: .unsignedInteger(248044));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(configValue);
        let parsedConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        #expect(
            parsedConfig.endOfSequenceTokenIds() == [248044, 248046],
            "eos_token_ids should resolve to [text_config_eos, QWEN_CHAT_EOS] when top-level is absent");
    }

    @Test
    func should_parse_config_when_top_level_partial_rotary_factor_is_absent_but_rope_parameters_has_it() throws {
        let configValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.minimalValidConfigJson()
            .removingObjectKey(path: ["text_config", "partial_rotary_factor"]);
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(configValue);
        let parsedConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        #expect(
            parsedConfig.partialRotaryFactorBits() == Float(0.25).bitPattern,
            "partial_rotary_factor should fall back to rope_parameters.partial_rotary_factor when absent at text_config level");
    }

    @Test
    func should_accept_an_ornith_config_with_tied_embeddings() throws {
        let frozenConfigValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes(), as: UTF8.self));
        let modifiedConfigValue: JsonWireValue = frozenConfigValue.settingObjectKey(
            path: ["tie_word_embeddings"], newValue: .boolean(true));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(modifiedConfigValue);
        let parsedConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        #expect(parsedConfig.hasTiedEmbeddings());
    }

    @Test
    func should_resolve_eos_token_ids_from_text_config_and_add_chat_eos_token() throws {
        var configValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.minimalValidConfigJson();
        // Remove top-level eos_token_id and pad_token_id — exactly the Agents-A1 pattern.
        configValue = configValue
            .removingObjectKey(path: ["eos_token_id"])
            .removingObjectKey(path: ["pad_token_id"])
            .settingObjectKey(path: ["text_config", "eos_token_id"], newValue: .unsignedInteger(248044));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(configValue);
        let parsedConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        #expect(
            parsedConfig.endOfSequenceTokenIds() == [248044, 248046],
            "eos_token_ids should be [text_config_eos, QWEN_CHAT_EOS_TOKEN_ID] when top-level is absent");
    }

    @Test
    func should_resolve_eos_token_ids_from_text_config_array_and_add_chat_token() throws {
        var configValue: JsonWireValue = try Qwen3_5MoeConfigFixtures.minimalValidConfigJson();
        configValue = configValue
            .removingObjectKey(path: ["eos_token_id"])
            .removingObjectKey(path: ["pad_token_id"])
            .settingObjectKey(
                path: ["text_config", "eos_token_id"], newValue: .array([.unsignedInteger(248044)]));
        let configBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(configValue);
        let parsedConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: configBytes);
        #expect(
            parsedConfig.endOfSequenceTokenIds() == [248044, 248046],
            "eos_token_ids should be [text_config_eos, QWEN_CHAT_EOS] when text_config has single-element array");
    }
}
