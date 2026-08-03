use super::*;

#[test]
fn should_accept_a_single_element_eos_token_id_array() {
    let mut config_value = serde_json::from_slice::<Value>(&certified_ornith_config_bytes())
        .expect("the certified test config should decode as JSON");
    // Single-element array should also work
    config_value["eos_token_id"] = json!([248046]);
    let config_bytes = serde_json::to_vec(&config_value)
        .expect("the modified Ornith config should serialize as JSON");

    let ornith_config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("a single-element eos_token_id array should be accepted");
    assert_eq!(ornith_config.end_of_sequence_token_ids()[0], 248046);
    assert_eq!(ornith_config.end_of_sequence_token_ids()[1], 248044);
}

#[test]
fn should_accept_a_single_eos_token_id_and_use_pad_token_id_as_second() {
    let mut config_value = serde_json::from_slice::<Value>(&certified_ornith_config_bytes())
        .expect("the certified test config should decode as JSON");
    // Simulate oQ6e-style config with a single eos_token_id integer
    config_value["eos_token_id"] = json!(248046);
    let config_bytes = serde_json::to_vec(&config_value)
        .expect("the modified Ornith config should serialize as JSON");

    let ornith_config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("a single eos_token_id should be accepted with pad_token_id as second stop token");
    // The primary EOS token should be 248046
    assert_eq!(ornith_config.end_of_sequence_token_ids()[0], 248046);
    // The second should be the pad_token_id (248044)
    assert_eq!(ornith_config.end_of_sequence_token_ids()[1], 248044);
}

#[test]
fn should_accept_an_ornith_config_with_a_different_rope_base() {
    let mut config_value = serde_json::from_slice::<Value>(&certified_ornith_config_bytes())
        .expect("the certified test config should decode as JSON");
    config_value["text_config"]["rope_theta"] = json!(100000.0);
    let config_bytes = serde_json::to_vec(&config_value)
        .expect("the modified Ornith config should serialize as JSON");

    assert!(Qwen3_5Config::from_json_bytes(&config_bytes).is_ok());
}

#[test]
fn should_accept_every_declared_end_of_sequence_token_id() {
    let mut config_value = serde_json::from_slice::<Value>(&certified_ornith_config_bytes())
        .expect("the certified test config should decode as JSON");
    config_value["eos_token_id"] = json!([248046, 248044, 248043]);
    let config_bytes = serde_json::to_vec(&config_value)
        .expect("the modified Ornith config should serialize as JSON");

    let config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("all declared end-of-sequence token IDs should be retained");
    assert_eq!(
        config.end_of_sequence_token_ids(),
        &[248_046, 248_044, 248_043]
    );
}

#[test]
fn should_accept_router_logits_as_an_ignored_generation_output_preference() {
    let mut config_value = serde_json::from_slice::<Value>(&certified_ornith_config_bytes())
        .expect("the certified test config should decode as JSON");
    config_value["text_config"]["output_router_logits"] = json!(true);
    let config_bytes = serde_json::to_vec(&config_value)
        .expect("the modified Ornith config should serialize as JSON");

    assert!(Qwen3_5Config::from_json_bytes(&config_bytes).is_ok());
}

#[test]
fn should_parse_agents_a1_style_config_with_multiple_missing_top_level_fields() {
    let mut config_value = minimal_valid_config_json();
    // Simulate the exact Agents-A1-OptiQ-4bit config pattern:
    // - No top-level eos_token_id
    // - No top-level pad_token_id
    // - No top-level dtype (falls back to text_config.dtype)
    // - No text_config.output_router_logits
    // - No text_config.partial_rotary_factor (in rope_parameters only)
    config_value.as_object_mut().unwrap().remove("eos_token_id");
    config_value.as_object_mut().unwrap().remove("dtype");
    config_value["text_config"]["eos_token_id"] = json!(248044);
    config_value["text_config"]["dtype"] = json!("bfloat16");
    config_value["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("output_router_logits");
    config_value["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("partial_rotary_factor");

    let config_bytes = serde_json::to_vec(&config_value)
        .expect("the Agents-A1-style config should serialize as JSON");

    let parsed_config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("Agents-A1-style config should parse with all field fallbacks working together");

    assert_eq!(
        parsed_config.end_of_sequence_token_ids(),
        [248044, 248046],
        "eos_token_ids should resolve from text_config with chat token appended"
    );
    assert_eq!(
        parsed_config.activation_dtype(),
        "bfloat16",
        "activation dtype should fall back to text_config.dtype"
    );
    assert_eq!(
        parsed_config.partial_rotary_factor_bits(),
        0.25_f32.to_bits(),
        "partial_rotary_factor should fall back to rope_parameters"
    );
}

#[test]
fn should_parse_config_when_rope_parameters_uses_rope_type_alias() {
    let mut config_value = minimal_valid_config_json();
    // Some Qwen models use "rope_type" instead of "type" in rope_parameters.
    // Qwen configurations use both spellings, and serde's alias should accept both.
    let rope_params = config_value["text_config"]["rope_parameters"]
        .as_object_mut()
        .unwrap();
    rope_params.remove("type");
    rope_params.insert("rope_type".to_owned(), json!("default"));

    let config_bytes =
        serde_json::to_vec(&config_value).expect("the modified config should serialize as JSON");

    let parsed_config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("config should parse when rope_parameters uses 'rope_type' alias");
    assert_eq!(
        parsed_config.rope_theta_bits(),
        10_000_000_f32.to_bits(),
        "rope_theta should still be parsed correctly with 'rope_type' alias"
    );
}

#[test]
fn should_parse_config_when_text_config_output_router_logits_is_absent() {
    let mut config_value = minimal_valid_config_json();
    config_value["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("output_router_logits");

    let config_bytes =
        serde_json::to_vec(&config_value).expect("the modified config should serialize as JSON");

    Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("an unused output preference should not be required");
}

#[test]
fn should_parse_config_when_top_level_eos_token_id_is_absent_but_text_config_has_it() {
    let mut config_value = minimal_valid_config_json();
    // Remove top-level eos_token_id — the parser should fall back to
    // text_config.eos_token_id and add the Qwen chat EOS token (248046).
    config_value.as_object_mut().unwrap().remove("eos_token_id");
    // Set text_config.eos_token_id to a single integer (as Agents-A1 does).
    config_value["text_config"]["eos_token_id"] = json!(248044);

    let config_bytes =
        serde_json::to_vec(&config_value).expect("the modified config should serialize as JSON");

    let parsed_config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("config should parse when top-level eos_token_id is absent and text_config.eos_token_id is present");

    // The resolved eos_token_ids should be [248044 (from text_config), 248046 (chat EOS appended)],
    // following the Qwen3.5 compatibility rule: text_config value comes first,
    // then the chat EOS token is appended if not already present.
    assert_eq!(
        parsed_config.end_of_sequence_token_ids(),
        [248044, 248046],
        "eos_token_ids should resolve to [text_config_eos, QWEN_CHAT_EOS] when top-level is absent"
    );
}

#[test]
fn should_parse_config_when_top_level_partial_rotary_factor_is_absent_but_rope_parameters_has_it() {
    let mut config_value = minimal_valid_config_json();
    // Remove the top-level partial_rotary_factor from text_config —
    // the parser should fall back to rope_parameters.partial_rotary_factor,
    // matching the Qwen3.5 configuration fallback.
    config_value["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("partial_rotary_factor");
    // rope_parameters.partial_rotary_factor is still present (0.25).

    let config_bytes =
        serde_json::to_vec(&config_value).expect("the modified config should serialize as JSON");

    let parsed_config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("config should parse when text_config.partial_rotary_factor is absent and rope_parameters.partial_rotary_factor is present");
    assert_eq!(
        parsed_config.partial_rotary_factor_bits(),
        0.25_f32.to_bits(),
        "partial_rotary_factor should fall back to rope_parameters.partial_rotary_factor when absent at text_config level"
    );
}

#[test]
fn should_reject_an_ornith_config_with_tied_embeddings() {
    let mut config_value = serde_json::from_slice::<Value>(&certified_ornith_config_bytes())
        .expect("the certified test config should decode as JSON");
    config_value["tie_word_embeddings"] = json!(true);
    let config_bytes = serde_json::to_vec(&config_value)
        .expect("the modified Ornith config should serialize as JSON");

    assert!(matches!(
        Qwen3_5Config::from_json_bytes(&config_bytes),
        Err(Qwen3_5ConfigError::UnexpectedBooleanValue {
            field_name: "tie_word_embeddings",
            expected_value: false,
            actual_value: true,
        })
    ));
}

#[test]
fn should_resolve_eos_token_ids_from_text_config_and_add_chat_eos_token() {
    let mut config_value = minimal_valid_config_json();
    // Remove top-level eos_token_id and pad_token_id — exactly the Agents-A1 pattern.
    config_value.as_object_mut().unwrap().remove("eos_token_id");
    config_value.as_object_mut().unwrap().remove("pad_token_id");
    config_value["text_config"]["eos_token_id"] = json!(248044);

    let config_bytes =
        serde_json::to_vec(&config_value).expect("the modified config should serialize as JSON");

    let parsed_config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("config should parse with text_config-only eos_token_id");

    // Following the Qwen3.5 end-of-sequence fallback:
    // - text_config.eos_token_id = 248044
    // - Chat EOS token 248046 is not in the list, so it gets appended
    // - Result: [248044, 248046]
    assert_eq!(
        parsed_config.end_of_sequence_token_ids(),
        [248044, 248046],
        "eos_token_ids should be [text_config_eos, QWEN_CHAT_EOS_TOKEN_ID] when top-level is absent"
    );
}

#[test]
fn should_resolve_eos_token_ids_from_text_config_array_and_add_chat_token() {
    let mut config_value = minimal_valid_config_json();
    config_value.as_object_mut().unwrap().remove("eos_token_id");
    config_value.as_object_mut().unwrap().remove("pad_token_id");
    // text_config.eos_token_id as an array
    config_value["text_config"]["eos_token_id"] = json!([248044]);

    let config_bytes =
        serde_json::to_vec(&config_value).expect("the modified config should serialize as JSON");

    let parsed_config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("config should parse with text_config.eos_token_id as array");

    assert_eq!(
        parsed_config.end_of_sequence_token_ids(),
        [248044, 248046],
        "eos_token_ids should be [text_config_eos, QWEN_CHAT_EOS] when text_config has single-element array"
    );
}
