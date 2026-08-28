//! Laguna load must follow the family contract, not one publisher's on-disk dialect.

use std::fs;

use astronomical_model_serving::{
    LagunaArtifactValidator, LagunaExecutionDtype, LagunaTargetNormalizer, LagunaTextArtifactError,
};
use serde_json::{Value, json};

use super::artifact_support::SyntheticLagunaArtifact;
use super::support::{config_bytes, config_value, normalize};
use super::text_support::{POOLSIDE_TEMPLATE, SyntheticLagunaTextArtifact};

#[test]
fn should_default_omitted_execution_dtype_to_bfloat16() {
    let mut omitted_dtype_config = config_value(2);
    omitted_dtype_config
        .as_object_mut()
        .expect("the fixture should be an object")
        .remove("torch_dtype");

    let omitted_dtype_contract = normalize(omitted_dtype_config);
    let mut explicit_bfloat16_config = config_value(2);
    explicit_bfloat16_config["torch_dtype"] = json!("bfloat16");

    assert_eq!(
        omitted_dtype_contract.model().execution_dtype(),
        LagunaExecutionDtype::Bfloat16
    );
    assert_eq!(omitted_dtype_contract, normalize(explicit_bfloat16_config));
}

#[test]
fn should_accept_dtype_as_the_torch_dtype_alias() {
    let mut alias_config = config_value(2);
    alias_config
        .as_object_mut()
        .expect("the fixture should be an object")
        .remove("torch_dtype");
    alias_config["dtype"] = json!("bf16");

    assert_eq!(
        normalize(alias_config).model().execution_dtype(),
        LagunaExecutionDtype::Bfloat16
    );

    let mut conflicting_alias_config = config_value(2);
    conflicting_alias_config["torch_dtype"] = json!("float16");
    conflicting_alias_config["dtype"] = json!("bfloat16");
    assert!(
        LagunaTargetNormalizer::normalize(&config_bytes(&conflicting_alias_config)).is_err(),
        "conflicting dtype aliases must not silently pick one meaning"
    );
}

#[test]
fn should_ignore_unknown_config_envelope_fields() {
    let mut config_with_unused_envelope = config_value(3);
    config_with_unused_envelope
        .as_object_mut()
        .expect("the fixture should be an object")
        .remove("torch_dtype");
    config_with_unused_envelope["vision_config"] = json!({});
    config_with_unused_envelope["unused_publisher_metadata"] = json!({"keep": true});
    config_with_unused_envelope["generation_config"] = json!({"unused_sidecar": true});

    let normalized_contract = normalize(config_with_unused_envelope);
    assert_eq!(
        normalized_contract.model().execution_dtype(),
        LagunaExecutionDtype::Bfloat16
    );
    assert_eq!(normalized_contract.model().layer_count(), 3);
}

#[test]
fn should_treat_optional_whitespace_around_poolside_tags_as_the_same_protocol() {
    let compact_descriptor = SyntheticLagunaTextArtifact::extra_small_inline().normalize();
    for tag_newline in ["", "\n"] {
        let mut spaced_artifact = SyntheticLagunaTextArtifact::extra_small_inline();
        spaced_artifact
            .set_embedded_chat_template(poolside_template_with_tag_newlines(tag_newline));
        let spaced_descriptor = spaced_artifact.normalize();
        assert_eq!(
            spaced_descriptor.default_thinking_enabled(),
            compact_descriptor.default_thinking_enabled()
        );
        assert_eq!(
            spaced_descriptor.reasoning_parser_id(),
            compact_descriptor.reasoning_parser_id()
        );
    }
}

#[test]
fn should_certify_special_tokens_from_tokenizer_json_when_the_decoder_map_is_absent() {
    let mut artifact_without_decoder = SyntheticLagunaTextArtifact::extra_small_inline();
    artifact_without_decoder
        .tokenizer_config
        .as_object_mut()
        .expect("the tokenizer config should be an object")
        .remove("added_tokens_decoder");
    artifact_without_decoder.tokenizer_config["unused_tokenizer_class"] = json!("any");
    artifact_without_decoder
        .generation_config
        .as_mut()
        .expect("generation config should exist")["unused_generation_sidecar"] = json!(true);

    let text_descriptor = artifact_without_decoder.normalize();
    assert!(text_descriptor.is_end_token(super::text_support::MODEL_EOS_TOKEN_ID));
    assert_eq!(text_descriptor.reasoning_parser_id(), "poolside_v1");
}

#[test]
fn should_still_reject_a_decoder_map_that_conflicts_with_tokenizer_json() {
    let mut conflicting_text_artifact = SyntheticLagunaTextArtifact::extra_small_inline();
    conflicting_text_artifact.tokenizer_config["added_tokens_decoder"]["2"]["content"] =
        json!("not-the-tokenizer-token");

    let normalization_error = conflicting_text_artifact
        .try_normalize()
        .expect_err("conflicting special-token maps must remain a load failure");
    assert!(matches!(
        normalization_error,
        LagunaTextArtifactError::SpecialTokenMismatch { .. }
    ));
}

#[test]
fn should_validate_a_complete_directory_that_omits_optional_publisher_metadata() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut artifact = SyntheticLagunaArtifact::dense("language_model.");
    artifact
        .config
        .as_object_mut()
        .expect("the fixture config should be an object")
        .remove("torch_dtype");
    artifact.config["unused_publisher_metadata"] = json!({"keep": true});
    artifact.write(model_directory.path());
    omit_optional_text_sidecar_fields(model_directory.path());

    let validated_artifact = LagunaArtifactValidator::new()
        .validate(model_directory.path())
        .expect("omitted optional publisher metadata must not block validation");
    assert_eq!(
        validated_artifact
            .target_contract()
            .model()
            .execution_dtype(),
        LagunaExecutionDtype::Bfloat16
    );
    assert_eq!(
        validated_artifact.text_artifact().reasoning_parser_id(),
        "poolside_v1"
    );
}

fn poolside_template_with_tag_newlines(tag_newline: &str) -> String {
    // Same protocol, parameterized padding around tags. Not a second publisher template.
    POOLSIDE_TEMPLATE
        .replace("<user>", &format!("<user>{tag_newline}"))
        .replace("</user>", &format!("{tag_newline}</user>"))
        .replace("<system>", &format!("<system>{tag_newline}"))
        .replace("</system>", &format!("{tag_newline}</system>"))
        .replace("<assistant>", &format!("<assistant>{tag_newline}"))
        .replace("</assistant>", &format!("{tag_newline}</assistant>"))
        .replace("<think>", &format!("<think>{tag_newline}"))
        .replace("</think>", &format!("{tag_newline}</think>"))
        .replace("<tool_response>", &format!("<tool_response>{tag_newline}"))
        .replace(
            "</tool_response>",
            &format!("{tag_newline}</tool_response>"),
        )
}

fn omit_optional_text_sidecar_fields(model_directory: &std::path::Path) {
    let tokenizer_config_path = model_directory.join("tokenizer_config.json");
    let mut tokenizer_config: Value = serde_json::from_slice(
        &fs::read(&tokenizer_config_path).expect("the tokenizer config should be readable"),
    )
    .expect("the tokenizer config should be JSON");
    tokenizer_config
        .as_object_mut()
        .expect("the tokenizer config should be an object")
        .remove("added_tokens_decoder");
    fs::write(
        tokenizer_config_path,
        serde_json::to_vec(&tokenizer_config).expect("the tokenizer config should serialize"),
    )
    .expect("the tokenizer config should be written");
}
