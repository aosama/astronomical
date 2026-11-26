//! The options and metadata limits the Responses endpoint refuses, checked in
//! a fixed order so a request with several of them always names the same one.

use std::collections::BTreeMap;

use super::{
    first_unsupported_option_name, first_unsupported_option_setting, validate_metadata_limits,
};
use crate::{OpenAiResponsesRequest, OpenAiResponsesValidationError};

fn request_with(extra_fields: &str) -> OpenAiResponsesRequest {
    let request_json = format!(
        r#"{{"model":"astronomical/fake","input":"Shall I compare thee to a summer's day?"{extra_fields}}}"#
    );
    serde_json::from_str(&request_json).expect("the request should deserialize")
}

#[test]
fn should_name_no_option_when_the_request_sets_none_of_the_unsupported_ones() {
    let request = request_with(
        r#","store":false,"background":false,"truncation":"disabled","service_tier":"auto","top_logprobs":0,"parallel_tool_calls":true"#,
    );

    assert_eq!(first_unsupported_option_name(&request), None);
}

#[test]
fn should_name_each_option_that_is_refused_when_present() {
    let refused_option_names = [
        "previous_response_id",
        "conversation",
        "context_management",
        "moderation",
        "prompt",
        "prompt_cache_options",
        "prompt_cache_retention",
    ];
    for option_name in refused_option_names {
        let request = request_with(&format!(r#","{option_name}":"thou art more lovely""#));

        assert_eq!(
            first_unsupported_option_name(&request),
            Some(option_name),
            "{option_name} should be refused"
        );
    }
}

#[test]
fn should_name_each_option_that_is_refused_only_when_set_to_a_non_default_setting() {
    let refused_settings = [
        (
            r#","parallel_tool_calls":false"#,
            "parallel_tool_calls=false",
        ),
        (r#","top_logprobs":1"#, "top_logprobs"),
        (r#","text":{"verbosity":"low"}"#, "text.verbosity"),
        (r#","background":true"#, "background=true"),
        (r#","truncation":"auto""#, "truncation"),
        (r#","service_tier":"priority""#, "service_tier"),
    ];
    for (extra_fields, option_name) in refused_settings {
        let request = request_with(extra_fields);

        assert_eq!(
            first_unsupported_option_setting(&request),
            Some(option_name)
        );
    }
}

#[test]
fn should_accept_the_default_service_tiers_and_a_text_block_without_verbosity() {
    for extra_fields in [
        r#","service_tier":"default""#,
        r#","service_tier":"auto""#,
        r#","text":{"format":{"type":"text"}}"#,
    ] {
        let request = request_with(extra_fields);

        assert_eq!(first_unsupported_option_setting(&request), None);
    }
}

#[test]
fn should_name_store_before_any_other_unsupported_option() {
    let request = request_with(r#","background":true,"prompt":"x","store":true"#);

    assert_eq!(first_unsupported_option_name(&request), Some("store=true"));
}

#[test]
fn should_name_a_refused_field_before_a_refused_setting() {
    let request = request_with(r#","background":true,"moderation":{}"#);

    assert_eq!(first_unsupported_option_name(&request), Some("moderation"));
}

#[test]
fn should_name_the_first_refused_setting_when_several_are_set() {
    let request = request_with(r#","service_tier":"flex","top_logprobs":3"#);

    assert_eq!(
        first_unsupported_option_name(&request),
        Some("top_logprobs")
    );
}

fn metadata_with_entries(entry_count: usize) -> BTreeMap<String, String> {
    (0..entry_count)
        .map(|entry_position| {
            (
                format!("sonnet-{entry_position}"),
                "Shall I compare thee".to_owned(),
            )
        })
        .collect()
}

#[test]
fn should_accept_metadata_at_exactly_the_entry_and_length_limits() {
    let mut metadata = metadata_with_entries(15);
    metadata.insert("k".repeat(64), "v".repeat(512));

    assert_eq!(validate_metadata_limits(&metadata), Ok(()));
}

#[test]
fn should_reject_metadata_with_more_than_sixteen_entries() {
    assert_eq!(
        validate_metadata_limits(&metadata_with_entries(17)),
        Err(OpenAiResponsesValidationError::MetadataEntryCountExceeded)
    );
}

#[test]
fn should_reject_a_metadata_key_or_text_over_its_length_limit() {
    let mut long_key = BTreeMap::new();
    long_key.insert("k".repeat(65), "v".to_owned());
    let mut long_text = BTreeMap::new();
    long_text.insert("k".to_owned(), "v".repeat(513));

    assert_eq!(
        validate_metadata_limits(&long_key),
        Err(OpenAiResponsesValidationError::MetadataTextTooLong)
    );
    assert_eq!(
        validate_metadata_limits(&long_text),
        Err(OpenAiResponsesValidationError::MetadataTextTooLong)
    );
}
