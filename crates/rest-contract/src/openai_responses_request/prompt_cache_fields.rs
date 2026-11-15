//! OpenAI's `prompt_cache_*` request fields keep OpenAI's spelling: GitHub
//! Copilot sends `prompt_cache_key` on every Responses request, and a renamed
//! field turned each of those requests into an "is unknown" refusal.

use crate::{OpenAiResponsesRequest, OpenAiResponsesValidationError};

fn validate_request_with(extra_field: &str) -> Result<(), OpenAiResponsesValidationError> {
    let request_json = format!(
        r#"{{"model":"astronomical/fake","input":"Shall I compare thee to a summer's day?",{extra_field}}}"#
    );
    serde_json::from_str::<OpenAiResponsesRequest>(&request_json)
        .expect("the request should deserialize")
        .into_parts()
        .map(|_| ())
}

#[test]
fn should_accept_the_prompt_cache_key_copilot_sends() {
    assert_eq!(
        validate_request_with(r#""prompt_cache_key":"copilot-session""#),
        Ok(())
    );
    assert_eq!(validate_request_with(r#""prompt_cache_key":null"#), Ok(()));
}

#[test]
fn should_name_the_prompt_cache_options_it_does_not_support_by_openai_names() {
    assert_eq!(
        validate_request_with(r#""prompt_cache_retention":"24h""#),
        Err(OpenAiResponsesValidationError::UnsupportedOption {
            option_name: "prompt_cache_retention",
        })
    );
    assert_eq!(
        validate_request_with(r#""prompt_cache_options":{}"#),
        Err(OpenAiResponsesValidationError::UnsupportedOption {
            option_name: "prompt_cache_options",
        })
    );
}

#[test]
fn should_refuse_a_prompt_cache_key_under_a_name_openai_does_not_define() {
    assert_eq!(
        validate_request_with(r#""astro_ssd_prompt_cache_key":"copilot-session""#),
        Err(OpenAiResponsesValidationError::UnknownField {
            field_name: "astro_ssd_prompt_cache_key".to_owned(),
        })
    );
}
