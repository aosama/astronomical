//! Beside-source coverage for the pure part of the `/v1/responses` endpoint:
//! what the reply plan reads from the request parts. The endpoint
//! orchestration itself runs under the daemon integration journeys.

use rest_contract::{OpenAiResponsesRequest, OpenAiResponsesRequestParts};

use super::responses_reply_plan;

fn shakespeare_request_parts(extra_fields: &str) -> OpenAiResponsesRequestParts {
    let request_json = format!(
        r#"{{"model":"local-model","input":"Shall I compare thee to a summer's day?"{extra_fields}}}"#
    );
    serde_json::from_str::<OpenAiResponsesRequest>(&request_json)
        .expect("the Shakespeare Responses request should parse")
        .into_parts()
        .expect("the Shakespeare Responses request should be valid")
}

#[test]
fn should_read_streaming_instructions_and_token_limit_into_the_reply_plan() {
    let request_parts = shakespeare_request_parts(
        r#","stream":true,"instructions":"Answer as a sonnet.","max_output_tokens":128"#,
    );

    let reply_plan = responses_reply_plan(&request_parts);

    assert!(reply_plan.should_stream_response);
    assert_eq!(
        reply_plan.instructions.as_deref(),
        Some("Answer as a sonnet.")
    );
    assert_eq!(
        reply_plan.request_configuration.max_output_tokens,
        Some(128)
    );
}

#[test]
fn should_plan_a_non_streaming_reply_without_extras_for_a_bare_request() {
    let request_parts = shakespeare_request_parts("");

    let reply_plan = responses_reply_plan(&request_parts);

    assert!(!reply_plan.should_stream_response);
    assert!(!reply_plan.reasoning_excluded);
    assert!(reply_plan.structured_output.is_none());
    assert!(reply_plan.instructions.is_none());
    assert_eq!(reply_plan.request_configuration.max_output_tokens, None);
    assert!(reply_plan.request_configuration.tools.is_empty());
}

#[test]
fn should_echo_the_requested_sampling_settings_in_the_reply_plan() {
    let request_parts = shakespeare_request_parts(r#","temperature":0.5,"top_p":0.9"#);

    let reply_plan = responses_reply_plan(&request_parts);

    assert_eq!(reply_plan.request_configuration.temperature, Some(0.5));
    assert_eq!(reply_plan.request_configuration.top_p, Some(0.9));
}
