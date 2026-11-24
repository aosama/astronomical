//! The shared `/v1/models` sample: `crates/contracts/rest-contract/fixtures/openai_model_list.json`.
//!
//! The OpenCode plugin in `integrations/opencode` parses this same file in
//! its tests. Holding the file to what these contract types serialize means a
//! renamed, retyped or removed field fails here first; once the file is
//! updated to match, the plugin's tests fail until the plugin follows. The
//! parts mirror what the daemon's models endpoint fills in for each kind of
//! model, one of every kind the listing can carry.

use std::path::PathBuf;

use crate::{
    OpenAiDecisionModelParts, OpenAiEmbeddingModelParts, OpenAiImageModelParts, OpenAiModel,
    OpenAiModelList, OpenAiModelParts, OpenAiSpeechModelParts,
};

const FIXTURE_RELATIVE_PATH: &str = "fixtures/openai_model_list.json";
const ADVERTISED_AT_UNIX_SECONDS: u64 = 1_784_231_803;
const MODEL_OWNER: &str = "astronomical";

fn owned_strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

/// A chat model as the daemon advertises one: every generation endpoint, and
/// the reasoning and tool-call formats it uses when the model supports them.
fn chat_model(
    model_id: &str,
    context_window: u32,
    max_input_tokens: u32,
    input_modalities: &[&str],
) -> OpenAiModel {
    OpenAiModel::from_parts(OpenAiModelParts {
        model_id: model_id.to_owned(),
        created: ADVERTISED_AT_UNIX_SECONDS,
        owned_by: MODEL_OWNER.to_owned(),
        context_window,
        max_input_tokens,
        max_output_tokens: 65_535,
        input_modalities: owned_strings(input_modalities),
        output_modalities: owned_strings(&["text"]),
        supports_streaming: true,
        supports_reasoning: true,
        reasoning_format: Some(
            "openai_chat_reasoning_content_and_responses_reasoning_summary_text".to_owned(),
        ),
        supports_tool_calls: true,
        tool_call_format: Some("openai_function_call".to_owned()),
        supported_endpoints: owned_strings(&[
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/messages",
        ]),
        supports_structured_outputs: true,
        structured_output_enforcement: Some("logits_mask".to_owned()),
    })
    .expect("a valid chat model should be advertised")
}

fn helper_models() -> Vec<OpenAiModel> {
    vec![
        OpenAiModel::from_image_parts(OpenAiImageModelParts {
            model_id: "image-gen-model".to_owned(),
            created: ADVERTISED_AT_UNIX_SECONDS,
            owned_by: MODEL_OWNER.to_owned(),
            input_modalities: owned_strings(&["text"]),
            output_modalities: owned_strings(&["image"]),
            supported_endpoints: owned_strings(&["/v1/images/generations"]),
        })
        .expect("a valid image model should be advertised"),
        OpenAiModel::from_embedding_parts(OpenAiEmbeddingModelParts {
            model_id: "embedding-model".to_owned(),
            created: ADVERTISED_AT_UNIX_SECONDS,
            owned_by: MODEL_OWNER.to_owned(),
            vector_width: 768,
            max_input_tokens: 8_192,
        })
        .expect("a valid embedding model should be advertised"),
        OpenAiModel::from_speech_parts(OpenAiSpeechModelParts {
            model_id: "speech-model".to_owned(),
            created: ADVERTISED_AT_UNIX_SECONDS,
            owned_by: MODEL_OWNER.to_owned(),
            voices: owned_strings(&["af_heart"]),
            default_voice: "af_heart".to_owned(),
            sample_rate_hertz: 24_000,
            supports_speed: true,
        })
        .expect("a valid speech model should be advertised"),
        OpenAiModel::from_decision_parts(OpenAiDecisionModelParts {
            model_id: "decision-model".to_owned(),
            created: ADVERTISED_AT_UNIX_SECONDS,
            owned_by: MODEL_OWNER.to_owned(),
            maximum_tokens_per_question: 512,
        })
        .expect("a valid decision model should be advertised"),
    ]
}

/// The listing the fixture records: two chat models first, because the
/// plugin's tests index them, then one model of every other kind.
fn fixture_model_list() -> OpenAiModelList {
    let mut advertised_models = vec![
        chat_model("text-only-model", 262_144, 196_608, &["text"]),
        chat_model("vision-model", 131_072, 65_536, &["text", "image"]),
    ];
    advertised_models.extend(helper_models());
    OpenAiModelList::from_models(advertised_models)
}

fn committed_fixture() -> serde_json::Value {
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_RELATIVE_PATH);
    let fixture_text = std::fs::read_to_string(&fixture_path).unwrap_or_else(|failure| {
        panic!("{} should be readable: {failure}", fixture_path.display())
    });
    serde_json::from_str(&fixture_text).expect("the committed model list should be JSON")
}

// Positive: the committed sample is exactly what the contract types serialize.
// Compared as parsed JSON so the file's formatting is free; on a mismatch the
// message carries the whole current listing to paste into the file.
#[test]
fn should_match_the_committed_model_list_the_opencode_plugin_reads() {
    let serialized_listing =
        serde_json::to_value(fixture_model_list()).expect("the model list should serialize");

    assert!(
        committed_fixture() == serialized_listing,
        "{FIXTURE_RELATIVE_PATH} no longer matches the contract; replace its contents with:\n{}",
        serde_json::to_string_pretty(&serialized_listing).expect("pretty JSON")
    );
}

// Boundary: the sample keeps the chat-only filter meaningful for the plugin,
// with exactly two chat models among models of every other kind.
#[test]
fn should_record_two_chat_models_among_every_other_kind() {
    let fixture = committed_fixture();
    let advertised_models = fixture["data"].as_array().expect("data should be a list");
    let mut chat_model_count = 0;
    for advertised_model in advertised_models {
        if advertised_model["supported_endpoints"]
            .as_array()
            .is_some_and(|endpoints| endpoints.contains(&"/v1/chat/completions".into()))
        {
            chat_model_count += 1;
        }
    }

    assert_eq!(advertised_models.len(), 6);
    assert_eq!(chat_model_count, 2);
}
