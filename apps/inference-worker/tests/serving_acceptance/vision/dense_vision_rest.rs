use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5FeedForwardArchitecture};
use tokio::time::timeout;

use crate::serving_acceptance::chat::openai_rest::{
    E2E_TIMEOUT, assert_successful_streaming_chat_response, image_chat_request_body_for_model,
    run_serving_chat_request_for_model,
};
use crate::support::http::streamed_model_text_from_chat_response;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the dense vision e2e fixture through REST and checks the synthetic-red fixture"]
async fn should_name_the_red_fixture_through_chat_completions() {
    let (model_directory, validated_artifact) = configured_dense_qwen3_5_vision_artifact()
        .expect("configured model roots should contain a dense vision artifact");
    assert_eq!(
        validated_artifact.config().feed_forward_architecture(),
        Qwen3_5FeedForwardArchitecture::Dense
    );
    assert!(validated_artifact.supports_image_input());
    let dense_vision_model_id = validated_artifact.model_id().to_owned();

    let chat_response = timeout(
        E2E_TIMEOUT,
        run_serving_chat_request_for_model(
            &dense_vision_model_id,
            model_directory,
            "synthetic-red-image",
            image_chat_request_body_for_model(&dense_vision_model_id),
        ),
    )
    .await
    .expect("the dense vision REST journey must finish within 115 seconds");

    assert_successful_streaming_chat_response(&chat_response);
    let streamed_model_text = streamed_model_text_from_chat_response(&chat_response);
    let matched_red_term = assert_streamed_model_text_mentions_red(&streamed_model_text);
    eprintln!("[e2e] dense-vision synthetic-red semantic match term={matched_red_term}");
}

fn assert_streamed_model_text_mentions_red(streamed_model_text: &str) -> &'static str {
    let normalized_model_text = streamed_model_text.to_lowercase();
    for red_term in ["red", "crimson", "scarlet"] {
        if normalized_model_text.contains(red_term) {
            return red_term;
        }
    }
    panic!("model output did not identify the synthetic image as red: {streamed_model_text:?}");
}

fn configured_dense_qwen3_5_vision_artifact() -> Option<(
    std::path::PathBuf,
    astronomical_model_serving::ValidatedQwen3_5Artifact,
)> {
    let dense_mtp_model_id = crate::support::dense_mtp_model_id();
    let discovered_model = crate::support::configured_discovered_models()
        .into_iter()
        .find(|discovered_model| discovered_model.model_id == dense_mtp_model_id)?;
    let maximum_output_tokens =
        crate::support::chat_capabilities(&discovered_model)?.max_output_tokens;
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&discovered_model.model_directory, maximum_output_tokens)
        .ok()?;
    (validated_artifact.config().feed_forward_architecture()
        == Qwen3_5FeedForwardArchitecture::Dense
        && validated_artifact.supports_image_input())
    .then_some((discovered_model.model_directory, validated_artifact))
}
