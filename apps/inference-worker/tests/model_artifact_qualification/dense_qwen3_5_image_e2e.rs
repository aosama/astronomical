use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5FeedForwardArchitecture};
use tokio::time::timeout;

use super::model_artifact_rest_qualification::{
    E2E_TIMEOUT, image_chat_request_body_for_model, run_model_artifact_request_e2e_for_model,
};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the smallest configured dense Qwen3.5 vision artifact through REST"]
async fn should_stream_image_output_from_a_dense_qwen3_5_vision_model_through_the_openai_endpoint()
{
    let (model_directory, validated_artifact) = configured_dense_qwen3_5_vision_artifact()
        .expect("configured model roots should contain a dense Qwen3.5 vision artifact");
    assert_eq!(
        validated_artifact.config().feed_forward_architecture(),
        Qwen3_5FeedForwardArchitecture::Dense
    );
    assert!(validated_artifact.supports_image_input());
    let dense_qwen3_5_vision_model_id = validated_artifact.model_id().to_owned();

    timeout(
        E2E_TIMEOUT,
        run_model_artifact_request_e2e_for_model(
            &dense_qwen3_5_vision_model_id,
            model_directory,
            "image",
            image_chat_request_body_for_model(&dense_qwen3_5_vision_model_id),
        ),
    )
    .await
    .expect("the dense Qwen3.5 image E2E test must finish within 115 seconds");
}

fn configured_dense_qwen3_5_vision_artifact() -> Option<(
    std::path::PathBuf,
    astronomical_model_serving::ValidatedQwen3_5Artifact,
)> {
    let mut discovered_vision_models = crate::common::configured_discovered_models()
        .into_iter()
        .filter(|discovered_model| discovered_model.has_vision)
        .collect::<Vec<_>>();
    discovered_vision_models.sort_by_key(|discovered_model| discovered_model.model_size_bytes);

    discovered_vision_models
        .into_iter()
        .find_map(|discovered_vision_model| {
            let validated_artifact = Qwen3_5ArtifactValidator::new()
                .validate(
                    &discovered_vision_model.model_directory,
                    discovered_vision_model.max_output_tokens,
                )
                .ok()?;
            (validated_artifact.config().feed_forward_architecture()
                == Qwen3_5FeedForwardArchitecture::Dense
                && validated_artifact.supports_image_input())
            .then_some((discovered_vision_model.model_directory, validated_artifact))
        })
}
