use std::path::PathBuf;

use astronomical_model_serving::Qwen3_5ArtifactValidator;
use tokio::time::timeout;

use super::model_artifact_rest_qualification::{
    E2E_TIMEOUT, image_chat_request_body_for_model, run_model_artifact_request_e2e_for_model,
};

const DENSE_QWEN3_5_VISION_MODEL_DIRECTORY_ENVIRONMENT_VARIABLE: &str =
    "ASTRONOMICAL_DENSE_QWEN3_5_VISION_MODEL_DIRECTORY";

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires ASTRONOMICAL_DENSE_QWEN3_5_VISION_MODEL_DIRECTORY"]
async fn should_stream_image_output_from_a_dense_qwen3_5_vision_model_through_the_openai_endpoint()
{
    let Some(model_directory) = configured_dense_qwen3_5_vision_model_directory() else {
        eprintln!(
            "[dense-qwen3-5-image-e2e] status=skipped reason=required_model_directory_missing"
        );
        return;
    };
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the dense Qwen3.5 vision artifact should validate before the REST smoke test");
    assert!(validated_artifact.config().is_dense_model());
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

fn configured_dense_qwen3_5_vision_model_directory() -> Option<PathBuf> {
    std::env::var_os(DENSE_QWEN3_5_VISION_MODEL_DIRECTORY_ENVIRONMENT_VARIABLE).map(PathBuf::from)
}
