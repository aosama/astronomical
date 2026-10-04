//! Artifact-validation contract for the configured `dense_mtp` model.
//!
//! The canonical dense multi-token-prediction artifact must pass the Qwen3.5
//! artifact validator unchanged: it is `model_type qwen3_5` with a bundled
//! multi-token sidecar and a separate vision sidecar, so this journey is the
//! guard that the validator accepts that layout as published. Validation only
//! reads metadata and the shard index; it loads no weights into GPU memory.

use std::time::Duration;

use astronomical_model_serving::Qwen3_5ArtifactValidator;
use tokio::time::timeout;

use crate::common::{configured_model_directory_by_id, dense_mtp_model_id};

const DENSE_MTP_VALIDATION_TIMEOUT: Duration = Duration::from_secs(60);
const DENSE_MTP_VALIDATION_MAXIMUM_OUTPUT_TOKENS: u32 = 2_048;

#[tokio::test]
async fn should_validate_the_configured_dense_mtp_artifact() {
    timeout(
        DENSE_MTP_VALIDATION_TIMEOUT,
        validate_configured_dense_mtp_artifact(),
    )
    .await
    .expect("the dense-MTP validation contract must finish within its timeout");
}

async fn validate_configured_dense_mtp_artifact() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory =
        configured_model_directory_by_id(dense_mtp_model_id()).unwrap_or_else(|| {
            panic!(
                "the configured model directories should discover the dense-MTP model {}",
                dense_mtp_model_id()
            )
        });

    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, DENSE_MTP_VALIDATION_MAXIMUM_OUTPUT_TOKENS)
        .expect("the configured dense-MTP artifact should validate");

    assert_eq!(
        validated_artifact.model_id(),
        dense_mtp_model_id(),
        "the validated artifact's leaf identity must match the dense_mtp role"
    );
}
