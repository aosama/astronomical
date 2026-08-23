//! Qualifies resident Qwen weight binding against the artifact's own config
//! and shard index rather than one quantization package's exact counts.

use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5Weights};
use astronomical_runtime_integration::MlxRuntime;

#[tokio::test]
#[ignore = "requires model_directories to discover the Ornith 1.5 qualification artifact"]
async fn should_bind_only_resident_language_tensors_for_automatic_sparse_expert_paging() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the reference Ornith artifact should validate before paged native loading");
    let expected_decoder_layer_count = validated_artifact.config().layer_count() as usize;
    let expected_model_shard_count = validated_artifact
        .shard_index()
        .model_shard_file_names()
        .len();
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let runtime = MlxRuntime::initialize(mlx_memory_limits)
        .expect("the direct MLX runtime should initialize");

    let ornith_resident_weights = Qwen3_5Weights::load(&runtime, validated_artifact)
        .expect("resident Ornith tensors should bind without sparse selected experts");

    assert_eq!(
        ornith_resident_weights.shard_count(),
        expected_model_shard_count
    );
    assert!(ornith_resident_weights.tensor_count() > 0);
    assert_eq!(
        ornith_resident_weights.decoder_layer_count(),
        expected_decoder_layer_count,
        "bound decoder layers must match the validated model config"
    );
    assert!(ornith_resident_weights.total_payload_bytes() > 0);
}
