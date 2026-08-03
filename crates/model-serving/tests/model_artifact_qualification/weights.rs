use astronomical_model_serving::{Qwen3_5MoEArtifactValidator, Qwen3_5MoEWeights};
use astronomical_runtime_integration::MlxRuntime;

#[tokio::test]
#[ignore = "requires model_directories to discover Ornith-1.0-35B-OptiQ-4bit"]
async fn should_bind_only_resident_language_tensors_for_automatic_sparse_expert_paging() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    let validated_artifact = Qwen3_5MoEArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the pinned Ornith artifact should validate before paged native loading");
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let runtime = MlxRuntime::initialize(mlx_memory_limits)
        .expect("the direct MLX runtime should initialize");

    let ornith_resident_weights = Qwen3_5MoEWeights::load(&runtime, validated_artifact)
        .expect("resident Ornith tensors should bind without sparse selected experts");

    assert_eq!(ornith_resident_weights.shard_count(), 5);
    assert_eq!(ornith_resident_weights.tensor_count(), 1_397);
    assert_eq!(ornith_resident_weights.decoder_layer_count(), 40);
    assert_eq!(ornith_resident_weights.total_payload_bytes(), 2_568_911_104);
}
