use std::collections::HashMap;

use astronomical_model_serving::{ExpertPager, Qwen3_5ArtifactValidator};
use astronomical_runtime_integration::MlxRuntime;

#[tokio::test]
#[ignore = "loads a standard safetensors expert page beside existing experimental pack files"]
async fn should_ignore_existing_experimental_aligned_packs_during_standard_expert_paging() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = super::qwen3_6_35b_a3b_oq4e_mtp_model_directory();
    assert!(
        model_directory
            .join(".astronomical-aligned-expert-packs")
            .is_dir(),
        "the qualification requires existing experimental files beside the production model"
    );

    eprintln!("[standard-expert-paging] status=progress phase=artifact_validation");
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the production model artifact should validate");
    let qwen3_5_config = validated_artifact.config().clone();
    let language_tensor_name_to_shard_file_name = validated_artifact
        .shard_index()
        .language_tensor_name_to_shard_file_name()
        .iter()
        .map(|(tensor_name, shard_file_name)| (tensor_name.clone(), shard_file_name.clone()))
        .collect::<HashMap<_, _>>();
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let runtime = MlxRuntime::initialize(mlx_memory_limits)
        .expect("the direct MLX runtime should initialize for standard expert paging");
    let expert_pager = ExpertPager::new(
        model_directory,
        &language_tensor_name_to_shard_file_name,
        &qwen3_5_config,
        mlx_memory_limits.active_memory_limit_bytes(),
        false,
    )
    .expect("the production pager should construct without inspecting experimental files");

    eprintln!("[standard-expert-paging] status=progress phase=bounded_safetensors_page_load");
    let (_paged_expert_weights, page_manifest, _memory_budget_snapshot) = expert_pager
        .load_selected_experts(&runtime, 0, &[0])
        .expect("standard bounded safetensors should load one expert beside experimental files");

    assert_eq!(page_manifest.expert_ids, vec![0]);
    assert!(page_manifest.payload_byte_count > 0);
    eprintln!("[standard-expert-paging] status=success source=standard_safetensors");
}
