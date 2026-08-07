use std::collections::HashMap;

use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5ExpertPager};
use astronomical_runtime_integration::MlxRuntime;

#[tokio::test]
#[ignore = "loads a standard safetensors expert page beside existing experimental pack files"]
async fn should_ignore_existing_experimental_aligned_packs_during_standard_expert_paging() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = configured_model_artifact_directory_with_experimental_aligned_packs();

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
    let expert_pager = Qwen3_5ExpertPager::new(
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

fn configured_model_artifact_directory_with_experimental_aligned_packs() -> std::path::PathBuf {
    use astronomical_config::{AstronomicalConfig, discover_models};

    let astronomical_config = AstronomicalConfig::load_from_default_location()
        .expect("the standard Astronomical configuration should load");
    discover_models(
        astronomical_config.model_directories(),
        astronomical_config.max_output_tokens(),
    )
    .expect("configured model discovery should complete")
    .into_iter()
    .flat_map(|model_directory_scan| model_directory_scan.discovered_models)
    .map(|discovered_model| discovered_model.model_directory)
    .find(|model_directory| {
        model_directory
            .join(".astronomical-aligned-expert-packs")
            .is_dir()
    })
    .expect("a configured model should contain experimental aligned expert packs")
}
