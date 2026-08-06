#[cfg(feature = "direct-mlx")]
mod aligned_expert_pack;
mod artifact;
#[cfg(feature = "direct-mlx")]
mod automatic_residency_endurance;
mod config;
mod engine;
#[cfg(feature = "direct-mlx")]
mod exact_model_prompt;
mod expert_paging;
#[cfg(feature = "direct-mlx")]
mod expert_paging_decode;
#[cfg(feature = "direct-mlx")]
mod expert_paging_prefill;
#[cfg(feature = "direct-mlx")]
mod expert_paging_prefill_performance;
#[cfg(feature = "direct-mlx")]
mod expert_paging_representative_performance;
#[cfg(feature = "direct-mlx")]
mod expert_route_reuse_performance;
#[cfg(feature = "direct-mlx")]
mod expert_weight_memory_cache_eviction;
mod model;
#[cfg(feature = "direct-mlx")]
mod mtp;
#[cfg(feature = "direct-mlx")]
mod paged_mode_endurance;
#[cfg(feature = "direct-mlx")]
mod performance_attribution;
#[cfg(feature = "direct-mlx")]
mod prefill_chunck_qualification;
#[cfg(feature = "direct-mlx")]
mod qwen3_6_35b_a3b_eight_bit_expert_paging;
mod tokenizer;
mod vision_model;
mod weights;

const CERTIFIED_SAY_HI_GREEDY_TOKEN_COUNT: u16 = 10;
const CERTIFIED_SAY_HI_GREEDY_TOKEN_IDS: [u32; CERTIFIED_SAY_HI_GREEDY_TOKEN_COUNT as usize] =
    [12_675, 0, 2_500, 628, 353, 1_438, 488, 3_242, 30, 248_046];
const ORNITH_IMAGE_PAD_TOKEN_ID: u32 = 248_069;

async fn construct_model_artifact_expert_pager(
    progress_log_prefix: &str,
) -> (
    astronomical_runtime_integration::MlxRuntime,
    astronomical_model_serving::Qwen3_5Config,
    astronomical_model_serving::Qwen3_5ExpertPager,
) {
    use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5ExpertPager};
    use astronomical_runtime_integration::MlxRuntime;

    eprintln!("{progress_log_prefix} status=progress phase=artifact_validation");
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the pinned Ornith artifact should validate before expert pager construction");
    eprintln!(
        "{progress_log_prefix} status=progress phase=artifact_validated shards={} payload_bytes={}",
        validated_artifact.shard_count(),
        validated_artifact.total_payload_bytes()
    );

    eprintln!("{progress_log_prefix} status=progress phase=runtime_init");
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let configured_mlx_memory_cap_bytes = mlx_memory_limits.active_memory_limit_bytes();
    let runtime = MlxRuntime::initialize(mlx_memory_limits)
        .expect("the direct MLX runtime should initialize for expert pager construction");

    eprintln!("{progress_log_prefix} status=progress phase=pager_construction");
    let config = validated_artifact.config().clone();
    let weight_map: std::collections::HashMap<String, String> = validated_artifact
        .shard_index()
        .language_tensor_name_to_shard_file_name()
        .iter()
        .map(|(tensor_name, shard_file_name)| (tensor_name.clone(), shard_file_name.clone()))
        .collect();
    let expert_pager = Qwen3_5ExpertPager::new(
        model_directory,
        &weight_map,
        &config,
        configured_mlx_memory_cap_bytes,
        false,
    )
    .expect("Qwen3_5ExpertPager should construct from the Ornith model-artifact directory");

    (runtime, config, expert_pager)
}

fn qwen3_6_35b_a3b_eight_bit_model_directory() -> std::path::PathBuf {
    crate::common::configured_model_artifact_directory_by_id("Qwen3.6-35B-A3B-8bit")
}

fn xyz_aquila_mini_optiq_four_bit_model_directory() -> std::path::PathBuf {
    crate::common::configured_model_artifact_directory_by_id("XYZ-Aquila-mini-OptiQ-4bit")
}

fn qwen3_6_35b_a3b_oq4e_mtp_model_directory() -> std::path::PathBuf {
    crate::common::configured_model_artifact_directory_by_id("Qwen3.6-35B-A3B-oQ4e-mtp")
}
