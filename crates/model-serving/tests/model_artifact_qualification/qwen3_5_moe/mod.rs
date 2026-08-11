#[cfg(feature = "direct-mlx")]
mod aligned_expert_pack;
mod artifact;
#[cfg(feature = "direct-mlx")]
mod automatic_residency;
#[cfg(feature = "direct-mlx")]
mod automatic_residency_failure;
#[cfg(feature = "direct-mlx")]
mod automatic_residency_mtp;
#[cfg(feature = "direct-mlx")]
mod automatic_residency_support;
mod config;
mod engine;
#[cfg(feature = "direct-mlx")]
mod exact_model_prompt;
#[cfg(feature = "direct-mlx")]
mod exact_paged_decode;
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
mod expert_paging_romeo_and_juliet_performance;
#[cfg(feature = "direct-mlx")]
mod expert_route_reuse_performance;
#[cfg(feature = "direct-mlx")]
mod expert_weight_memory_cache_eviction;
mod model;
#[cfg(feature = "direct-mlx")]
mod mtp;
#[cfg(feature = "direct-mlx")]
mod one_expert_cache_endurance;
#[cfg(feature = "direct-mlx")]
mod paged_mode_endurance;
#[cfg(feature = "direct-mlx")]
mod performance_attribution;
#[cfg(feature = "direct-mlx")]
mod prefill_chunck_qualification;
#[cfg(feature = "direct-mlx")]
mod qwen3_6_35b_a3b_eight_bit_expert_paging;
#[cfg(feature = "direct-mlx")]
mod speculative_prefill;
#[cfg(feature = "direct-mlx")]
mod speculative_prefill_fail_closed;
#[cfg(feature = "direct-mlx")]
mod speculative_prefill_full_retention;
#[cfg(feature = "direct-mlx")]
mod speculative_prefill_memory_admission;
#[cfg(feature = "direct-mlx")]
mod speculative_prefill_persistent_cache;
mod speculative_prefill_qualification_support;
#[cfg(feature = "direct-mlx")]
mod speculative_prefill_tool_control;
#[cfg(feature = "direct-mlx")]
mod speculative_prefill_tool_false_positive;
mod speculative_prefill_tool_process_prompt;
#[cfg(feature = "direct-mlx")]
mod speculative_prefill_tool_process_restart;
#[cfg(feature = "direct-mlx")]
mod speculative_prefill_tool_sampled;
#[cfg(feature = "direct-mlx")]
mod speculative_prefill_visual_memory_lifecycle;
#[cfg(feature = "direct-mlx")]
mod speculative_prefill_visual_tool;
mod tokenizer;
mod vision_model;
mod weights;

use std::path::PathBuf;

const CERTIFIED_SAY_HI_GREEDY_TOKEN_COUNT: u16 = 10;
const CERTIFIED_SAY_HI_GREEDY_TOKEN_IDS: [u32; CERTIFIED_SAY_HI_GREEDY_TOKEN_COUNT as usize] =
    [12_675, 0, 2_500, 628, 353, 1_438, 488, 3_242, 30, 248_046];
const ORNITH_IMAGE_PAD_TOKEN_ID: u32 = 248_069;
const MAXIMUM_SPECULATIVE_PREFILL_DRAFT_PAYLOAD_BYTES: u64 = 3_000_000_000;

#[test]
fn should_select_a_small_compatible_speculative_prefill_draft_model() {
    let selected_draft_model = select_smallest_compatible_speculative_prefill_draft_model(
        vec![
            (
                30_124_710_752,
                "Qwen3.6-35B-A3B-oQ6-mtp".to_owned(),
                PathBuf::from("oversized-qwen-draft"),
            ),
            (
                3_034_147_328,
                "Qwen3.5-4B-MLX-4bit".to_owned(),
                PathBuf::from("large-qwen-draft"),
            ),
            (
                1_722_149_056,
                "Qwen3.5-2B-4bit".to_owned(),
                PathBuf::from("two-billion-qwen-draft"),
            ),
            (
                650_168_512,
                "Qwen3.5-0.8B-OptiQ-4bit".to_owned(),
                PathBuf::from("small-qwen-draft"),
            ),
        ],
        3_000_000_000,
    );

    assert_eq!(
        selected_draft_model,
        Some((
            650_168_512,
            "Qwen3.5-0.8B-OptiQ-4bit".to_owned(),
            PathBuf::from("small-qwen-draft"),
        )),
    );
}

fn select_smallest_compatible_speculative_prefill_draft_model(
    draft_model_candidates: impl IntoIterator<Item = (u64, String, PathBuf)>,
    maximum_draft_payload_bytes: u64,
) -> Option<(u64, String, PathBuf)> {
    draft_model_candidates
        .into_iter()
        .filter(|draft_model_candidate| draft_model_candidate.0 <= maximum_draft_payload_bytes)
        .min_by(|left_candidate, right_candidate| {
            left_candidate
                .0
                .cmp(&right_candidate.0)
                .then_with(|| left_candidate.1.cmp(&right_candidate.1))
        })
}

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
        &runtime,
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

fn configured_depth_one_mtp_model_artifact_directory() -> std::path::PathBuf {
    // Preserve deterministic model-identity selection for existing qualifications.
    configured_depth_one_mtp_model_artifacts()
        .into_iter()
        .min_by(|left_candidate, right_candidate| left_candidate.1.cmp(&right_candidate.1))
        .map(|(_payload_bytes, _model_id, model_directory)| model_directory)
        .expect(
            "model_directories should contain a complete supported depth-one mixture-of-experts MTP artifact",
        )
}

fn configured_smallest_depth_one_mtp_model_artifact_directory() -> std::path::PathBuf {
    // Residency qualifications materialize every expert, so choose the smallest
    // compatible payload to keep the real-artifact journey within its timeout.
    configured_depth_one_mtp_model_artifacts()
        .into_iter()
        .min_by(|left_candidate, right_candidate| {
            left_candidate
                .0
                .cmp(&right_candidate.0)
                .then_with(|| left_candidate.1.cmp(&right_candidate.1))
        })
        .map(|(_payload_bytes, _model_id, model_directory)| model_directory)
        .expect(
            "model_directories should contain a complete supported depth-one mixture-of-experts MTP artifact",
        )
}

fn configured_depth_one_mtp_model_artifacts() -> Vec<(u64, String, std::path::PathBuf)> {
    use astronomical_config::{AstronomicalConfig, discover_models};
    use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5FeedForwardArchitecture};

    let astronomical_config = AstronomicalConfig::load_from_default_location()
        .expect("the standard Astronomical configuration should load for MTP qualification");
    let discovered_models = discover_models(
        astronomical_config.model_directories(),
        astronomical_config.max_output_tokens(),
    )
    .expect("configured model discovery should complete for MTP qualification")
    .into_iter()
    .flat_map(|model_directory_scan| model_directory_scan.discovered_models)
    .collect::<Vec<_>>();
    discovered_models
        .into_iter()
        .filter_map(|discovered_model| {
            let validated_artifact = Qwen3_5ArtifactValidator::new()
                .validate(&discovered_model.model_directory, 20_480)
                .ok()?;
            (validated_artifact.config().feed_forward_architecture()
                == Qwen3_5FeedForwardArchitecture::MixtureOfExperts
                && validated_artifact.config().mtp_layer_count() == 1
                && validated_artifact
                    .mtp_artifact_capability()
                    .is_mtp_capable())
            .then_some((
                validated_artifact.total_payload_bytes(),
                discovered_model.model_id,
                discovered_model.model_directory,
            ))
        })
        .collect()
}

#[cfg(feature = "direct-mlx")]
fn configured_speculative_prefill_draft_model_artifact(
    target_model_directory: &std::path::Path,
) -> (std::path::PathBuf, String) {
    use astronomical_config::{AstronomicalConfig, discover_models};
    use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5Tokenizer};

    let astronomical_config = AstronomicalConfig::load_from_default_location().expect(
        "the standard Astronomical configuration should load for SpecPrefill qualification",
    );
    let target_artifact = Qwen3_5ArtifactValidator::new()
        .validate(
            target_model_directory,
            astronomical_config.max_output_tokens(),
        )
        .expect("the configured SpecPrefill target should validate");
    let target_token_identifier_mapping_digest = Qwen3_5Tokenizer::token_identifier_mapping_digest(
        target_artifact
            .tokenizer_bytes()
            .expect("the configured SpecPrefill target should retain tokenizer bytes"),
    )
    .expect("the configured SpecPrefill target tokenizer mapping should digest");
    let maximum_draft_payload_bytes = (target_artifact.total_payload_bytes() / 10)
        .min(MAXIMUM_SPECULATIVE_PREFILL_DRAFT_PAYLOAD_BYTES);
    let draft_model_candidates = discover_models(
        astronomical_config.model_directories(),
        astronomical_config.max_output_tokens(),
    )
    .expect("configured model discovery should complete for SpecPrefill qualification")
    .into_iter()
    .flat_map(|model_directory_scan| model_directory_scan.discovered_models)
    .filter_map(|discovered_model| {
        if discovered_model.model_directory == target_model_directory {
            return None;
        }
        let draft_artifact = match Qwen3_5ArtifactValidator::new().validate(
                &discovered_model.model_directory,
                astronomical_config.max_output_tokens(),
            ) {
                Ok(draft_artifact) => draft_artifact,
                Err(artifact_validation_error) => {
                    eprintln!(
                        "[speculative-prefill-artifact] status=progress phase=draft_rejected draft_model_id={} reason=validation error={artifact_validation_error:?}",
                        discovered_model.model_id,
                    );
                    return None;
                }
            };
        let tokenizer_matches = draft_artifact.tokenizer_bytes().is_some_and(|tokenizer_bytes| {
            Qwen3_5Tokenizer::token_identifier_mapping_digest(tokenizer_bytes)
                .is_ok_and(|draft_token_identifier_mapping_digest| {
                    draft_token_identifier_mapping_digest
                        == target_token_identifier_mapping_digest
                })
        });
        let vocabulary_matches = draft_artifact.config().vocabulary_size()
            == target_artifact.config().vocabulary_size();
        let payload_fits = draft_artifact.total_payload_bytes() <= maximum_draft_payload_bytes;
        eprintln!(
            "[speculative-prefill-artifact] status=progress phase=draft_evaluated draft_model_id={} payload_bytes={} maximum_payload_bytes={} tokenizer_matches={} vocabulary_matches={} payload_fits={}",
            discovered_model.model_id,
            draft_artifact.total_payload_bytes(),
            maximum_draft_payload_bytes,
            tokenizer_matches,
            vocabulary_matches,
            payload_fits,
        );
        (tokenizer_matches && vocabulary_matches).then(|| {
            (
                draft_artifact.total_payload_bytes(),
                discovered_model.model_id,
                discovered_model.model_directory,
            )
        })
    })
    .collect::<Vec<_>>();
    select_smallest_compatible_speculative_prefill_draft_model(
        draft_model_candidates,
        maximum_draft_payload_bytes,
    )
    .map(|(draft_payload_bytes, draft_model_id, draft_model_directory)| {
        eprintln!(
            "[speculative-prefill-artifact] status=progress phase=draft_selected draft_model_id={} payload_bytes={} maximum_payload_bytes={}",
            draft_model_id,
            draft_payload_bytes,
            maximum_draft_payload_bytes,
        );
        (draft_model_directory, draft_model_id)
    })
    .expect("SpecPrefill qualification requires a compatible Qwen draft no larger than the configured target-relative limit and the three-billion-byte hard cap")
}
