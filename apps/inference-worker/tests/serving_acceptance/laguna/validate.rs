//! Structural validity tests for Laguna artifacts.
//!
//! Tests prove the normalization pipeline remains valid across packaging variants
//! of the Laguna XS e2e fixture, which is resolved by public leaf
//! id through Development model directories. Tests fail fast if the model is missing.
//!
//! Assertions validate structural properties derived from the artifact's own
//! config and index, not golden-master constants that break when swapping
//! quantization packaging.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use astronomical_config::{ModelFamily, classify_model_directory};
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, ChatToolDefinition,
    RequestId,
};
use astronomical_model_serving::{
    LagunaArtifactValidator, LagunaGenerationProcessor, LagunaTensorStorageEncoding,
};

pub(crate) const ACCEPTANCE_TIMEOUT: Duration = Duration::from_secs(115);
pub(crate) const ACCEPTANCE_POLL_INTERVAL: Duration = Duration::from_millis(100);
pub(crate) const ACCEPTANCE_PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
const ACCEPTANCE_CHILD_MODEL_ID: &str = "ASTRONOMICAL_LAGUNA_ACCEPTANCE_CHILD_MODEL_ID";
const MAXIMUM_ACCEPTANCE_SOURCE_CHARACTERS: usize = 4_000;
const MAXIMUM_COMPACT_ACCEPTANCE_SOURCE_CHARACTERS: usize = 800;
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

/// Public leaf id of the locally available Laguna XS reference model.
/// Tests fail fast if this model is not present in Development model roots.
pub(crate) fn laguna_xs_public_model_id() -> &'static str {
    crate::support::laguna_xs_model_id()
}

/// Valid affine bit widths that MLX supports for direct affine quantization.
const VALID_AFFINE_BITS: &[u32] = &[2, 3, 4, 5, 6, 8];

/// Valid affine group sizes that MLX supports.
const VALID_AFFINE_GROUP_SIZES: &[u32] = &[32, 64, 128];

#[test]
#[ignore = "requires the reference Laguna XS artifact in configured Development model roots"]
fn should_validate_installed_laguna_xs() {
    run_bounded_acceptance(
        "serving_acceptance::laguna::validate::should_validate_installed_laguna_xs",
    );
}

fn run_bounded_acceptance(test_name: &str) {
    let laguna_xs_model_id = laguna_xs_public_model_id();
    if std::env::var(ACCEPTANCE_CHILD_MODEL_ID).as_deref() == Ok(laguna_xs_model_id) {
        validate_reference_artifact();
        return;
    }
    eprintln!("[laguna-artifact] starting model={laguna_xs_model_id}");
    let test_executable =
        std::env::current_exe().expect("the acceptance test executable path should resolve");
    let mut child_process = Command::new(test_executable)
        .args([
            test_name,
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads",
            "1",
        ])
        .env(ACCEPTANCE_CHILD_MODEL_ID, laguna_xs_model_id)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("the isolated Laguna acceptance process should start");
    let start_time = Instant::now();
    let mut next_progress_time = ACCEPTANCE_PROGRESS_INTERVAL;
    loop {
        if let Some(exit_status) = child_process
            .try_wait()
            .expect("the Laguna acceptance process status should be readable")
        {
            assert!(
                exit_status.success(),
                "isolated Laguna artifact acceptance failed"
            );
            eprintln!("[laguna-artifact] completed model={laguna_xs_model_id}");
            return;
        }
        let elapsed_time = start_time.elapsed();
        if elapsed_time >= ACCEPTANCE_TIMEOUT {
            let _kill_outcome = child_process.kill();
            let _wait_outcome = child_process.wait();
            panic!(
                "Laguna artifact acceptance exceeded {} seconds for {laguna_xs_model_id}",
                ACCEPTANCE_TIMEOUT.as_secs(),
            );
        }
        if elapsed_time >= next_progress_time {
            eprintln!(
                "[laguna-artifact] validating model={laguna_xs_model_id} elapsed_seconds={}",
                elapsed_time.as_secs()
            );
            next_progress_time += ACCEPTANCE_PROGRESS_INTERVAL;
        }
        thread::sleep(ACCEPTANCE_POLL_INTERVAL);
    }
}

fn validate_reference_artifact() {
    let model_directory = resolve_reference_model_directory();
    assert_eq!(
        classify_model_directory(&model_directory)
            .expect("the reference config should be readable"),
        Some(ModelFamily::Laguna)
    );
    let publicly_discovered_model = crate::support::configured_discovered_models()
        .into_iter()
        .find(|discovered_model| {
            discovered_model.model_family == ModelFamily::Laguna
                && discovered_model.model_id == laguna_xs_public_model_id()
        })
        .expect("the reference Laguna artifact should be publicly discoverable");
    assert_eq!(
        fs::canonicalize(publicly_discovered_model.model_directory)
            .expect("the discovered Laguna path should canonicalize"),
        model_directory
    );
    let validated_artifact = LagunaArtifactValidator::new()
        .validate(&model_directory)
        .expect("the reference Laguna artifact should validate before model construction");

    assert_structurally_valid_laguna_artifact(&validated_artifact);

    // CPU-only prompt readiness: prepare a Romeo and Juliet chat command and
    // verify nonempty prompt tokens and that the rendered prompt contains the excerpt.
    let generation_processor = LagunaGenerationProcessor::new(
        laguna_xs_public_model_id(),
        validated_artifact.text_artifact().clone(),
    )
    .expect("the reference Laguna text artifact should construct a generation processor");
    let source_excerpt = bounded_romeo_and_juliet_source();
    let chat_command = ChatGenerationCommand {
        request_id: RequestId::new(98),
        model: laguna_xs_public_model_id().to_owned(),
        messages: vec![ChatMessage::User {
            content: format!(
                "Use the supplied Romeo and Juliet source for literary analysis.\n\n{source_excerpt}"
            ),
            images: Vec::new(),
        }],
        tools: vec![ChatToolDefinition {
            name: "record_literary_theme".to_owned(),
            description: Some("Record one theme supported by the supplied play.".to_owned()),
            parameters_json: r#"{"type":"object","properties":{"theme":{"type":"string"}},"required":["theme"],"additionalProperties":false}"#
                .to_owned(),
        }],
        tool_choice: ChatToolChoice::Auto,
        settings: ChatGenerationSettings {
            max_output_tokens: 128,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: Some(98),
            thinking_budget: None,
        },
        qwen_thinking_channel_seed: None,
    };
    let prepared_generation = generation_processor
        .prepare_chat(&chat_command)
        .expect("the reference text-only Romeo and Juliet request should prepare");
    assert!(!prepared_generation.prompt_token_ids().is_empty());
    assert!(
        prepared_generation
            .rendered_prompt()
            .contains(source_excerpt)
    );
    assert!(prepared_generation.thinking_enabled());
    assert!(prepared_generation.generation_starts_in_reasoning());
    assert_eq!(prepared_generation.thinking_budget(), None);

    let validated_shard_count = validated_artifact.shard_index().shard_file_names().count();
    let validated_shard_file_bytes = validated_artifact.total_shard_file_bytes();
    let retained_files = validated_artifact
        .into_retained_files()
        .expect("validated descriptor ownership should transfer without reopening paths");
    assert_eq!(retained_files.shard_files().len(), validated_shard_count);
    assert_eq!(
        retained_files
            .shard_files()
            .values()
            .map(astronomical_model_serving::ValidatedWeightsFile::size_bytes)
            .sum::<u64>(),
        validated_shard_file_bytes,
        "the aggregate shard bytes must equal the retained descriptor sizes"
    );
}

/// Validates structural properties derived from the artifact's own config and
/// index, not golden-master constants tied to one packaging variant.
fn assert_structurally_valid_laguna_artifact(
    validated: &astronomical_model_serving::ValidatedLagunaArtifact,
) {
    let layer_count = validated.target_contract().model().layer_count();
    let hidden_size = validated.target_contract().model().hidden_size();
    assert!(
        layer_count > 0,
        "layer count must be positive, got {layer_count}"
    );
    assert!(
        hidden_size > 0,
        "hidden size must be positive, got {hidden_size}"
    );

    let shard_count = validated.shard_index().shard_file_names().count();
    assert!(
        shard_count > 0,
        "shard count must be positive, got {shard_count}"
    );
    assert!(
        validated.total_shard_file_bytes() > 0,
        "total shard file bytes must be positive"
    );
    assert!(
        validated.total_tensor_payload_bytes() > 0,
        "total tensor payload bytes must be positive"
    );

    assert!(
        validated
            .tensor_contract()
            .descriptors()
            .values()
            .all(|descriptor| !descriptor.sources().is_empty()
                && descriptor.sources().iter().all(|source| {
                    source.data_start_offset_bytes() < source.data_end_offset_bytes()
                })),
        "every tensor descriptor must have nonempty sources with valid byte intervals"
    );

    // Packaging variants can change the observed profile set, but every profile
    // still has to remain executable by MLX.
    let affine_profiles: std::collections::BTreeSet<(u32, u32)> = validated
        .tensor_contract()
        .descriptors()
        .values()
        .filter_map(|descriptor| match descriptor.storage_encoding() {
            LagunaTensorStorageEncoding::DirectAffine { profile } => {
                Some((profile.bits(), profile.group_size()))
            }
            _ => None,
        })
        .collect();
    for (bits, group_size) in &affine_profiles {
        assert!(
            VALID_AFFINE_BITS.contains(bits),
            "affine bits must be a valid MLX width, got {bits}"
        );
        assert!(
            VALID_AFFINE_GROUP_SIZES.contains(group_size),
            "affine group size must be valid, got {group_size}"
        );
    }

    let text_artifact = validated.text_artifact();
    assert!(
        !text_artifact.end_token_ids().is_empty(),
        "end token ids must be nonempty"
    );
    assert_eq!(text_artifact.reasoning_parser_id(), "poolside_v1");
    assert_eq!(text_artifact.tool_call_parser_id(), "poolside_v1");

    let artifact_sampler = text_artifact.sampler_config();
    assert!(artifact_sampler.uses_sampling());
    assert!(
        artifact_sampler.temperature_thousandths() > 0
            && artifact_sampler.temperature_thousandths() <= 2_000,
        "temperature must be in (0, 2000]‰, got {}",
        artifact_sampler.temperature_thousandths()
    );
    assert!(
        artifact_sampler.top_p_thousandths() > 0 && artifact_sampler.top_p_thousandths() <= 1_000,
        "top_p must be in (0, 1000]‰, got {}",
        artifact_sampler.top_p_thousandths()
    );
    assert!(
        artifact_sampler.min_p_thousandths() <= 1_000,
        "min_p must be in [0, 1000]‰, got {}",
        artifact_sampler.min_p_thousandths()
    );

    assert!(
        text_artifact
            .generation_default_thinking_enabled()
            .is_some(),
        "generation default thinking enabled must be present"
    );
}

pub(crate) fn bounded_romeo_and_juliet_source() -> &'static str {
    romeo_and_juliet_source_with_character_limit(MAXIMUM_ACCEPTANCE_SOURCE_CHARACTERS)
}

pub(crate) fn compact_romeo_and_juliet_source() -> &'static str {
    romeo_and_juliet_source_with_character_limit(MAXIMUM_COMPACT_ACCEPTANCE_SOURCE_CHARACTERS)
}

/// Returns the complete repository fixture for long public prompt journeys.
pub(crate) const fn full_romeo_and_juliet_source() -> &'static str {
    ROMEO_AND_JULIET_SOURCE
}

fn romeo_and_juliet_source_with_character_limit(maximum_characters: usize) -> &'static str {
    // A character boundary keeps tokenization representative while bounding acceptance work.
    let excerpt_end_byte = ROMEO_AND_JULIET_SOURCE
        .char_indices()
        .nth(maximum_characters)
        .map_or(ROMEO_AND_JULIET_SOURCE.len(), |(byte_index, _character)| {
            byte_index
        });
    &ROMEO_AND_JULIET_SOURCE[..excerpt_end_byte]
}

/// Resolves the reference Laguna XS model directory through Development model
/// discovery by public leaf id. Panics (fail-fast) if the model is not found.
pub(crate) fn resolve_reference_model_directory() -> PathBuf {
    crate::support::configured_installed_model_directory_by_id(laguna_xs_public_model_id())
}
