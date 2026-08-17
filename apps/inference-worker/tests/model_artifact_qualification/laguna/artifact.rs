use std::collections::BTreeSet;
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

const QUALIFICATION_TIMEOUT: Duration = Duration::from_secs(115);
const QUALIFICATION_POLL_INTERVAL: Duration = Duration::from_millis(100);
const QUALIFICATION_PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
const QUALIFICATION_CHILD_MODEL_ID: &str = "ASTRONOMICAL_LAGUNA_QUALIFICATION_CHILD_MODEL_ID";
const MAXIMUM_QUALIFICATION_SOURCE_CHARACTERS: usize = 4_000;
const MAXIMUM_COMPACT_QUALIFICATION_SOURCE_CHARACTERS: usize = 800;
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

#[derive(Clone, Copy)]
pub(super) struct PinnedLagunaArtifact {
    pub(super) model_id: &'static str,
    pub(super) revision: &'static str,
    expected_layer_count: usize,
    expected_hidden_size: u32,
    expected_shard_count: usize,
    expected_shard_file_bytes: u64,
    // These are pinned qualification facts, not Laguna-wide production constants.
    expected_end_token_ids: &'static [u32],
    expected_template_default_thinking_enabled: bool,
    expected_generation_default_thinking_enabled: bool,
    expected_preserves_prior_reasoning: bool,
    expected_top_k: Option<u16>,
    expected_repetition_penalty_thousandths: u16,
    expected_affine_profiles: &'static [(u32, u32)],
}

impl PinnedLagunaArtifact {
    pub(super) fn public_model_id(self) -> &'static str {
        astronomical_config::leaf_model_id(self.model_id)
    }
}

pub(super) const LAGUNA_XS: PinnedLagunaArtifact = PinnedLagunaArtifact {
    model_id: "mlx-works/Laguna-XS-2.1-oQ2",
    revision: "0ae2dc9bba4130a78ec73ae669cad78a30205af3",
    expected_layer_count: 40,
    expected_hidden_size: 2_048,
    expected_shard_count: 3,
    expected_shard_file_bytes: 11_751_261_776,
    expected_end_token_ids: &[2, 24],
    expected_template_default_thinking_enabled: false,
    expected_generation_default_thinking_enabled: true,
    expected_preserves_prior_reasoning: false,
    expected_top_k: None,
    expected_repetition_penalty_thousandths: 1_000,
    expected_affine_profiles: &[(2, 64), (3, 64), (4, 64), (8, 64), (8, 128)],
};

pub(super) const LAGUNA_S: PinnedLagunaArtifact = PinnedLagunaArtifact {
    model_id: "mlx-community/Laguna-S-2.1-oQ2e-fast",
    revision: "c98939f4fe0918fc670fb41d6627370b44c1d2c7",
    expected_layer_count: 48,
    expected_hidden_size: 3_072,
    expected_shard_count: 7,
    expected_shard_file_bytes: 34_736_172_776,
    expected_end_token_ids: &[2, 24],
    expected_template_default_thinking_enabled: true,
    expected_generation_default_thinking_enabled: true,
    expected_preserves_prior_reasoning: false,
    expected_top_k: Some(20),
    expected_repetition_penalty_thousandths: 1_050,
    expected_affine_profiles: &[(2, 128), (3, 64), (4, 64), (6, 64), (8, 64), (8, 128)],
};

#[test]
#[ignore = "requires the pinned Laguna XS artifact in configured Development model roots"]
fn should_validate_the_pinned_laguna_xs_artifact_from_development_model_roots() {
    run_bounded_qualification(
        LAGUNA_XS,
        "model_artifact_qualification::laguna::artifact::should_validate_the_pinned_laguna_xs_artifact_from_development_model_roots",
    );
}

#[test]
#[ignore = "requires the pinned Laguna S artifact in configured Development model roots"]
fn should_validate_the_pinned_laguna_s_artifact_from_development_model_roots() {
    run_bounded_qualification(
        LAGUNA_S,
        "model_artifact_qualification::laguna::artifact::should_validate_the_pinned_laguna_s_artifact_from_development_model_roots",
    );
}

fn run_bounded_qualification(pinned_artifact: PinnedLagunaArtifact, test_name: &str) {
    if std::env::var(QUALIFICATION_CHILD_MODEL_ID).as_deref() == Ok(pinned_artifact.model_id) {
        qualify_pinned_artifact(pinned_artifact);
        return;
    }
    eprintln!(
        "[laguna-artifact] starting model={} revision={}",
        pinned_artifact.model_id, pinned_artifact.revision
    );
    let test_executable =
        std::env::current_exe().expect("the qualification test executable path should resolve");
    let mut child_process = Command::new(test_executable)
        .args([
            test_name,
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads",
            "1",
        ])
        .env(QUALIFICATION_CHILD_MODEL_ID, pinned_artifact.model_id)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("the isolated Laguna qualification process should start");
    let start_time = Instant::now();
    let mut next_progress_time = QUALIFICATION_PROGRESS_INTERVAL;
    loop {
        if let Some(exit_status) = child_process
            .try_wait()
            .expect("the Laguna qualification process status should be readable")
        {
            assert!(
                exit_status.success(),
                "isolated Laguna artifact qualification failed"
            );
            eprintln!(
                "[laguna-artifact] completed model={}",
                pinned_artifact.model_id
            );
            return;
        }
        let elapsed_time = start_time.elapsed();
        if elapsed_time >= QUALIFICATION_TIMEOUT {
            let _kill_outcome = child_process.kill();
            let _wait_outcome = child_process.wait();
            panic!(
                "Laguna artifact qualification exceeded {} seconds for {}",
                QUALIFICATION_TIMEOUT.as_secs(),
                pinned_artifact.model_id
            );
        }
        if elapsed_time >= next_progress_time {
            eprintln!(
                "[laguna-artifact] validating model={} elapsed_seconds={}",
                pinned_artifact.model_id,
                elapsed_time.as_secs()
            );
            next_progress_time += QUALIFICATION_PROGRESS_INTERVAL;
        }
        thread::sleep(QUALIFICATION_POLL_INTERVAL);
    }
}

fn qualify_pinned_artifact(pinned_artifact: PinnedLagunaArtifact) {
    let model_directory = resolve_pinned_artifact_directory(pinned_artifact);
    assert_eq!(
        classify_model_directory(&model_directory).expect("the pinned config should be readable"),
        Some(ModelFamily::Laguna)
    );
    let development_config =
        astronomical_config::AstronomicalConfig::load_from_development_location()
            .expect("Development configuration should load for executable discovery");
    let publicly_discovered_model = astronomical_config::discover_models(
        development_config.model_directories(),
        development_config.max_output_tokens(),
    )
    .expect("executable model discovery should complete")
    .into_iter()
    .flat_map(|directory_scan| directory_scan.discovered_models)
    .find(|discovered_model| {
        discovered_model.model_family == ModelFamily::Laguna
            && discovered_model.model_id == pinned_artifact.public_model_id()
            && discovered_model.revision == pinned_artifact.revision
    })
    .expect("the pinned Laguna artifact should be publicly discoverable");
    assert_eq!(
        fs::canonicalize(publicly_discovered_model.model_directory)
            .expect("the discovered Laguna path should canonicalize"),
        model_directory
    );
    let validated_artifact = LagunaArtifactValidator::new()
        .validate(&model_directory)
        .expect("the pinned Laguna artifact should validate before model construction");

    assert_eq!(
        validated_artifact.target_contract().model().layer_count(),
        pinned_artifact.expected_layer_count
    );
    assert_eq!(
        validated_artifact.target_contract().model().hidden_size(),
        pinned_artifact.expected_hidden_size
    );
    assert_eq!(
        validated_artifact.shard_index().shard_file_names().count(),
        pinned_artifact.expected_shard_count
    );
    assert_eq!(
        validated_artifact.total_shard_file_bytes(),
        pinned_artifact.expected_shard_file_bytes
    );
    assert!(validated_artifact.total_tensor_payload_bytes() > 0);
    assert!(
        validated_artifact
            .tensor_contract()
            .descriptors()
            .values()
            .all(|descriptor| !descriptor.sources().is_empty()
                && descriptor.sources().iter().all(|source| {
                    source.data_start_offset_bytes() < source.data_end_offset_bytes()
                }))
    );
    let observed_affine_profiles = validated_artifact
        .tensor_contract()
        .descriptors()
        .values()
        .filter_map(|descriptor| match descriptor.storage_encoding() {
            LagunaTensorStorageEncoding::DirectAffine { profile } => {
                Some((profile.bits(), profile.group_size()))
            }
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        observed_affine_profiles,
        pinned_artifact
            .expected_affine_profiles
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
    );

    let text_artifact = validated_artifact.text_artifact();
    assert_eq!(
        text_artifact.end_token_ids(),
        pinned_artifact.expected_end_token_ids
    );
    assert_eq!(text_artifact.reasoning_parser_id(), "poolside_v1");
    assert_eq!(text_artifact.tool_call_parser_id(), "poolside_v1");
    assert_eq!(
        text_artifact.default_thinking_enabled(),
        pinned_artifact.expected_template_default_thinking_enabled
    );
    assert_eq!(
        text_artifact.generation_default_thinking_enabled(),
        Some(pinned_artifact.expected_generation_default_thinking_enabled)
    );
    assert_eq!(
        text_artifact.preserves_prior_reasoning(),
        pinned_artifact.expected_preserves_prior_reasoning
    );
    let artifact_sampler = text_artifact.sampler_config();
    assert!(artifact_sampler.uses_sampling());
    assert_eq!(artifact_sampler.temperature_thousandths(), 1_000);
    assert_eq!(artifact_sampler.top_p_thousandths(), 1_000);
    assert_eq!(artifact_sampler.min_p_thousandths(), 0);
    assert_eq!(artifact_sampler.top_k(), pinned_artifact.expected_top_k);
    assert_eq!(
        artifact_sampler.repetition_penalty_thousandths(),
        pinned_artifact.expected_repetition_penalty_thousandths
    );

    // Preparing from cloned validated text metadata proves CPU-only prompt readiness before
    // ownership of the retained weight and sidecar descriptors is consumed below.
    let generation_processor = LagunaGenerationProcessor::new(
        pinned_artifact.model_id,
        validated_artifact.text_artifact().clone(),
    )
    .expect("the pinned Laguna text artifact should construct a generation processor");
    let source_excerpt = bounded_romeo_and_juliet_source();
    let chat_command = ChatGenerationCommand {
        request_id: RequestId::new(98),
        model: pinned_artifact.model_id.to_owned(),
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
    };
    let prepared_generation = generation_processor
        .prepare_chat(&chat_command)
        .expect("the pinned text-only Romeo and Juliet request should prepare");
    assert!(!prepared_generation.prompt_token_ids().is_empty());
    assert!(
        prepared_generation
            .rendered_prompt()
            .contains(source_excerpt)
    );
    assert!(prepared_generation.thinking_enabled());
    assert!(prepared_generation.generation_starts_in_reasoning());
    assert_eq!(prepared_generation.thinking_budget(), None);
    let effective_sampler = prepared_generation.sampler_config();
    assert!(effective_sampler.uses_sampling());
    assert_eq!(effective_sampler.temperature_thousandths(), 1_000);
    assert_eq!(effective_sampler.top_p_thousandths(), 1_000);
    assert_eq!(effective_sampler.min_p_thousandths(), 0);
    assert_eq!(effective_sampler.top_k(), pinned_artifact.expected_top_k);
    assert_eq!(
        effective_sampler.repetition_penalty_thousandths(),
        pinned_artifact.expected_repetition_penalty_thousandths
    );
    assert_eq!(effective_sampler.seed(), Some(98));

    let retained_files = validated_artifact
        .into_retained_files()
        .expect("validated descriptor ownership should transfer without reopening paths");
    assert_eq!(
        retained_files.shard_files().len(),
        pinned_artifact.expected_shard_count
    );
}

pub(super) fn bounded_romeo_and_juliet_source() -> &'static str {
    romeo_and_juliet_source_with_character_limit(MAXIMUM_QUALIFICATION_SOURCE_CHARACTERS)
}

pub(super) fn compact_romeo_and_juliet_source() -> &'static str {
    romeo_and_juliet_source_with_character_limit(MAXIMUM_COMPACT_QUALIFICATION_SOURCE_CHARACTERS)
}

/// Returns the complete repository fixture for long public prompt journeys.
pub(super) const fn full_romeo_and_juliet_source() -> &'static str {
    ROMEO_AND_JULIET_SOURCE
}

fn romeo_and_juliet_source_with_character_limit(maximum_characters: usize) -> &'static str {
    // A character boundary keeps tokenization representative while bounding qualification work.
    let excerpt_end_byte = ROMEO_AND_JULIET_SOURCE
        .char_indices()
        .nth(maximum_characters)
        .map_or(ROMEO_AND_JULIET_SOURCE.len(), |(byte_index, _character)| {
            byte_index
        });
    &ROMEO_AND_JULIET_SOURCE[..excerpt_end_byte]
}

pub(super) fn resolve_pinned_artifact_directory(pinned_artifact: PinnedLagunaArtifact) -> PathBuf {
    let development_config =
        astronomical_config::AstronomicalConfig::load_from_development_location()
            .expect("Development configuration should load for Laguna qualification");
    let classified_artifacts = astronomical_config::discover_classified_model_artifacts(
        development_config.model_directories(),
    )
    .expect("configured classified-model discovery should complete");
    eprintln!(
        "[laguna-artifact] phase=discovery model={} candidate_count={}",
        pinned_artifact.model_id,
        classified_artifacts.len()
    );
    let matching_directories = classified_artifacts
        .into_iter()
        .filter(|artifact| {
            artifact.model_family == ModelFamily::Laguna
                && artifact.model_id == pinned_artifact.model_id
                && artifact.upstream_revision.as_deref() == Some(pinned_artifact.revision)
        })
        .map(|artifact| {
            fs::canonicalize(artifact.model_directory)
                .expect("the pinned Laguna directory should canonicalize")
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        matching_directories.len(),
        1,
        "configured Development model roots should contain exactly one pinned {} revision {} artifact",
        pinned_artifact.model_id,
        pinned_artifact.revision
    );
    matching_directories
        .into_iter()
        .next()
        .expect("one exact pinned Laguna artifact should remain")
}
