//! Process-boundary acceptance tests for transactional worker replacement.

use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationCompletionReason, ChatGenerationSettings, ChatMessage,
    ChatToolChoice, RequestId, WorkerAutoregressiveModelConfiguration, WorkerChunkingConfiguration,
    WorkerFlux2KleinModelConfiguration, WorkerImageGenerationModelFamily, WorkerLogLevel,
    WorkerModelConfiguration, WorkerStartupConfiguration,
};
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, GenerationPerformanceLog,
    RuntimeModelGenerationDefaults, RuntimeModelPolicy, WorkerHandle, WorkerHealthSnapshot,
    WorkerHealthStatus,
};
use tokio::time::{Instant, sleep};

const INITIAL_GENERATION: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const CANDIDATE_GENERATION: &str =
    "2222222222222222222222222222222222222222222222222222222222222222";
const PROTOCOL_INVALID_GENERATION: &str =
    "3333333333333333333333333333333333333333333333333333333333333333";
const CONFIGURATION_BEFORE_READY_GENERATION: &str =
    "4444444444444444444444444444444444444444444444444444444444444444";
const INCONSISTENT_READY_GENERATION: &str =
    "5555555555555555555555555555555555555555555555555555555555555555";
const MATCHING_FLUX_GENERATION: &str =
    "6666666666666666666666666666666666666666666666666666666666666666";
const TRUSTED_MODEL_ID: &str = "astronomical/trusted-model";
const CANDIDATE_MODEL_ID: &str = "astronomical/candidate-model";
const FLUX_MODEL_ID: &str = "FLUX.2-klein-4B";

#[tokio::test]
async fn should_replace_the_trusted_worker_only_after_exact_candidate_acknowledgement() {
    let test_context = ReplacementTestContext::launch(Duration::from_secs(2)).await;
    let candidate_catalog = model_policy_catalog(CANDIDATE_MODEL_ID);

    let acknowledged_configuration = test_context
        .worker_handle
        .restart_worker_with_startup_configuration(
            fixture_path("astronomical-supervisor-idle-worker"),
            candidate_catalog,
            startup_configuration(CANDIDATE_GENERATION, &test_context.temporary_directory),
        )
        .await
        .expect("the fully acknowledged candidate should replace the trusted worker");

    assert_eq!(
        acknowledged_configuration.configuration_generation,
        CANDIDATE_GENERATION
    );
    let candidate_health = test_context.worker_handle.worker_health_snapshot();
    assert_eq!(candidate_health.status, WorkerHealthStatus::Ready);
    assert_eq!(
        candidate_health
            .worker_runtime_feature_configuration
            .as_ref()
            .map(|configuration| configuration.configuration_generation.as_str()),
        Some(CANDIDATE_GENERATION)
    );
    assert_generation_succeeds(&test_context.worker_handle, CANDIDATE_MODEL_ID).await;
    test_context.shutdown().await;
}

#[tokio::test]
async fn should_accept_candidate_configuration_before_initial_readiness() {
    let test_context = ReplacementTestContext::launch(Duration::from_secs(2)).await;

    let acknowledged_configuration = test_context
        .worker_handle
        .restart_worker_with_startup_configuration(
            fixture_path("astronomical-supervisor-replacement-ready-worker"),
            model_policy_catalog(CANDIDATE_MODEL_ID),
            startup_configuration(
                CONFIGURATION_BEFORE_READY_GENERATION,
                &test_context.temporary_directory,
            ),
        )
        .await
        .expect("configuration may precede candidate readiness");

    assert_eq!(
        acknowledged_configuration.configuration_generation,
        CONFIGURATION_BEFORE_READY_GENERATION
    );
    assert_eq!(
        test_context.worker_handle.worker_health_snapshot().status,
        WorkerHealthStatus::Ready
    );
    test_context.shutdown().await;
}

#[tokio::test]
async fn should_match_the_exact_tagged_flux_runtime_configuration_during_replacement() {
    let test_context = ReplacementTestContext::launch(Duration::from_secs(2)).await;

    let acknowledged_configuration = test_context
        .worker_handle
        .restart_worker_with_startup_configuration(
            fixture_path("astronomical-supervisor-replacement-ready-worker"),
            flux_model_policy_catalog("reviewed-revision"),
            startup_configuration(MATCHING_FLUX_GENERATION, &test_context.temporary_directory),
        )
        .await
        .expect("the exact tagged FLUX policy should match candidate acknowledgement");

    assert_eq!(
        acknowledged_configuration.configuration_generation,
        MATCHING_FLUX_GENERATION
    );
    assert_eq!(
        test_context
            .worker_handle
            .worker_health_snapshot()
            .ready_model_id
            .as_deref(),
        Some(FLUX_MODEL_ID)
    );
    test_context.shutdown().await;
}

#[tokio::test]
async fn should_reject_flux_replacement_when_the_acknowledged_revision_differs() {
    let test_context = ReplacementTestContext::launch(Duration::from_secs(2)).await;
    let trusted_health = test_context.worker_handle.worker_health_snapshot();

    let replacement_error = test_context
        .worker_handle
        .restart_worker_with_startup_configuration(
            fixture_path("astronomical-supervisor-replacement-ready-worker"),
            flux_model_policy_catalog("different-revision"),
            startup_configuration(MATCHING_FLUX_GENERATION, &test_context.temporary_directory),
        )
        .await
        .expect_err("replacement must compare the exact tagged runtime configuration");

    assert!(
        replacement_error
            .to_string()
            .contains("disagrees with its acknowledged policy"),
        "unexpected replacement error: {replacement_error}"
    );
    assert_trusted_worker_unchanged(&test_context.worker_handle, &trusted_health);
    test_context.shutdown().await;
}

#[tokio::test]
async fn should_keep_the_trusted_worker_when_candidate_launch_fails() {
    let test_context = ReplacementTestContext::launch(Duration::from_secs(2)).await;
    let trusted_health = test_context.worker_handle.worker_health_snapshot();

    let replacement_error = test_context
        .worker_handle
        .restart_worker_with_startup_configuration(
            test_context.temporary_directory.join("missing-worker"),
            model_policy_catalog(CANDIDATE_MODEL_ID),
            startup_configuration(CANDIDATE_GENERATION, &test_context.temporary_directory),
        )
        .await
        .expect_err("a missing candidate executable must fail replacement");

    assert!(
        replacement_error
            .to_string()
            .contains("failed to start worker process")
    );
    assert_trusted_worker_unchanged(&test_context.worker_handle, &trusted_health);
    assert_generation_succeeds(&test_context.worker_handle, TRUSTED_MODEL_ID).await;
    test_context.shutdown().await;
}

#[tokio::test]
async fn should_reject_ready_candidate_without_matching_loaded_model_acknowledgement() {
    let test_context = ReplacementTestContext::launch(Duration::from_secs(2)).await;
    let trusted_health = test_context.worker_handle.worker_health_snapshot();

    let replacement_error = test_context
        .worker_handle
        .restart_worker_with_startup_configuration(
            fixture_path("astronomical-supervisor-replacement-ready-worker"),
            model_policy_catalog(CANDIDATE_MODEL_ID),
            startup_configuration(
                INCONSISTENT_READY_GENERATION,
                &test_context.temporary_directory,
            ),
        )
        .await
        .expect_err("ready candidates must acknowledge the exact loaded model policy");

    assert!(
        replacement_error
            .to_string()
            .contains("loaded model policy")
    );
    assert_trusted_worker_unchanged(&test_context.worker_handle, &trusted_health);
    assert_generation_succeeds(&test_context.worker_handle, TRUSTED_MODEL_ID).await;
    test_context.shutdown().await;
}

#[tokio::test]
async fn should_reap_mismatched_candidate_and_keep_trusted_worker_serving() {
    let test_context = ReplacementTestContext::launch(Duration::from_secs(2)).await;
    let trusted_health = test_context.worker_handle.worker_health_snapshot();

    let replacement_error = test_context
        .worker_handle
        .restart_worker_with_startup_configuration(
            fixture_path("astronomical-supervisor-replacement-ready-worker"),
            model_policy_catalog(CANDIDATE_MODEL_ID),
            startup_configuration(CANDIDATE_GENERATION, &test_context.temporary_directory),
        )
        .await
        .expect_err("a mismatched candidate generation must fail replacement");

    assert!(
        replacement_error
            .to_string()
            .contains("different configuration generation")
    );
    assert_trusted_worker_unchanged(&test_context.worker_handle, &trusted_health);
    assert_candidate_was_reaped(&test_context.temporary_directory);
    assert_generation_succeeds(&test_context.worker_handle, TRUSTED_MODEL_ID).await;
    test_context.shutdown().await;
}

#[tokio::test]
async fn should_keep_the_trusted_worker_when_candidate_readiness_times_out() {
    let test_context = ReplacementTestContext::launch(Duration::from_millis(500)).await;
    let trusted_health = test_context.worker_handle.worker_health_snapshot();

    let replacement_error = test_context
        .worker_handle
        .restart_worker_with_startup_configuration(
            fixture_path("astronomical-supervisor-loading-forever-worker"),
            model_policy_catalog(CANDIDATE_MODEL_ID),
            startup_configuration(CANDIDATE_GENERATION, &test_context.temporary_directory),
        )
        .await
        .expect_err("an unready candidate must time out");

    assert!(replacement_error.to_string().contains("candidate"));
    assert_trusted_worker_unchanged(&test_context.worker_handle, &trusted_health);
    assert_generation_succeeds(&test_context.worker_handle, TRUSTED_MODEL_ID).await;
    test_context.shutdown().await;
}

#[tokio::test]
async fn should_keep_the_trusted_worker_when_candidate_exits_before_acknowledgement() {
    let test_context = ReplacementTestContext::launch(Duration::from_secs(2)).await;
    let trusted_health = test_context.worker_handle.worker_health_snapshot();

    let replacement_error = test_context
        .worker_handle
        .restart_worker_with_startup_configuration(
            fixture_path("astronomical-supervisor-mismatched-ready-worker"),
            model_policy_catalog(CANDIDATE_MODEL_ID),
            startup_configuration(CANDIDATE_GENERATION, &test_context.temporary_directory),
        )
        .await
        .expect_err("a candidate that exits before acknowledgement must fail replacement");

    assert!(
        replacement_error
            .to_string()
            .contains("worker process exited")
    );
    assert_trusted_worker_unchanged(&test_context.worker_handle, &trusted_health);
    assert_generation_succeeds(&test_context.worker_handle, TRUSTED_MODEL_ID).await;
    test_context.shutdown().await;
}

#[tokio::test]
async fn should_reject_generation_scoped_candidate_events_and_keep_trusted_worker() {
    let test_context = ReplacementTestContext::launch(Duration::from_secs(2)).await;
    let trusted_health = test_context.worker_handle.worker_health_snapshot();

    let replacement_error = test_context
        .worker_handle
        .restart_worker_with_startup_configuration(
            fixture_path("astronomical-supervisor-replacement-ready-worker"),
            model_policy_catalog(CANDIDATE_MODEL_ID),
            startup_configuration(
                PROTOCOL_INVALID_GENERATION,
                &test_context.temporary_directory,
            ),
        )
        .await
        .expect_err("generation-scoped candidate startup events must be rejected");

    assert!(
        replacement_error
            .to_string()
            .contains("invalid startup event")
    );
    assert_trusted_worker_unchanged(&test_context.worker_handle, &trusted_health);
    assert_generation_succeeds(&test_context.worker_handle, TRUSTED_MODEL_ID).await;
    test_context.shutdown().await;
}

struct ReplacementTestContext {
    temporary_directory: PathBuf,
    worker_handle: WorkerHandle,
}

impl ReplacementTestContext {
    async fn launch(model_load_timeout: Duration) -> Self {
        let temporary_directory = tempfile::tempdir()
            .expect("the replacement test directory should be created")
            .keep();
        let worker_handle = WorkerHandle::launch_with_startup_configuration(
            fixture_path("astronomical-supervisor-idle-worker"),
            model_load_timeout,
            GenerationPerformanceLog::open(&temporary_directory)
                .expect("the performance log should open"),
            model_policy_catalog(TRUSTED_MODEL_ID),
            startup_configuration(INITIAL_GENERATION, &temporary_directory),
        )
        .await
        .expect("the trusted worker should launch");
        wait_for_effective_generation(&worker_handle, INITIAL_GENERATION).await;
        Self {
            temporary_directory,
            worker_handle,
        }
    }

    async fn shutdown(self) {
        self.worker_handle
            .shutdown()
            .await
            .expect("the retained worker should shut down");
    }
}

fn assert_trusted_worker_unchanged(
    worker_handle: &WorkerHandle,
    trusted_health: &WorkerHealthSnapshot,
) {
    assert_eq!(&worker_handle.worker_health_snapshot(), trusted_health);
}

async fn wait_for_effective_generation(worker_handle: &WorkerHandle, expected_generation: &str) {
    let readiness_deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let health_snapshot = worker_handle.worker_health_snapshot();
        if health_snapshot.status == WorkerHealthStatus::Ready
            && health_snapshot
                .worker_runtime_feature_configuration
                .as_ref()
                .is_some_and(|configuration| {
                    configuration.configuration_generation == expected_generation
                })
        {
            return;
        }
        assert!(
            Instant::now() < readiness_deadline,
            "worker acknowledgement timed out"
        );
        sleep(Duration::from_millis(10)).await;
    }
}

async fn assert_generation_succeeds(worker_handle: &WorkerHandle, model_id: &str) {
    let mut generation_events = worker_handle
        .start_chat_generation(chat_command(model_id))
        .await
        .expect("the trusted model policy should remain routable");
    assert!(matches!(
        generation_events.recv().await,
        Some(ChatGenerationStreamEvent::Completed {
            reason: ChatGenerationCompletionReason::EndOfSequence,
            ..
        })
    ));
}

fn assert_candidate_was_reaped(temporary_directory: &std::path::Path) {
    let process_id_text =
        std::fs::read_to_string(temporary_directory.join("replacement-candidate.pid"))
            .expect("the candidate fixture should record its process identifier");
    let process_id = process_id_text.trim();
    let process_probe = std::process::Command::new("kill")
        .args(["-0", process_id])
        .output()
        .expect("the process probe should run");
    assert!(
        !process_probe.status.success(),
        "the rejected candidate must be reaped"
    );
}

fn fixture_path(fixture_name: &str) -> PathBuf {
    PathBuf::from(
        std::env::var(format!("CARGO_BIN_EXE_{fixture_name}"))
            .unwrap_or_else(|_| panic!("Cargo should provide the {fixture_name} fixture path")),
    )
}

fn startup_configuration(
    configuration_generation: &str,
    temporary_directory: &std::path::Path,
) -> WorkerStartupConfiguration {
    WorkerStartupConfiguration {
        configuration_generation: configuration_generation.to_owned(),
        global_prompt_cache_root_directory: temporary_directory.join("cache"),
        global_prompt_cache_maximum_size_bytes: 1_000_000_000,
        persistent_prompt_cache_enabled: true,
        configured_maximum_mlx_memory_bytes: None,
        performance_attribution_enabled: true,
        logging_directory: temporary_directory.to_path_buf(),
        logging_level: WorkerLogLevel::Warn,
        retained_log_file_count: 1,
    }
}

fn model_policy_catalog(model_id: &str) -> Arc<HashMap<String, RuntimeModelPolicy>> {
    Arc::new(HashMap::from([(
        model_id.to_owned(),
        RuntimeModelPolicy {
            model_directory: PathBuf::from(format!("/fictional/models/{model_id}")),
            generation_defaults: RuntimeModelGenerationDefaults {
                maximum_output_tokens: 128,
                configured_maximum_output_tokens: None,
                temperature_thousandths: None,
                top_p_thousandths: None,
            },
            configured_maximum_context_tokens: None,
            default_maximum_context_tokens: 2_048,
            configured_chunking_fields: Default::default(),
            acceleration_availability: Default::default(),
            worker_model_configuration: WorkerModelConfiguration::Autoregressive(
                WorkerAutoregressiveModelConfiguration {
                    model_id: model_id.to_owned(),
                    maximum_context_tokens: 2_048,
                    maximum_output_tokens: 128,
                    chunking: WorkerChunkingConfiguration {
                        fixed_prompt_processing_chunk_size_tokens: 256,
                        fixed_ssd_streaming_prompt_processing_chunk_size_tokens: None,
                        full_attention_key_value_growth_tokens: 256,
                        speculative_prefill_draft_forward_tokens: 256,
                        prefill_graph_submission_layer_interval: 1,
                        experimental_ssd_paging_generation_graph_submission_layer_interval: 3,
                        prompt_cache_block_tokens: None,
                        prompt_cache_common_prefix_stride_blocks: 4,
                    },
                    mtp_draft_depth: None,
                    speculative_prefill: None,
                },
            ),
        },
    )]))
}

fn flux_model_policy_catalog(revision: &str) -> Arc<HashMap<String, RuntimeModelPolicy>> {
    Arc::new(HashMap::from([(
        FLUX_MODEL_ID.to_owned(),
        RuntimeModelPolicy {
            model_directory: PathBuf::from("/fictional/models/FLUX.2-klein-4B"),
            generation_defaults: RuntimeModelGenerationDefaults {
                maximum_output_tokens: 0,
                configured_maximum_output_tokens: None,
                temperature_thousandths: None,
                top_p_thousandths: None,
            },
            configured_maximum_context_tokens: None,
            default_maximum_context_tokens: 0,
            configured_chunking_fields: Default::default(),
            acceleration_availability: Default::default(),
            worker_model_configuration: WorkerModelConfiguration::Flux2Klein(
                WorkerFlux2KleinModelConfiguration {
                    model_id: FLUX_MODEL_ID.to_owned(),
                    model_family: WorkerImageGenerationModelFamily::Flux2Klein,
                    artifact_revision: revision.to_owned(),
                },
            ),
        },
    )]))
}

fn chat_command(model_id: &str) -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id: RequestId::new(1),
        model: model_id.to_owned(),
        messages: vec![ChatMessage::User {
            content: include_str!(
                "../../../inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
            )
            .to_owned(),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 1,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
    }
}
