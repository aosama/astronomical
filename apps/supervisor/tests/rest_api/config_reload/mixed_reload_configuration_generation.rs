//! Regression journey for the effective generation produced by a mixed live reload.

use std::collections::HashMap;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationCompletionReason, ChatGenerationSettings, ChatMessage,
    ChatToolChoice, RequestId, WorkerAutoregressiveModelConfiguration, WorkerChunkingConfiguration,
    WorkerModelConfiguration,
};
use astronomical_supervisor::{
    ChatGenerationStreamEvent, RuntimeModelGenerationDefaults, RuntimeModelPolicy,
};

use super::*;

const MODEL_ID: &str = "astronomical/mixed-memory-reload-model";

#[tokio::test]
async fn should_load_a_model_after_a_mixed_reload_applies_only_the_memory_configuration_generation()
{
    timeout(Duration::from_secs(10), async {
        let worker_executable_path = PathBuf::from(
            std::env::var("CARGO_BIN_EXE_astronomical-supervisor-idle-worker")
                .expect("Cargo should provide the idle worker fixture path"),
        );
        let config_home =
            tempfile::tempdir().expect("a mixed-reload config home should be created");
        let config_home_directory = config_home.path().to_path_buf();
        write_config_file(
            &config_home_directory,
            r#"{
                "runtime": {
                    "model_directories": [],
                    "maximum_mlx_memory_gb": 32
                },
                "diagnostics": { "log_level": "info" }
            }"#,
        );
        let performance_log_directory = config_home_directory.join("performance");
        std::fs::create_dir_all(&performance_log_directory)
            .expect("the mixed-reload performance log directory should be created");
        let mut initial_resolved_config = sample_resolved_config();
        initial_resolved_config.worker_executable_path = worker_executable_path.clone();
        let model_policy_catalog = Arc::new(HashMap::from([(
            MODEL_ID.to_owned(),
            mixed_reload_model_policy(&config_home_directory),
        )]));
        let worker_handle = WorkerHandle::launch_with_startup_configuration(
            &worker_executable_path,
            Duration::from_secs(2),
            GenerationPerformanceLog::open(&performance_log_directory)
                .expect("the mixed-reload performance log should open"),
            model_policy_catalog,
            initial_resolved_config.worker_startup_configuration(),
        )
        .await
        .expect("the idle worker should launch for the mixed reload journey");
        wait_for_mixed_reload_worker_configuration(&worker_handle).await;
        let runtime_config_resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
            config_home_directory.clone(),
            worker_executable_path,
        );
        let application = build_application_with_full_control(
            worker_handle.clone(),
            Arc::new(RwLock::new(initial_resolved_config)),
            runtime_config_resolver,
            ShutdownController::new(),
        );

        let reload_response = post_config_reload(&application).await;
        assert_eq!(reload_response.status(), StatusCode::OK);
        let mut generation_events = worker_handle
            .start_chat_generation(mixed_reload_generation_command())
            .await
            .expect("the model swap should accept the memory-only effective generation");
        let generation_event = generation_events
            .recv()
            .await
            .expect("the mixed-reload generation should complete");
        assert!(matches!(
            generation_event,
            ChatGenerationStreamEvent::Completed {
                reason: ChatGenerationCompletionReason::EndOfSequence,
                ..
            }
        ));

        worker_handle
            .shutdown()
            .await
            .expect("the mixed-reload worker should shut down");
    })
    .await
    .expect("the mixed memory and logging reload journey should finish within ten seconds");
}

fn mixed_reload_model_policy(model_root: &std::path::Path) -> RuntimeModelPolicy {
    RuntimeModelPolicy {
        model_directory: model_root.join("mixed-memory-reload-model"),
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
                model_id: MODEL_ID.to_owned(),
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
    }
}

async fn wait_for_mixed_reload_worker_configuration(worker_handle: &WorkerHandle) {
    let readiness_deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let worker_health_snapshot = worker_handle.worker_health_snapshot();
        if worker_health_snapshot.status == WorkerHealthStatus::Ready
            && worker_health_snapshot
                .worker_runtime_feature_configuration
                .is_some()
        {
            return;
        }
        assert!(
            Instant::now() < readiness_deadline,
            "the mixed-reload worker should acknowledge startup configuration"
        );
        sleep(Duration::from_millis(10)).await;
    }
}

fn mixed_reload_generation_command() -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id: RequestId::new(9_002),
        model: MODEL_ID.to_owned(),
        messages: vec![ChatMessage::User {
            content: "Wherefore art thou Romeo?".to_owned(),
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
