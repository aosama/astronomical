use std::collections::HashMap;

use astronomical_ipc_protocol::{
    WorkerAutoregressiveModelConfiguration, WorkerChunkingConfiguration, WorkerModelConfiguration,
};
use astronomical_supervisor::{RuntimeModelGenerationDefaults, RuntimeModelPolicy};

use super::*;

const PRIMARY_MODEL_ID: &str = "astronomical/application-test-model";
const SECONDARY_MODEL_ID: &str = "organization/secondary-model";

#[tokio::test]
async fn should_apply_each_models_defaults_to_omitted_chat_settings() {
    let (application, received_generation_commands) = application_with_model_defaults([
        (PRIMARY_MODEL_ID, generation_defaults(4_000, 700, 900)),
        (SECONDARY_MODEL_ID, generation_defaults(6_000, 200, 800)),
    ]);

    for model_id in [PRIMARY_MODEL_ID, SECONDARY_MODEL_ID] {
        let response = post_json(
            &application,
            "/v1/chat/completions",
            format!(
                r#"{{"model":"{model_id}","messages":[{{"role":"user","content":"hello"}}],"stream":true}}"#
            ),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    let received_generation_commands = received_generation_commands
        .lock()
        .expect("the scripted executor command log should be readable");
    assert_eq!(received_generation_commands.len(), 2);
    assert_eq!(
        received_generation_commands[0].settings.max_output_tokens,
        4_000
    );
    assert_eq!(
        received_generation_commands[0]
            .settings
            .temperature_thousandths,
        Some(700)
    );
    assert_eq!(
        received_generation_commands[0].settings.top_p_thousandths,
        Some(900)
    );
    assert_eq!(
        received_generation_commands[1].settings.max_output_tokens,
        6_000
    );
    assert_eq!(
        received_generation_commands[1]
            .settings
            .temperature_thousandths,
        Some(200)
    );
    assert_eq!(
        received_generation_commands[1].settings.top_p_thousandths,
        Some(800)
    );
}

#[tokio::test]
async fn should_keep_explicit_chat_settings_above_the_models_output_default() {
    let (application, received_generation_commands) =
        application_with_model_defaults([(PRIMARY_MODEL_ID, generation_defaults(128, 700, 900))]);

    let response = post_json(
        &application,
        "/v1/chat/completions",
        format!(
            r#"{{"model":"{PRIMARY_MODEL_ID}","messages":[{{"role":"user","content":"hello"}}],"max_tokens":20000,"temperature":0.3,"top_p":0.6,"stream":true}}"#
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let received_generation_commands = received_generation_commands
        .lock()
        .expect("the scripted executor command log should be readable");
    assert_eq!(
        received_generation_commands[0].settings.max_output_tokens,
        20_000
    );
    assert_eq!(
        received_generation_commands[0]
            .settings
            .temperature_thousandths,
        Some(300)
    );
    assert_eq!(
        received_generation_commands[0].settings.top_p_thousandths,
        Some(600)
    );
}

#[tokio::test]
async fn should_apply_each_models_defaults_to_omitted_responses_settings() {
    let (application, received_generation_commands) = application_with_model_defaults([
        (PRIMARY_MODEL_ID, generation_defaults(4_000, 700, 900)),
        (SECONDARY_MODEL_ID, generation_defaults(6_000, 200, 800)),
    ]);

    for model_id in [PRIMARY_MODEL_ID, SECONDARY_MODEL_ID] {
        let response = post_json(
            &application,
            "/v1/responses",
            format!(r#"{{"model":"{model_id}","input":"hello","stream":true}}"#),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    let received_generation_commands = received_generation_commands
        .lock()
        .expect("the scripted executor command log should be readable");
    assert_eq!(received_generation_commands.len(), 2);
    assert_eq!(
        received_generation_commands[0].settings.max_output_tokens,
        4_000
    );
    assert_eq!(
        received_generation_commands[0]
            .settings
            .temperature_thousandths,
        Some(700)
    );
    assert_eq!(
        received_generation_commands[0].settings.top_p_thousandths,
        Some(900)
    );
    assert_eq!(
        received_generation_commands[1].settings.max_output_tokens,
        6_000
    );
    assert_eq!(
        received_generation_commands[1]
            .settings
            .temperature_thousandths,
        Some(200)
    );
    assert_eq!(
        received_generation_commands[1].settings.top_p_thousandths,
        Some(800)
    );
}

#[tokio::test]
async fn should_keep_explicit_responses_settings_above_the_models_output_default() {
    let (application, received_generation_commands) =
        application_with_model_defaults([(PRIMARY_MODEL_ID, generation_defaults(128, 700, 900))]);

    let response = post_json(
        &application,
        "/v1/responses",
        format!(
            r#"{{"model":"{PRIMARY_MODEL_ID}","input":"hello","max_output_tokens":20000,"temperature":0.3,"top_p":0.6,"stream":true}}"#
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let received_generation_commands = received_generation_commands
        .lock()
        .expect("the scripted executor command log should be readable");
    assert_eq!(
        received_generation_commands[0].settings.max_output_tokens,
        20_000
    );
    assert_eq!(
        received_generation_commands[0]
            .settings
            .temperature_thousandths,
        Some(300)
    );
    assert_eq!(
        received_generation_commands[0].settings.top_p_thousandths,
        Some(600)
    );
}

fn application_with_model_defaults<const MODEL_COUNT: usize>(
    model_defaults: [(&str, RuntimeModelGenerationDefaults); MODEL_COUNT],
) -> (
    axum::Router,
    Arc<std::sync::Mutex<Vec<astronomical_ipc_protocol::ChatGenerationCommand>>>,
) {
    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    let mut resolved_config = sample_resolved_config();
    resolved_config.discovered_models = model_defaults
        .iter()
        .map(|(model_id, _)| discovered_model(model_id))
        .collect();
    resolved_config.model_policy_catalog = Arc::new(
        model_defaults
            .into_iter()
            .map(|(model_id, generation_defaults)| {
                (
                    model_id.to_owned(),
                    runtime_model_policy(model_id, generation_defaults),
                )
            })
            .collect::<HashMap<_, _>>(),
    );
    let reloadable_config = Arc::new(RwLock::new(resolved_config));
    let executor = ScriptedExecutor::ready(Vec::new());
    let received_generation_commands = executor.received_generation_commands();
    let application = build_development_application_with_reload(
        executor,
        reloadable_config,
        config_home_directory,
    );
    (application, received_generation_commands)
}

fn generation_defaults(
    maximum_output_tokens: u16,
    temperature_thousandths: u16,
    top_p_thousandths: u16,
) -> RuntimeModelGenerationDefaults {
    RuntimeModelGenerationDefaults {
        maximum_output_tokens,
        configured_maximum_output_tokens: Some(maximum_output_tokens),
        temperature_thousandths: Some(temperature_thousandths),
        top_p_thousandths: Some(top_p_thousandths),
    }
}

fn runtime_model_policy(
    model_id: &str,
    generation_defaults: RuntimeModelGenerationDefaults,
) -> RuntimeModelPolicy {
    RuntimeModelPolicy {
        model_directory: PathBuf::from(format!("/fictional/models/{model_id}")),
        generation_defaults,
        configured_maximum_context_tokens: None,
        default_maximum_context_tokens: 262_144,
        configured_chunking_fields: Default::default(),
        acceleration_availability: Default::default(),
        worker_model_configuration: WorkerModelConfiguration::Autoregressive(
            WorkerAutoregressiveModelConfiguration {
                model_id: model_id.to_owned(),
                maximum_context_tokens: 262_144,
                maximum_output_tokens: u32::from(u16::MAX),
                chunking: WorkerChunkingConfiguration {
                    fixed_prompt_processing_chunk_size_tokens: 2_048,
                    fixed_ssd_streaming_prompt_processing_chunk_size_tokens: None,
                    full_attention_key_value_growth_tokens: 256,
                    speculative_prefill_draft_forward_tokens: 2_048,
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

fn discovered_model(model_id: &str) -> astronomical_config::DiscoveredModel {
    astronomical_config::DiscoveredModel {
        model_id: model_id.to_owned(),
        provider_model_id: None,
        model_family: astronomical_config::ModelFamily::Qwen3_5,
        revision: "test-revision".to_owned(),
        model_directory: PathBuf::from(format!("/fictional/models/{model_id}")),
        capabilities: astronomical_config::ModelCapabilities::Chat(
            astronomical_config::ChatModelCapabilities {
                context_window: 262_144,
                max_input_tokens: 262_143,
                max_output_tokens: u32::from(u16::MAX),
                supports_vision: false,
                supports_reasoning: true,
                supports_tool_calls: true,
            },
        ),
        license: None,
        model_size_bytes: 1,
    }
}

async fn post_json(
    application: &axum::Router,
    uri: &str,
    body: String,
) -> axum::response::Response {
    application
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body))
                .expect("the generation request should be valid"),
        )
        .await
        .expect("the application should return a response")
}
