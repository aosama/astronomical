//! User-journey coverage for the config-gated Qwen thinking-channel seed.

use super::*;

const TEST_MODEL_ID: &str = "astronomical/application-test-model";
const ROMEO_AND_JULIET_THINKING_SEED: &str =
    "Two households, both alike in dignity, in fair Verona.";

#[tokio::test]
async fn should_guard_both_chat_surfaces_with_the_experimental_thinking_seed_flag() {
    for (thinking_channel_seed_enabled, model_family, expected_seed) in [
        (false, astronomical_config::ModelFamily::Qwen3_5, None),
        (
            true,
            astronomical_config::ModelFamily::Qwen3_5,
            Some(ROMEO_AND_JULIET_THINKING_SEED),
        ),
        (true, astronomical_config::ModelFamily::Laguna, None),
    ] {
        let test_context =
            ThinkingSeedTestContext::new(thinking_channel_seed_enabled, model_family);

        for (endpoint, request_body) in [
            (
                "/v1/chat/completions",
                format!(
                    r#"{{"model":"{TEST_MODEL_ID}","messages":[{{"role":"user","content":"Romeo and Juliet"}}],"stream":true}}"#
                ),
            ),
            (
                "/v1/responses",
                format!(
                    r#"{{"model":"{TEST_MODEL_ID}","input":"Romeo and Juliet","stream":true}}"#
                ),
            ),
        ] {
            let response = post_json(&test_context.application, endpoint, request_body).await;
            assert_eq!(response.status(), StatusCode::OK);
        }

        let received_generation_commands = test_context
            .received_generation_commands
            .lock()
            .expect("the scripted executor command log should be readable");
        assert_eq!(received_generation_commands.len(), 2);
        for generation_command in received_generation_commands.iter() {
            assert_eq!(
                generation_command.qwen_thinking_channel_seed.as_deref(),
                expected_seed
            );
        }
    }
}

struct ThinkingSeedTestContext {
    application: axum::Router,
    received_generation_commands:
        Arc<std::sync::Mutex<Vec<astronomical_ipc_protocol::ChatGenerationCommand>>>,
    _config_home_directory: tempfile::TempDir,
}

impl ThinkingSeedTestContext {
    fn new(
        thinking_channel_seed_enabled: bool,
        model_family: astronomical_config::ModelFamily,
    ) -> Self {
        let config_home_directory = tempfile::tempdir().expect("a config home should be created");
        let instance_state_directory = config_home_directory.path().join(".astronomical-dev");
        std::fs::create_dir_all(&instance_state_directory)
            .expect("the Development instance state should be created");
        std::fs::write(
            instance_state_directory.join("thinking.md"),
            format!("{ROMEO_AND_JULIET_THINKING_SEED}\n"),
        )
        .expect("the thinking seed should be written");

        let mut resolved_config = sample_resolved_config();
        resolved_config.discovered_models = vec![discovered_model(model_family)];
        resolved_config.experimental_qwen_thinking_channel_seed_enabled =
            thinking_channel_seed_enabled;
        let executor = ScriptedExecutor::ready(Vec::new());
        let received_generation_commands = executor.received_generation_commands();
        let application = build_development_application_with_reload(
            executor,
            Arc::new(RwLock::new(resolved_config)),
            config_home_directory.path().to_path_buf(),
        );
        Self {
            application,
            received_generation_commands,
            _config_home_directory: config_home_directory,
        }
    }
}

fn discovered_model(
    model_family: astronomical_config::ModelFamily,
) -> astronomical_config::DiscoveredModel {
    astronomical_config::DiscoveredModel {
        model_id: TEST_MODEL_ID.to_owned(),
        provider_model_id: None,
        model_family,
        revision: "test-revision".to_owned(),
        model_directory: PathBuf::from("/fictional/models/application-test-model"),
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
    endpoint: &str,
    request_body: String,
) -> axum::response::Response {
    application
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(endpoint)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(request_body))
                .expect("the request should be valid"),
        )
        .await
        .expect("the application should return a response")
}
