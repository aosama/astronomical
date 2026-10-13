use astronomical_config::DiscoveredModel;
use astronomical_model_serving::Qwen3_5ArtifactValidator;

pub(crate) fn configured_deployment_litmus_model() -> DiscoveredModel {
    let small_dense_model_id = crate::support::small_dense_model_id();
    let selected_model = crate::support::configured_discovered_models()
        .into_iter()
        .find(|discovered_model| discovered_model.model_id == small_dense_model_id)
        .unwrap_or_else(|| {
            panic!(
                "Development model_directories must discover the small_dense e2e fixture {small_dense_model_id}"
            )
        });
    let chat_capabilities =
        crate::support::chat_capabilities(&selected_model).unwrap_or_else(|| {
            panic!("the small_dense e2e fixture {small_dense_model_id} must be a chat model")
        });
    Qwen3_5ArtifactValidator::new()
        .validate(
            &selected_model.model_directory,
            chat_capabilities.max_output_tokens,
        )
        .unwrap_or_else(|artifact_validation_error| {
            panic!(
                "the small_dense e2e fixture {small_dense_model_id} must validate: {artifact_validation_error}"
            )
        });
    eprintln!(
        "[deployment-litmus] selected model={} size_bytes={}",
        selected_model.model_id, selected_model.model_size_bytes
    );
    selected_model
}

fn sort_deployment_litmus_models(discovered_models: &mut [DiscoveredModel]) {
    discovered_models.sort_by(|candidate_model, other_model| {
        candidate_model
            .model_size_bytes
            .cmp(&other_model.model_size_bytes)
            .then_with(|| candidate_model.model_id.cmp(&other_model.model_id))
    });
}

#[test]
fn should_order_the_smallest_available_model_first_for_deployment_litmus() {
    let mut discovered_models = vec![
        DiscoveredModel {
            model_id: "larger-model".to_owned(),
            provider_model_id: None,
            model_family: astronomical_config::ModelFamily::Qwen3_5,
            revision: "larger-revision".to_owned(),
            model_directory: "/models/larger-model".into(),
            capabilities: chat_model_capabilities(),
            license: None,
            model_size_bytes: 9,
        },
        DiscoveredModel {
            model_id: "same-size-z-model".to_owned(),
            provider_model_id: None,
            model_family: astronomical_config::ModelFamily::Qwen3_5,
            revision: "same-size-z-revision".to_owned(),
            model_directory: "/models/same-size-z-model".into(),
            capabilities: chat_model_capabilities(),
            license: None,
            model_size_bytes: 3,
        },
        DiscoveredModel {
            model_id: "same-size-a-model".to_owned(),
            provider_model_id: None,
            model_family: astronomical_config::ModelFamily::Qwen3_5,
            revision: "same-size-a-revision".to_owned(),
            model_directory: "/models/same-size-a-model".into(),
            capabilities: chat_model_capabilities(),
            license: None,
            model_size_bytes: 3,
        },
    ];
    sort_deployment_litmus_models(&mut discovered_models);

    assert_eq!(
        discovered_models
            .first()
            .map(|discovered_model| discovered_model.model_id.as_str()),
        Some("same-size-a-model")
    );
}

fn chat_model_capabilities() -> astronomical_config::ModelCapabilities {
    astronomical_config::ModelCapabilities::Chat(astronomical_config::ChatModelCapabilities {
        context_window: 262_144,
        max_input_tokens: 241_664,
        max_output_tokens: 20_480,
        supports_vision: true,
        supports_reasoning: true,
        supports_tool_calls: true,
    })
}
