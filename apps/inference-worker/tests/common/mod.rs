#![cfg(any(
    feature = "model-artifact-qualification",
    feature = "memory-management-acceptance",
    feature = "performance-measurement"
))]

use std::{fs, path::PathBuf};

pub(crate) mod exact_model_prompt;
#[cfg(feature = "memory-management-acceptance")]
#[allow(dead_code)] // Shared module is compiled into independently gated test binaries.
pub(crate) mod real_model_rest_server;
#[cfg(feature = "memory-management-acceptance")]
pub(crate) mod solid_png;

#[allow(dead_code)] // Shared by independently feature-gated qualification binaries.
pub(crate) const ORNITH_MODEL_ARTIFACT_QUALIFICATION_MODEL_ID: &str = "Ornith-1.5-35B-A3B-oQ6e-mtp";

/// Current large sparse fixture for model-weight SSD-streaming journeys.
#[allow(dead_code)] // Shared by independently feature-gated qualification binaries.
pub(crate) const ORNITH_SSD_STREAMING_MODEL_ID: &str = "Ornith-1.5-35B-A3B-oQ6e-mtp";

/// Copies Development configuration into a temporary Development-shaped home
/// so qualification reads user policy without sharing mutable runtime state.
#[allow(dead_code)] // Shared by independently feature-gated qualification binaries.
pub(crate) fn isolated_development_home_from_user_config() -> tempfile::TempDir {
    let development_config =
        astronomical_config::AstronomicalConfig::load_from_development_location()
            .expect("Development configuration should load for isolated qualification");
    let isolated_development_home =
        tempfile::tempdir().expect("isolated Development home should be created");
    let isolated_state_directory = isolated_development_home.path().join(".astronomical-dev");
    fs::create_dir_all(&isolated_state_directory)
        .expect("isolated Development state should be created");
    fs::copy(
        development_config.instance_paths().config_file_path(),
        isolated_state_directory.join("config.json"),
    )
    .expect("Development config should copy into isolated qualification state");
    isolated_development_home
}

#[cfg(feature = "model-artifact-qualification")]
#[allow(dead_code)] // Used only by feature-specific test binaries.
pub(crate) fn discovered_model_artifact(
    model_id: &str,
    model_directory: &std::path::Path,
    max_output_tokens: u32,
) -> astronomical_config::DiscoveredModel {
    const CONTEXT_WINDOW: u32 = 262_144;
    astronomical_config::DiscoveredModel {
        model_id: model_id.to_owned(),
        provider_model_id: None,
        model_family: astronomical_config::ModelFamily::Qwen3_5,
        revision: "local-model-artifact-test".to_owned(),
        model_directory: model_directory.to_path_buf(),
        capabilities: astronomical_config::ModelCapabilities::Chat(
            astronomical_config::ChatModelCapabilities {
                context_window: CONTEXT_WINDOW,
                max_input_tokens: CONTEXT_WINDOW.saturating_sub(1),
                max_output_tokens,
                supports_vision: true,
                supports_reasoning: true,
                supports_tool_calls: true,
            },
        ),
        license: None,
        model_size_bytes: 0,
    }
}

#[cfg(feature = "model-artifact-qualification")]
#[allow(dead_code)] // Shared by independently feature-gated qualification modules.
pub(crate) fn chat_capabilities(
    discovered_model: &astronomical_config::DiscoveredModel,
) -> Option<&astronomical_config::ChatModelCapabilities> {
    match &discovered_model.capabilities {
        astronomical_config::ModelCapabilities::Chat(chat_capabilities) => Some(chat_capabilities),
        astronomical_config::ModelCapabilities::ImageGeneration(_) => None,
    }
}

#[cfg(feature = "model-artifact-qualification")]
#[allow(dead_code)] // Used only by model-artifact qualification test binaries.
pub(crate) fn configured_discovered_models() -> Vec<astronomical_config::DiscoveredModel> {
    let astronomical_config =
        astronomical_config::AstronomicalConfig::load_from_development_location().expect(
            "the Development Astronomical configuration should load for model qualification",
        );
    astronomical_config::discover_models(astronomical_config.model_directories())
        .expect("configured model-directory discovery should complete")
        .into_iter()
        .flat_map(|configured_model_directory_scan| {
            configured_model_directory_scan.discovered_models
        })
        .collect()
}

#[cfg(any(
    feature = "model-artifact-qualification",
    feature = "memory-management-acceptance",
    feature = "performance-measurement"
))]
#[allow(dead_code)]
pub(crate) fn configured_model_artifact_directory_by_id(model_id: &str) -> PathBuf {
    let astronomical_config =
        astronomical_config::AstronomicalConfig::load_from_development_location().expect(
            "the Development Astronomical configuration should load for model qualification",
        );
    astronomical_config
        .find_configured_model_directory_by_id(model_id)
        .unwrap_or_else(|discovery_error| {
            panic!(
                "model_directories discovery should complete for model ID {model_id}: {discovery_error}"
            )
        })
        .unwrap_or_else(|| {
            panic!(
                "the Development Astronomical configuration model_directories should discover model ID {model_id}"
            )
        })
}
