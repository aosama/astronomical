#![cfg(any(
    feature = "model-artifact-qualification",
    feature = "performance-measurement"
))]

use std::{collections::HashMap, path::Path, path::PathBuf, sync::Arc};

pub(crate) mod exact_model_prompt;

#[allow(dead_code)] // Shared by independently feature-gated qualification binaries.
pub(crate) const ORNITH_MODEL_ARTIFACT_QUALIFICATION_MODEL_ID: &str =
    astronomical_model_serving::ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID;

#[allow(dead_code)] // Used only by feature-specific test binaries.
pub(crate) fn single_model_directories(
    model_id: &str,
    model_directory: &Path,
) -> Arc<HashMap<String, PathBuf>> {
    Arc::new(HashMap::from([(
        model_id.to_owned(),
        model_directory.to_path_buf(),
    )]))
}

#[cfg(feature = "model-artifact-qualification")]
#[allow(dead_code)] // Used only by feature-specific test binaries.
pub(crate) fn discovered_model_artifact(
    model_id: &str,
    model_directory: &Path,
    max_output_tokens: u32,
) -> astronomical_config::DiscoveredModel {
    const CONTEXT_WINDOW: u32 = 262_144;
    astronomical_config::DiscoveredModel {
        model_id: model_id.to_owned(),
        model_family: astronomical_config::ModelFamily::Qwen3_5,
        revision: "local-model-artifact-test".to_owned(),
        model_directory: model_directory.to_path_buf(),
        context_window: CONTEXT_WINDOW,
        max_input_tokens: CONTEXT_WINDOW.saturating_sub(max_output_tokens),
        max_output_tokens,
        has_vision: true,
        model_size_bytes: 0,
    }
}

#[cfg(feature = "model-artifact-qualification")]
#[allow(dead_code)] // Used only by model-artifact qualification test binaries.
pub(crate) fn configured_discovered_models() -> Vec<astronomical_config::DiscoveredModel> {
    let astronomical_config = astronomical_config::AstronomicalConfig::load_from_default_location()
        .expect("the standard Astronomical configuration should load for model qualification");
    astronomical_config::discover_models(
        astronomical_config.model_directories(),
        astronomical_config.max_output_tokens(),
    )
    .expect("configured model-directory discovery should complete")
    .into_iter()
    .flat_map(|configured_model_directory_scan| configured_model_directory_scan.discovered_models)
    .collect()
}

#[cfg(any(
    feature = "model-artifact-qualification",
    feature = "performance-measurement"
))]
#[allow(dead_code)]
pub(crate) fn configured_model_artifact_directory_by_id(model_id: &str) -> PathBuf {
    let astronomical_config = astronomical_config::AstronomicalConfig::load_from_default_location()
        .expect("the standard Astronomical configuration should load for model qualification");
    astronomical_config
        .find_configured_model_directory_by_id(model_id)
        .unwrap_or_else(|discovery_error| {
            panic!(
                "model_directories discovery should complete for model ID {model_id}: {discovery_error}"
            )
        })
        .unwrap_or_else(|| {
            panic!(
                "the standard Astronomical configuration model_directories should discover model ID {model_id}"
            )
        })
}
