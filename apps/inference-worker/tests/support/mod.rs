//! Test bootstrapping that is not a product behavior: Development home, installed-model lookup, Romeo prompts, and one REST launcher.

use std::{fs, path::PathBuf};

pub(crate) mod exact_model_prompt;
#[cfg(any(
    feature = "serving-acceptance",
    feature = "memory-management-acceptance",
))]
pub(crate) mod http;
#[cfg(any(
    feature = "serving-acceptance",
    feature = "memory-management-acceptance",
))]
#[allow(dead_code)]
pub(crate) mod serving_rest;

#[path = "../../../../crates/model-serving/tests/common/e2e_test_model_names.rs"]
#[allow(dead_code)]
mod e2e_test_model_names;
#[allow(unused_imports)]
pub(crate) use e2e_test_model_names::{
    dense_mtp_model_id, e2e_test_model_ids, flux2_klein_model_id, laguna_xs_model_id,
    large_sparse_moe_model_id, required_e2e_test_model_ids, resident_sparse_moe_model_id,
    small_dense_model_id,
};

pub(crate) fn isolated_development_home_from_user_config() -> tempfile::TempDir {
    let development_config =
        astronomical_config::AstronomicalConfig::load_from_development_location()
            .expect("Development configuration should load for isolated acceptance");
    let isolated_development_home =
        tempfile::tempdir().expect("isolated Development home should be created");
    let isolated_state_directory = isolated_development_home.path().join(".astronomical-dev");
    fs::create_dir_all(&isolated_state_directory)
        .expect("isolated Development state should be created");
    fs::copy(
        development_config.instance_paths().config_file_path(),
        isolated_state_directory.join("config.json"),
    )
    .expect("Development config should copy into isolated acceptance state");
    isolated_development_home
}

#[allow(dead_code)]
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

#[allow(dead_code)]
pub(crate) fn chat_capabilities(
    discovered_model: &astronomical_config::DiscoveredModel,
) -> Option<&astronomical_config::ChatModelCapabilities> {
    match &discovered_model.capabilities {
        astronomical_config::ModelCapabilities::Chat(chat_capabilities) => Some(chat_capabilities),
        astronomical_config::ModelCapabilities::ImageGeneration(_) => None,
    }
}

#[allow(dead_code)]
pub(crate) fn configured_discovered_models() -> Vec<astronomical_config::DiscoveredModel> {
    let astronomical_config =
        astronomical_config::AstronomicalConfig::load_from_development_location().expect(
            "the Development Astronomical configuration should load for installed-model lookup",
        );
    astronomical_config::discover_models(astronomical_config.model_directories())
        .expect("configured model-directory discovery should complete")
        .into_iter()
        .flat_map(|configured_model_directory_scan| {
            configured_model_directory_scan.discovered_models
        })
        .collect()
}

pub(crate) fn configured_installed_model_directory_by_id(model_id: &str) -> PathBuf {
    let astronomical_config =
        astronomical_config::AstronomicalConfig::load_from_development_location().expect(
            "the Development Astronomical configuration should load for installed-model lookup",
        );
    match astronomical_config.find_configured_model_directory_by_id(model_id) {
        Ok(Some(model_directory)) => model_directory,
        Ok(None) => panic!(
            "the Development Astronomical configuration model_directories should discover model ID {model_id}"
        ),
        Err(_) => first_configured_model_directory_by_id(&astronomical_config, model_id)
            .unwrap_or_else(|| {
                panic!(
                    "the Development Astronomical configuration model_directories should discover model ID {model_id}"
                )
            }),
    }
}

fn first_configured_model_directory_by_id(
    astronomical_config: &astronomical_config::AstronomicalConfig,
    model_id: &str,
) -> Option<PathBuf> {
    for model_root in astronomical_config.model_directories() {
        if !model_root.is_dir() {
            continue;
        }
        let direct_child = model_root.join(model_id);
        if direct_child.is_dir() {
            return Some(direct_child);
        }
        let Ok(root_entries) = fs::read_dir(model_root) else {
            continue;
        };
        for root_entry in root_entries.flatten() {
            let organization_directory = root_entry.path();
            if !organization_directory.is_dir() {
                continue;
            }
            let named_model_directory = organization_directory.join(model_id);
            if named_model_directory.is_dir() {
                return Some(named_model_directory);
            }
        }
    }
    None
}
