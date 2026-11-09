use std::path::{Path, PathBuf};

use astronomical_config::{ChatModelCapabilities, DiscoveredModel, ModelCapabilities, ModelFamily};
use astronomical_ipc_protocol::ModelLoadInFlight;

use super::{ModelFolderBusy, model_folder_busy_reason, model_ids_served_from_folder};

const REMOVED_FOLDER: &str = "/fictional/models/example-org/example-qwen";

fn discovered_model(model_id: &str, model_directory: &str) -> DiscoveredModel {
    DiscoveredModel {
        model_id: model_id.to_owned(),
        provider_model_id: None,
        model_family: ModelFamily::Qwen3_5,
        revision: "fictional-revision".to_owned(),
        model_directory: PathBuf::from(model_directory),
        capabilities: ModelCapabilities::Chat(ChatModelCapabilities {
            context_window: 65_536,
            max_input_tokens: 65_535,
            max_output_tokens: 20_480,
            supports_vision: false,
            supports_reasoning: true,
            supports_tool_calls: true,
        }),
        license: None,
        model_size_bytes: 1_000,
    }
}

fn folder_model_ids() -> Vec<String> {
    vec!["example-qwen".to_owned()]
}

fn loading(model_id: Option<&str>) -> ModelLoadInFlight {
    ModelLoadInFlight {
        model_id: model_id.map(str::to_owned),
    }
}

#[test]
fn should_name_only_the_models_discovered_inside_the_folder() {
    let discovered_models = [
        discovered_model("example-qwen", REMOVED_FOLDER),
        discovered_model("example-qwen-nested", &format!("{REMOVED_FOLDER}/variant")),
        discovered_model("other-model", "/fictional/models/example-org/other-model"),
        // A sibling whose name merely starts with the folder's name is elsewhere.
        discovered_model(
            "example-qwen-2",
            "/fictional/models/example-org/example-qwen-2",
        ),
    ];

    assert_eq!(
        model_ids_served_from_folder(&discovered_models, Path::new(REMOVED_FOLDER)),
        ["example-qwen", "example-qwen-nested"]
    );
}

#[test]
fn should_report_a_loaded_model_from_the_folder() {
    assert_eq!(
        model_folder_busy_reason(&folder_model_ids(), Some("example-qwen"), None),
        Some(ModelFolderBusy::Loaded)
    );
}

#[test]
fn should_report_a_load_in_flight_of_a_model_from_the_folder() {
    assert_eq!(
        model_folder_busy_reason(
            &folder_model_ids(),
            None,
            Some(&loading(Some("example-qwen")))
        ),
        Some(ModelFolderBusy::Loading)
    );
}

#[test]
fn should_refuse_while_a_load_cannot_name_its_model() {
    assert_eq!(
        model_folder_busy_reason(
            &folder_model_ids(),
            Some("other-model"),
            Some(&loading(None))
        ),
        Some(ModelFolderBusy::Loading)
    );
}

#[test]
fn should_allow_removal_while_another_model_is_loaded_or_loading() {
    assert_eq!(
        model_folder_busy_reason(
            &folder_model_ids(),
            Some("other-model"),
            Some(&loading(Some("another-model")))
        ),
        None
    );
}

#[test]
fn should_allow_removal_when_nothing_is_loaded() {
    assert_eq!(
        model_folder_busy_reason(&folder_model_ids(), None, None),
        None
    );
    assert_eq!(
        model_folder_busy_reason(&[], Some("example-qwen"), None),
        None
    );
}
