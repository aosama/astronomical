//! Unit tests for `crate::model_lifecycle`: catalog matching, download
//! progress rendering, and unknown-model guidance.

use astronomical_ipc_protocol::{DaemonCatalogEntry, DaemonDownloadJob, DaemonListedModel};

use crate::model_lifecycle::{self, RequiredCapability};

fn catalog_entry(huggingface_id: &str, requestable_model_id: Option<&str>) -> DaemonCatalogEntry {
    DaemonCatalogEntry {
        huggingface_id: huggingface_id.to_owned(),
        display_name: huggingface_id.to_owned(),
        family: "qwen".to_owned(),
        approximate_size_bytes: 1_000_000_000,
        ready_on_this_mac: false,
        requestable_model_id: requestable_model_id.map(str::to_owned),
        download_state: None,
        context_window: Some(32_768),
        supports_reasoning: false,
        supports_vision: false,
        supports_tool_calls: false,
        supports_image_generation: false,
        supports_embeddings: false,
    }
}

#[test]
fn should_match_catalog_entries_by_huggingface_id_or_leaf_even_when_not_ready() {
    let ready_catalog_entry =
        catalog_entry("mlx-community/Qwen3.5-2B-4bit", Some("Qwen3.5-2B-4bit"));
    assert!(model_lifecycle::catalog_entry_matches(
        &ready_catalog_entry,
        "mlx-community/Qwen3.5-2B-4bit"
    ));
    assert!(model_lifecycle::catalog_entry_matches(
        &ready_catalog_entry,
        "Qwen3.5-2B-4bit"
    ));
    assert!(model_lifecycle::catalog_entry_matches(
        &ready_catalog_entry,
        "other-ns/Qwen3.5-2B-4bit"
    ));
    assert!(!model_lifecycle::catalog_entry_matches(
        &ready_catalog_entry,
        "Qwen3.5-4B-4bit"
    ));
    // Not-ready entries carry no requestable id on the wire; the leaf
    // must still match because the daemon derives it from the hf id.
    let not_ready_catalog_entry = catalog_entry("mlx-community/Qwen3.5-2B-4bit", None);
    assert!(model_lifecycle::catalog_entry_matches(
        &not_ready_catalog_entry,
        "Qwen3.5-2B-4bit"
    ));
}

#[test]
fn should_render_download_progress_with_decimal_gigabytes() {
    let download_job = DaemonDownloadJob {
        huggingface_id: "example/2gb-model".to_owned(),
        state: "downloading".to_owned(),
        bytes_completed: 1_500_000_000,
        bytes_total: 2_000_000_000,
        error: None,
    };
    assert_eq!(
        model_lifecycle::render_download_progress(&download_job),
        "example/2gb-model: downloading 1.5 GB / 2 GB (75%)"
    );
}

#[test]
fn should_render_download_progress_without_a_total_as_state_only() {
    let download_job = DaemonDownloadJob {
        huggingface_id: "example/unknown-size".to_owned(),
        state: "fetching_manifest".to_owned(),
        bytes_completed: 0,
        bytes_total: 0,
        error: None,
    };
    assert_eq!(
        model_lifecycle::render_download_progress(&download_job),
        "example/unknown-size: fetching_manifest"
    );
}

#[test]
fn should_list_near_matches_and_the_supported_hint_for_an_unknown_model() {
    let installed_models = vec![DaemonListedModel {
        model_id: "Qwen3.5-2B-4bit".to_owned(),
        family: "qwen3_5".to_owned(),
        context_window: Some(32_768),
        supports_embeddings: false,
        is_resident: false,
        size_bytes: 1_500_000_000,
    }];
    let reason = model_lifecycle::unknown_model_reason(
        "Qwen3.5-14B-4bit",
        &installed_models,
        RequiredCapability::Chat,
    );
    assert!(reason.contains("did you mean: qwen3.5-2b-4bit"), "{reason}");
    assert!(reason.contains("astronomical models supported"), "{reason}");
}
