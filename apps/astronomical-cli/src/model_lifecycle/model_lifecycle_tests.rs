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
        supports_decisions: false,
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
fn should_render_download_progress_with_apple_box_gigabytes() {
    let download_job = DaemonDownloadJob {
        huggingface_id: "example/2gb-model".to_owned(),
        state: "downloading".to_owned(),
        bytes_completed: 1_500_000_000,
        bytes_total: 2_000_000_000,
        error: None,
    };
    // The user-facing gigabyte is the one Apple prints on the box (2^30,
    // per AGENTS.md), so 1.5/2.0 decimal gigabytes of bytes render as
    // 1.4/1.86 user-facing gigabytes at the same 75%.
    assert_eq!(
        model_lifecycle::render_download_progress(&download_job),
        "example/2gb-model: downloading 1.4 GB / 1.86 GB (75%)"
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
        supports_decisions: false,
        supports_image_generation: false,
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

fn listed_model(model_id: &str, context_window: Option<u32>, can_draw: bool) -> DaemonListedModel {
    DaemonListedModel {
        model_id: model_id.to_owned(),
        family: "stub".to_owned(),
        context_window,
        supports_embeddings: false,
        supports_decisions: false,
        supports_image_generation: can_draw,
        is_resident: false,
        size_bytes: 1,
    }
}

#[test]
fn should_describe_only_image_models_as_image_capable() {
    let image_model = listed_model("Qwen-Image-2.1-MLX-4bit", None, true);
    let chat_model = listed_model("Qwen3.5-2B-4bit", Some(32_768), false);

    assert!(RequiredCapability::ImageGeneration.describes_installed_model(&image_model));
    assert!(!RequiredCapability::ImageGeneration.describes_installed_model(&chat_model));
    assert!(!RequiredCapability::Chat.describes_installed_model(&image_model));
    assert!(!RequiredCapability::Embeddings.describes_installed_model(&image_model));
}

#[test]
fn should_describe_only_image_catalog_entries_as_image_capable() {
    let mut image_entry = catalog_entry("mlx-community/Qwen-Image-2.1-MLX-4bit", None);
    image_entry.context_window = None;
    image_entry.supports_image_generation = true;
    let chat_entry = catalog_entry("mlx-community/Qwen3.5-2B-4bit", None);

    assert!(RequiredCapability::ImageGeneration.describes_catalog_entry(&image_entry));
    assert!(!RequiredCapability::ImageGeneration.describes_catalog_entry(&chat_entry));
}

#[test]
fn should_say_image_generation_when_a_chat_model_is_asked_to_draw() {
    let chat_model = listed_model("Qwen3.5-2B-4bit", Some(32_768), false);
    let reason = model_lifecycle::unknown_model_reason(
        "Qwen-Image-2.1",
        &[chat_model],
        RequiredCapability::ImageGeneration,
    );

    assert!(reason.contains("image-generation models"), "{reason}");
}

#[test]
fn should_map_a_full_huggingface_id_to_the_installed_models_own_id() {
    let installed_models = vec![listed_model("Custom-4bit", Some(32_768), false)];

    assert_eq!(
        model_lifecycle::installed_model_id_for(
            &installed_models,
            "someorg/Custom-4bit".to_owned()
        ),
        "Custom-4bit"
    );
    assert_eq!(
        model_lifecycle::installed_model_id_for(&installed_models, "Custom-4bit".to_owned()),
        "Custom-4bit"
    );
}

#[test]
fn should_leave_an_id_that_names_no_installed_model_unchanged() {
    let installed_models = vec![listed_model("Custom-4bit", Some(32_768), false)];

    assert_eq!(
        model_lifecycle::installed_model_id_for(&installed_models, "someorg/Other-4bit".to_owned()),
        "someorg/Other-4bit"
    );
    assert_eq!(
        model_lifecycle::installed_model_id_for(&installed_models, String::new()),
        ""
    );
    assert_eq!(
        model_lifecycle::installed_model_id_for(&[], "someorg/Custom-4bit".to_owned()),
        "someorg/Custom-4bit"
    );
}
