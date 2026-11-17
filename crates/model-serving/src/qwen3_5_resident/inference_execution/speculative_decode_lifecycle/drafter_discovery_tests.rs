use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use super::drafter_discovery::{
    BUNDLED_DRAFTER_DIRECTORY_NAME, DrafterPairing, paired_drafter_directory,
};

const TARGET_27B: DrafterPairing = DrafterPairing {
    target_layer_count: 64,
    target_hidden_size: 5120,
    target_vocabulary_size: 248_320,
};

/// A DFlash2 drafter configuration for a target with `target_layer_count` layers.
fn drafter_document(target_layer_count: i32) -> Value {
    json!({
        "architectures": ["DFlash2DraftModel"],
        "dflash_config": {
            "block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2,
            "mask_token_id": 248070, "selector_rank": 256, "selector_top_k": 16,
            "target_layer_ids": [5, 19, 33, 47, 61]
        },
        "head_dim": 128, "hidden_size": 5120, "intermediate_size": 17408,
        "is_causal": false,
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention"],
        "num_attention_heads": 32, "num_hidden_layers": 5, "num_key_value_heads": 8,
        "num_target_layers": target_layer_count, "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "sliding_window": 2048, "vocab_size": 248320
    })
}

fn write_config(model_directory: &Path, document: &Value) {
    std::fs::create_dir_all(model_directory).expect("the model directory is created");
    std::fs::write(
        model_directory.join("config.json"),
        serde_json::to_vec(document).expect("the document serializes"),
    )
    .expect("the config is written");
}

/// A `huggingface_hub_cache` snapshot directory for `organization/name`.
fn hub_snapshot(hub_root: &Path, organization: &str, name: &str) -> PathBuf {
    hub_root
        .join(format!("models--{organization}--{name}"))
        .join("snapshots")
        .join("0123abcd")
}

#[test]
fn should_find_a_pairing_drafter_in_the_huggingface_hub_cache_layout() {
    let hub_root = tempfile::tempdir().expect("a temporary directory");
    let target = hub_snapshot(hub_root.path(), "acme", "target-27b");
    write_config(&target, &json!({"model_type": "qwen3_5"}));
    let drafter = hub_snapshot(hub_root.path(), "lab", "target-27b-drafter");
    write_config(&drafter, &drafter_document(64));
    let found = paired_drafter_directory(&target, &TARGET_27B)
        .expect("the library lists")
        .expect("the drafter pairs");
    assert_eq!(found.0, drafter);
    assert_eq!(found.1.target_layer_count, 64);
}

#[test]
fn should_find_a_pairing_drafter_in_a_plain_library_layout() {
    let library_root = tempfile::tempdir().expect("a temporary directory");
    let target = library_root.path().join("acme").join("target-27b");
    write_config(&target, &json!({"model_type": "qwen3_5"}));
    let drafter = library_root.path().join("lab").join("target-27b-drafter");
    write_config(&drafter, &drafter_document(64));
    let found = paired_drafter_directory(&target, &TARGET_27B)
        .expect("the library lists")
        .expect("the drafter pairs");
    assert_eq!(found.0, drafter);
}

#[test]
fn should_pass_over_a_drafter_built_for_another_target_shape() {
    let library_root = tempfile::tempdir().expect("a temporary directory");
    let target = library_root.path().join("acme").join("target-27b");
    write_config(&target, &json!({"model_type": "qwen3_5"}));
    write_config(
        &library_root.path().join("lab").join("other-drafter"),
        &drafter_document(48),
    );
    assert!(
        paired_drafter_directory(&target, &TARGET_27B)
            .expect("the library lists")
            .is_none()
    );
}

#[test]
fn should_pass_over_drafters_it_does_not_implement_and_unreadable_configs() {
    let library_root = tempfile::tempdir().expect("a temporary directory");
    let target = library_root.path().join("acme").join("target-27b");
    write_config(&target, &json!({"model_type": "qwen3_5"}));
    // A drafter of another architecture, and a config that is not JSON.
    let mut other_architecture = drafter_document(64);
    other_architecture["architectures"] = json!(["SomeOtherDraftModel"]);
    write_config(
        &library_root.path().join("lab").join("other-kind"),
        &other_architecture,
    );
    let broken = library_root.path().join("lab").join("broken");
    std::fs::create_dir_all(&broken).expect("the directory is created");
    std::fs::write(broken.join("config.json"), b"not json").expect("the file is written");
    assert!(
        paired_drafter_directory(&target, &TARGET_27B)
            .expect("the library lists")
            .is_none()
    );
}

#[test]
fn should_find_nothing_for_a_library_with_only_the_target() {
    let library_root = tempfile::tempdir().expect("a temporary directory");
    let target = library_root.path().join("acme").join("target-27b");
    write_config(&target, &json!({"model_type": "qwen3_5"}));
    assert!(
        paired_drafter_directory(&target, &TARGET_27B)
            .expect("the library lists")
            .is_none()
    );
}

/// A drafter in the target's own `drafter` folder belongs to that model and
/// wins over a library drafter of the same shape, so a draft-tree build and
/// its plain twin can share one library.
#[test]
fn should_prefer_the_drafter_bundled_inside_the_target_directory() {
    let library_root = tempfile::tempdir().expect("a temporary directory");
    let target = library_root.path().join("local").join("target-27b-drafted");
    write_config(&target, &json!({"model_type": "qwen3_5"}));
    let bundled_drafter = target.join(BUNDLED_DRAFTER_DIRECTORY_NAME);
    write_config(&bundled_drafter, &drafter_document(64));
    write_config(
        &library_root.path().join("lab").join("a-library-drafter"),
        &drafter_document(64),
    );

    let found = paired_drafter_directory(&target, &TARGET_27B)
        .expect("the library lists")
        .expect("the bundled drafter pairs");

    assert_eq!(found.0, bundled_drafter);
}

#[test]
fn should_fall_back_to_the_library_when_the_bundled_drafter_does_not_pair() {
    let library_root = tempfile::tempdir().expect("a temporary directory");
    let target = library_root.path().join("local").join("target-27b-drafted");
    write_config(&target, &json!({"model_type": "qwen3_5"}));
    write_config(
        &target.join(BUNDLED_DRAFTER_DIRECTORY_NAME),
        &drafter_document(40),
    );
    let library_drafter = library_root.path().join("lab").join("a-library-drafter");
    write_config(&library_drafter, &drafter_document(64));

    let found = paired_drafter_directory(&target, &TARGET_27B)
        .expect("the library lists")
        .expect("the library drafter pairs");

    assert_eq!(found.0, library_drafter);
}
