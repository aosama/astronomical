//! Unit contracts for removing local paths from a copied document.

use serde_json::json;

use super::without_local_paths;

#[test]
fn should_remove_path_like_text_at_any_depth_and_keep_everything_else() {
    let scrubbed_document = without_local_paths(json!({
        "cls_token": "[CLS]",
        "model_max_length": 8192,
        "nested": {
            "cache_dir": "~/fictional-cache",
            "kept": "plain text",
            "windows": "C:\\fictional\\models"
        },
        "files": ["./vocabulary.txt", "../other.txt", "kept.txt", "file:///fictional/x"]
    }));

    assert_eq!(
        scrubbed_document,
        json!({
            "cls_token": "[CLS]",
            "model_max_length": 8192,
            "nested": {"kept": "plain text"},
            "files": ["kept.txt"]
        })
    );
}

#[test]
fn should_keep_text_that_only_contains_a_slash() {
    let scrubbed_document = without_local_paths(json!({"encoder": "example-org/example-encoder"}));

    assert_eq!(
        scrubbed_document,
        json!({"encoder": "example-org/example-encoder"})
    );
}
