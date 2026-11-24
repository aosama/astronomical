use crate::model_identity::{configured_model_id_for, near_model_matches, resolve_model_id};

#[test]
fn near_model_matches_survive_partial_typos() {
    let candidates = &["Qwen3.5-2B-4bit", "Ornith-1.5-35B-6bit", "gpt-oss-20b"];
    assert_eq!(
        near_model_matches("qwen3.5-2b", candidates),
        vec!["qwen3.5-2b-4bit".to_owned()]
    );
    assert_eq!(
        near_model_matches("qwen3-5-2b-4bit", candidates),
        vec!["qwen3.5-2b-4bit".to_owned()]
    );
    assert_eq!(
        near_model_matches("other-ns/Qwen3.5-2B-4bit", candidates),
        vec!["qwen3.5-2b-4bit".to_owned()]
    );
    assert!(near_model_matches("definitely-not-a-model", candidates).is_empty());
}

#[test]
fn near_model_matches_accepts_owned_candidate_strings() {
    let candidates = vec!["Qwen3.5-2B-4bit".to_owned()];
    assert_eq!(
        near_model_matches("QWEN3.5", &candidates),
        vec!["qwen3.5-2b-4bit".to_owned()]
    );
}

#[test]
fn should_resolve_a_full_huggingface_id_to_its_known_leaf() {
    let known_model_ids = ["Ornith-1.5-35B-A3B-OptiQ-4bit", "other-model"];

    assert_eq!(
        resolve_model_id(
            "mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit",
            &known_model_ids
        ),
        "Ornith-1.5-35B-A3B-OptiQ-4bit"
    );
    assert_eq!(
        resolve_model_id("Ornith-1.5-35B-A3B-OptiQ-4bit", &known_model_ids),
        "Ornith-1.5-35B-A3B-OptiQ-4bit"
    );
}

#[test]
fn should_return_an_unknown_or_empty_model_id_unchanged() {
    let known_model_ids = ["other-model"];

    assert_eq!(
        resolve_model_id("mlx-community/not-installed", &known_model_ids),
        "mlx-community/not-installed"
    );
    assert_eq!(resolve_model_id("", &known_model_ids), "");
    assert_eq!(resolve_model_id("anything", &[]), "anything");
}

#[test]
fn should_prefer_an_exact_known_id_over_its_leaf() {
    let known_model_ids = ["org/model", "model"];

    assert_eq!(resolve_model_id("org/model", &known_model_ids), "org/model");
}

#[test]
fn should_pick_the_exact_entry_before_any_full_id_entry() {
    let configured_model_ids = ["a-org/model", "model"];

    assert_eq!(
        configured_model_id_for("model", Some("a-org/model"), configured_model_ids),
        Some("model")
    );
}

#[test]
fn should_pick_the_entry_equal_to_the_recorded_provider_id() {
    let configured_model_ids = ["a-org/model", "b-org/model"];

    assert_eq!(
        configured_model_id_for("model", Some("b-org/model"), configured_model_ids),
        Some("b-org/model")
    );
}

#[test]
fn should_not_apply_another_organisations_entry_once_the_provider_id_is_known() {
    assert_eq!(
        configured_model_id_for("model", Some("b-org/model"), ["a-org/model"]),
        None
    );
}

#[test]
fn should_fall_back_to_the_first_leaf_match_only_without_a_recorded_provider_id() {
    let configured_model_ids = ["a-org/model", "b-org/model"];

    assert_eq!(
        configured_model_id_for("model", None, configured_model_ids),
        Some("a-org/model")
    );
}

#[test]
fn should_not_match_an_entry_for_a_different_model_or_a_longer_name() {
    assert_eq!(configured_model_id_for("model", None, ["org/other"]), None);
    assert_eq!(
        configured_model_id_for("model", None, ["org/model-4bit"]),
        None
    );
    assert_eq!(
        configured_model_id_for("model", None, ["org/my-model"]),
        None
    );
    assert_eq!(configured_model_id_for("model", None, []), None);
}

#[test]
fn should_match_a_multi_segment_entry_by_its_last_segment_and_ignore_a_trailing_slash() {
    assert_eq!(
        configured_model_id_for("model", None, ["x/org/model"]),
        Some("x/org/model")
    );
    assert_eq!(configured_model_id_for("model", None, ["org/model/"]), None);
}

#[test]
fn should_resolve_multi_segment_and_malformed_ids_by_their_last_segment() {
    let known_model_ids = ["model"];

    assert_eq!(resolve_model_id("a/b/model", &known_model_ids), "model");
    assert_eq!(resolve_model_id("/model", &known_model_ids), "model");
    assert_eq!(
        resolve_model_id("org/model/", &known_model_ids),
        "org/model/"
    );
    assert_eq!(resolve_model_id("org/MODEL", &known_model_ids), "org/MODEL");
}
