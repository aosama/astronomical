//! Unit tests for checkpoint monotonicity and the OptiQ-compatible shape.

use std::collections::BTreeMap;

use super::*;
use crate::bit_width::BitWidth;

fn result(module_name: &str, kl_at_four: f64) -> SensitivityResult {
    SensitivityResult {
        module_name: module_name.to_owned(),
        kl_divergence_by_bit_width: BTreeMap::from([
            (BitWidth::new(4).expect("valid"), kl_at_four),
            (BitWidth::new(8).expect("valid"), kl_at_four / 2.0),
        ]),
        logical_parameter_count: 4096,
    }
}

#[test]
fn should_only_ever_grow_the_checkpoint_when_saving_a_partial_result_set() {
    let checkpoint_directory =
        tempfile::tempdir().expect("a temporary directory should be creatable");
    let checkpoint_path = checkpoint_directory
        .path()
        .join("sensitivity_checkpoint.json");

    merge_and_save_checkpoint(&checkpoint_path, &[result("a", 1.0), result("b", 2.0)])
        .expect("the first save should succeed");
    merge_and_save_checkpoint(&checkpoint_path, &[result("c", 3.0)])
        .expect("the second save should succeed");

    let reloaded = load_checkpoint(&checkpoint_path);
    let module_names: Vec<&str> = reloaded
        .iter()
        .map(|entry| entry.module_name.as_str())
        .collect();
    assert_eq!(module_names, vec!["a", "b", "c"]);
}

#[test]
fn should_read_an_optiq_python_checkpoint_document() {
    // The exact JSON shape the Python tool writes; string bit-width keys.
    let checkpoint_directory =
        tempfile::tempdir().expect("a temporary directory should be creatable");
    let checkpoint_path = checkpoint_directory
        .path()
        .join("sensitivity_checkpoint.json");
    std::fs::write(
        &checkpoint_path,
        r#"[{"layer_name": "model.layers.0.mlp.up_proj", "sensitivities": {"4": 0.01, "8": 0.002}, "param_count": 4096, "metadata": {}}]"#,
    )
    .expect("the fixture checkpoint should be writable");

    let reloaded = load_checkpoint(&checkpoint_path);

    assert_eq!(reloaded.len(), 1);
    assert_eq!(reloaded[0].logical_parameter_count, 4096);
    assert_eq!(
        reloaded[0].kl_divergence_at(BitWidth::new(8).expect("valid")),
        Some(0.002)
    );
}

#[test]
fn should_treat_a_corrupt_checkpoint_as_empty() {
    let checkpoint_directory =
        tempfile::tempdir().expect("a temporary directory should be creatable");
    let checkpoint_path = checkpoint_directory
        .path()
        .join("sensitivity_checkpoint.json");
    std::fs::write(&checkpoint_path, "{not json").expect("the fixture should be writable");

    assert!(load_checkpoint(&checkpoint_path).is_empty());
}

#[test]
fn should_drop_checkpoint_entries_that_do_not_cover_every_requested_width() {
    let candidate_bits = [
        BitWidth::new(2).expect("valid"),
        BitWidth::new(4).expect("valid"),
        BitWidth::new(8).expect("valid"),
    ];

    let complete = complete_entries_for(vec![result("a", 1.0)], &candidate_bits);

    assert!(complete.is_empty());
}

#[test]
fn should_read_a_python_checkpoint_whose_divergence_is_a_bare_nan_literal() {
    let checkpoint_directory =
        tempfile::tempdir().expect("a temporary directory should be creatable");
    let checkpoint_path = checkpoint_directory
        .path()
        .join("sensitivity_checkpoint.json");
    std::fs::write(
        &checkpoint_path,
        r#"[{"layer_name": "NaN.layer", "sensitivities": {"4": NaN, "8": 0.5}, "param_count": 8}]"#,
    )
    .expect("the fixture checkpoint should be writable");

    let reloaded = load_checkpoint(&checkpoint_path);

    assert_eq!(reloaded.len(), 1);
    assert_eq!(reloaded[0].module_name, "NaN.layer");
    assert!(reloaded[0].has_non_finite_measurement());
}

#[test]
fn should_round_trip_a_non_finite_divergence_and_replace_a_rescored_module() {
    let checkpoint_directory =
        tempfile::tempdir().expect("a temporary directory should be creatable");
    let checkpoint_path = checkpoint_directory
        .path()
        .join("sensitivity_checkpoint.json");
    merge_and_save_checkpoint(&checkpoint_path, &[result("a", f64::NAN)])
        .expect("the first save should succeed");

    merge_and_save_checkpoint(&checkpoint_path, &[result("a", 0.25)])
        .expect("the second save should succeed");
    let reloaded = load_checkpoint(&checkpoint_path);

    assert_eq!(reloaded.len(), 1);
    assert_eq!(
        reloaded[0].kl_divergence_at(BitWidth::new(4).expect("valid")),
        Some(0.25)
    );
    assert!(
        !checkpoint_directory
            .path()
            .join("sensitivity_checkpoint.json.tmp")
            .exists()
    );
}

#[test]
fn should_load_nothing_from_an_absent_checkpoint() {
    let checkpoint_directory =
        tempfile::tempdir().expect("a temporary directory should be creatable");

    assert!(load_checkpoint(&checkpoint_directory.path().join("absent.json")).is_empty());
}

#[test]
fn should_keep_an_entry_that_covers_every_requested_width() {
    let candidate_bits = [
        BitWidth::new(4).expect("valid"),
        BitWidth::new(8).expect("valid"),
    ];

    let complete = complete_entries_for(vec![result("a", 1.0)], &candidate_bits);

    assert_eq!(complete.len(), 1);
}
