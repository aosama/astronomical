//! Unit tests for reading a folder's trained head.

use decision_contract::head_tuning_file::head_tuning_bytes;

use super::*;

fn laya_scorer_bias() -> HeadTuning {
    HeadTuning {
        tensors: vec![TunedTensor {
            name: "scorer.layers.3.bias".to_owned(),
            shape: vec![1],
            values: vec![0.5],
        }],
    }
}

#[test]
fn should_read_no_tuning_from_a_folder_without_the_file() {
    let model_directory = tempfile::tempdir().unwrap();
    let head_tuning = read_head_tuning(model_directory.path(), TunableHeadFamily::Laya).unwrap();
    assert!(head_tuning.is_empty());
}

#[test]
fn should_read_the_tuning_a_folder_carries() {
    let model_directory = tempfile::tempdir().unwrap();
    std::fs::write(
        model_directory.path().join(DECISION_HEAD_TUNING_FILE_NAME),
        head_tuning_bytes(&laya_scorer_bias()).unwrap(),
    )
    .unwrap();
    let head_tuning = read_head_tuning(model_directory.path(), TunableHeadFamily::Laya).unwrap();
    assert_eq!(head_tuning, laya_scorer_bias());
}

#[test]
fn should_refuse_to_load_with_a_tuning_it_cannot_use() {
    let model_directory = tempfile::tempdir().unwrap();
    std::fs::write(
        model_directory.path().join(DECISION_HEAD_TUNING_FILE_NAME),
        b"Wherefore art thou Romeo",
    )
    .unwrap();
    let read_outcome = read_head_tuning(model_directory.path(), TunableHeadFamily::Laya);
    assert!(read_outcome.is_err());
    // A Laya file is not a Kev file.
    std::fs::write(
        model_directory.path().join(DECISION_HEAD_TUNING_FILE_NAME),
        head_tuning_bytes(&laya_scorer_bias()).unwrap(),
    )
    .unwrap();
    assert!(read_head_tuning(model_directory.path(), TunableHeadFamily::Kev).is_err());
}

#[test]
fn should_accept_only_the_shipped_shape() {
    let tuned_bias = &laya_scorer_bias().tensors[0];
    assert!(check_tuned_shape(tuned_bias, &[1]).is_ok());
    assert!(check_tuned_shape(tuned_bias, &[2]).is_err());
    assert!(check_tuned_shape(tuned_bias, &[]).is_err());
}
