//! Unit tests for the proven copy of a model folder.

use astronomical_decision_contract::DecisionQuestionKind;
use astronomical_decision_contract::calibration_file::{
    KindCalibration, parse_calibration_document,
};
use astronomical_decision_contract::decision_adapters_file::LinearAdapter;
use astronomical_decision_contract::head_tuning_file::TunedTensor;

use super::*;
use crate::test_support::ROMEO_AND_JULIET_LINES;

#[derive(Default)]
struct RecordedProgress {
    reports: Vec<(u64, u64)>,
}

impl FolderCopyProgress for RecordedProgress {
    fn report(&mut self, processed_bytes: u64, total_bytes: u64) {
        self.reports.push((processed_bytes, total_bytes));
    }
}

fn noul_calibration() -> CalibrationOverride {
    CalibrationOverride {
        noul: Some(KindCalibration {
            temperature: 3.5,
            range_temperatures: Vec::new(),
            fitted_question_count: 100,
        }),
        ..CalibrationOverride::default()
    }
}

/// A small model folder: weights, a config, a nested folder, and a weights
/// file reached through a symbolic link, as in a Hugging Face snapshot.
fn source_model_folder(parent_directory: &Path) -> PathBuf {
    let source_directory = parent_directory.join("laya-source");
    fs::create_dir_all(source_directory.join("multilingual")).unwrap();
    fs::write(
        source_directory.join("config.json"),
        br#"{"model_type": "laya"}"#,
    )
    .unwrap();
    let weight_bytes: Vec<u8> = (0..3 * COMPARISON_READ_BYTES + 17)
        .map(|byte_index| (byte_index % 251) as u8)
        .collect();
    fs::write(source_directory.join("model.safetensors"), &weight_bytes).unwrap();
    fs::write(
        source_directory.join("multilingual").join("tokenizer.json"),
        ROMEO_AND_JULIET_LINES[0],
    )
    .unwrap();
    let blob_path = parent_directory.join("blob-tokenizer-config");
    fs::write(&blob_path, ROMEO_AND_JULIET_LINES[1]).unwrap();
    std::os::unix::fs::symlink(&blob_path, source_directory.join("tokenizer_config.json")).unwrap();
    source_directory
}

/// The whole source folder, with no other model bundled in it.
fn whole_folder(model_directory: &Path) -> SourceModelFolder<'_> {
    SourceModelFolder {
        model_directory,
        other_model_subdirectories: &[],
    }
}

fn calibration_only(calibration_override: &CalibrationOverride) -> FolderAdditions<'_> {
    FolderAdditions {
        calibration_override: Some(calibration_override),
        head_tuning: None,
        decision_adapters: None,
    }
}

fn scorer_bias_tuning(bias_value: f32) -> HeadTuning {
    HeadTuning {
        tensors: vec![TunedTensor {
            name: "scorer.layers.3.bias".to_owned(),
            shape: vec![1],
            values: vec![bias_value],
        }],
    }
}

fn snapshot_of(directory: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut found_files = Vec::new();
    for source_file in source_files(directory, &[], &[DECISION_CALIBRATION_FILE_NAME]).unwrap() {
        let file_bytes = fs::read(directory.join(&source_file.relative_path)).unwrap();
        found_files.push((source_file.relative_path, file_bytes));
    }
    found_files
}

#[test]
fn should_copy_every_file_unchanged_and_add_the_calibration() {
    let work_directory = tempfile::tempdir().unwrap();
    let source_directory = source_model_folder(work_directory.path());
    let output_directory = work_directory.path().join("laya-calibrated");
    let mut recorded_progress = RecordedProgress::default();
    write_calibrated_model_folder(
        &whole_folder(&source_directory),
        &output_directory,
        calibration_only(&noul_calibration()),
        &mut recorded_progress,
    )
    .unwrap();
    // The snapshot leaves the calibration file out, as the copy pass does,
    // so it compares exactly the files that came from the source.
    assert_eq!(
        snapshot_of(&output_directory),
        snapshot_of(&source_directory)
    );
    let calibration_bytes =
        fs::read(output_directory.join(DECISION_CALIBRATION_FILE_NAME)).unwrap();
    let written_calibration = parse_calibration_document(&calibration_bytes).unwrap();
    assert_eq!(
        written_calibration.temperature_for(DecisionQuestionKind::Noul, 2),
        Some(3.5)
    );
    assert!(
        !fs::symlink_metadata(output_directory.join("tokenizer_config.json"))
            .unwrap()
            .file_type()
            .is_symlink()
    );
    let mut model_byte_count = 0;
    for source_file in
        source_files(&source_directory, &[], &[DECISION_CALIBRATION_FILE_NAME]).unwrap()
    {
        model_byte_count += source_file.byte_count;
    }
    let last_report = recorded_progress.reports.last().copied().unwrap();
    assert_eq!(last_report, (model_byte_count, model_byte_count));
}

#[test]
fn should_leave_the_library_stamp_and_download_bookkeeping_behind() {
    let work_directory = tempfile::tempdir().unwrap();
    let source_directory = source_model_folder(work_directory.path());
    fs::write(
        source_directory.join(".astronomical-library-provenance.json"),
        br#"{"provider_model_id":"cavi-ai/laya-MLX-8bit"}"#,
    )
    .unwrap();
    fs::create_dir_all(source_directory.join(".cache").join("huggingface")).unwrap();
    fs::write(
        source_directory
            .join(".cache")
            .join("huggingface")
            .join("config.json.metadata"),
        ROMEO_AND_JULIET_LINES[2],
    )
    .unwrap();
    let output_directory = work_directory.path().join("laya-calibrated");
    write_calibrated_model_folder(
        &whole_folder(&source_directory),
        &output_directory,
        calibration_only(&noul_calibration()),
        &mut RecordedProgress::default(),
    )
    .unwrap();
    assert!(
        !output_directory
            .join(".astronomical-library-provenance.json")
            .exists()
    );
    assert!(!output_directory.join(".cache").exists());
    assert!(output_directory.join("model.safetensors").exists());
}

#[test]
fn should_leave_the_source_folder_exactly_as_it_was() {
    let work_directory = tempfile::tempdir().unwrap();
    let source_directory = source_model_folder(work_directory.path());
    let source_before = snapshot_of(&source_directory);
    let output_directory = work_directory.path().join("laya-calibrated");
    write_calibrated_model_folder(
        &whole_folder(&source_directory),
        &output_directory,
        calibration_only(&noul_calibration()),
        &mut RecordedProgress::default(),
    )
    .unwrap();
    assert_eq!(snapshot_of(&source_directory), source_before);
}

#[test]
fn should_replace_a_calibration_the_source_already_carried() {
    let work_directory = tempfile::tempdir().unwrap();
    let source_directory = source_model_folder(work_directory.path());
    fs::write(
        source_directory.join(DECISION_CALIBRATION_FILE_NAME),
        b"stale calibration",
    )
    .unwrap();
    let output_directory = work_directory.path().join("laya-recalibrated");
    write_calibrated_model_folder(
        &whole_folder(&source_directory),
        &output_directory,
        calibration_only(&noul_calibration()),
        &mut RecordedProgress::default(),
    )
    .unwrap();
    let written_bytes = fs::read(output_directory.join(DECISION_CALIBRATION_FILE_NAME)).unwrap();
    assert!(parse_calibration_document(&written_bytes).is_ok());
}

#[test]
fn should_refuse_an_output_folder_that_already_exists() {
    let work_directory = tempfile::tempdir().unwrap();
    let source_directory = source_model_folder(work_directory.path());
    let output_directory = work_directory.path().join("taken");
    fs::create_dir(&output_directory).unwrap();
    let write_outcome = write_calibrated_model_folder(
        &whole_folder(&source_directory),
        &output_directory,
        calibration_only(&noul_calibration()),
        &mut RecordedProgress::default(),
    );
    assert!(matches!(
        write_outcome,
        Err(CalibratedFolderError::OutputExists { .. })
    ));
}

#[test]
fn should_refuse_an_output_folder_inside_the_model_folder() {
    let work_directory = tempfile::tempdir().unwrap();
    let source_directory = source_model_folder(work_directory.path());
    let output_directory = source_directory.join("calibrated");
    let write_outcome = write_calibrated_model_folder(
        &whole_folder(&source_directory),
        &output_directory,
        calibration_only(&noul_calibration()),
        &mut RecordedProgress::default(),
    );
    assert!(matches!(
        write_outcome,
        Err(CalibratedFolderError::OutputOverlapsSource)
    ));
    assert!(!output_directory.exists());
}

#[test]
fn should_write_into_folders_that_do_not_exist_yet() {
    let work_directory = tempfile::tempdir().unwrap();
    let source_directory = source_model_folder(work_directory.path());
    let output_directory = work_directory
        .path()
        .join("calibrated")
        .join("verona")
        .join("kev");
    write_calibrated_model_folder(
        &whole_folder(&source_directory),
        &output_directory,
        calibration_only(&noul_calibration()),
        &mut RecordedProgress::default(),
    )
    .unwrap();
    assert!(
        output_directory
            .join(DECISION_CALIBRATION_FILE_NAME)
            .is_file()
    );
}

#[test]
fn should_refuse_a_not_yet_existing_folder_nested_inside_the_model_folder() {
    let work_directory = tempfile::tempdir().unwrap();
    let source_directory = source_model_folder(work_directory.path());
    let output_directory = source_directory.join("later").join("calibrated");
    let write_outcome = write_calibrated_model_folder(
        &whole_folder(&source_directory),
        &output_directory,
        calibration_only(&noul_calibration()),
        &mut RecordedProgress::default(),
    );
    assert!(matches!(
        write_outcome,
        Err(CalibratedFolderError::OutputOverlapsSource)
    ));
    assert!(!source_directory.join("later").exists());
}

#[test]
fn should_leave_no_partial_folder_when_the_source_is_missing() {
    let work_directory = tempfile::tempdir().unwrap();
    let missing_source = work_directory.path().join("no-such-model");
    let output_directory = work_directory.path().join("laya-calibrated");
    let write_outcome = write_calibrated_model_folder(
        &whole_folder(&missing_source),
        &output_directory,
        calibration_only(&noul_calibration()),
        &mut RecordedProgress::default(),
    );
    assert!(matches!(
        write_outcome,
        Err(CalibratedFolderError::Io { .. })
    ));
    assert!(!output_directory.exists());
    assert!(!partial_directory_for(&output_directory).exists());
}

#[test]
fn should_tell_identical_files_from_files_that_differ_in_one_byte_or_in_length() {
    let work_directory = tempfile::tempdir().unwrap();
    let original_path = work_directory.path().join("original");
    let same_path = work_directory.path().join("same");
    let one_byte_off_path = work_directory.path().join("one-byte-off");
    let longer_path = work_directory.path().join("longer");
    let original_bytes = vec![7_u8; COMPARISON_READ_BYTES + 5];
    let mut one_byte_off = original_bytes.clone();
    *one_byte_off.last_mut().unwrap() = 8;
    let mut longer = original_bytes.clone();
    longer.push(7);
    fs::write(&original_path, &original_bytes).unwrap();
    fs::write(&same_path, &original_bytes).unwrap();
    fs::write(&one_byte_off_path, &one_byte_off).unwrap();
    fs::write(&longer_path, &longer).unwrap();
    assert!(files_are_identical(&original_path, &same_path).unwrap());
    assert!(!files_are_identical(&original_path, &one_byte_off_path).unwrap());
    assert!(!files_are_identical(&original_path, &longer_path).unwrap());
}

#[test]
fn should_treat_two_empty_files_as_identical() {
    let work_directory = tempfile::tempdir().unwrap();
    let first_empty = work_directory.path().join("first");
    let second_empty = work_directory.path().join("second");
    fs::write(&first_empty, b"").unwrap();
    fs::write(&second_empty, b"").unwrap();
    assert!(files_are_identical(&first_empty, &second_empty).unwrap());
}

#[test]
fn should_leave_out_another_model_bundled_in_a_subfolder() {
    let work_directory = tempfile::tempdir().unwrap();
    let source_directory = source_model_folder(work_directory.path());
    let bundled_checkpoint = source_directory.join("multilingual-checkpoint");
    fs::create_dir(&bundled_checkpoint).unwrap();
    fs::write(
        bundled_checkpoint.join("config.json"),
        br#"{"model_type": "laya"}"#,
    )
    .unwrap();
    let output_directory = work_directory.path().join("laya-calibrated");
    let other_models = [PathBuf::from("multilingual-checkpoint")];
    write_calibrated_model_folder(
        &SourceModelFolder {
            model_directory: &source_directory,
            other_model_subdirectories: &other_models,
        },
        &output_directory,
        calibration_only(&noul_calibration()),
        &mut RecordedProgress::default(),
    )
    .unwrap();
    assert!(!output_directory.join("multilingual-checkpoint").exists());
    // A plain subfolder that is not a model of its own is still copied.
    assert!(
        output_directory
            .join("multilingual")
            .join("tokenizer.json")
            .exists()
    );
    assert!(bundled_checkpoint.join("config.json").exists());
}

#[test]
fn should_write_a_trained_head_that_reads_back_exactly() {
    let work_directory = tempfile::tempdir().unwrap();
    let source_directory = source_model_folder(work_directory.path());
    let output_directory = work_directory.path().join("laya-trained");
    write_calibrated_model_folder(
        &whole_folder(&source_directory),
        &output_directory,
        FolderAdditions {
            calibration_override: None,
            head_tuning: Some((&scorer_bias_tuning(0.75), TunableHeadFamily::Laya)),
            decision_adapters: None,
        },
        &mut RecordedProgress::default(),
    )
    .unwrap();
    let written = fs::read(output_directory.join(DECISION_HEAD_TUNING_FILE_NAME)).unwrap();
    assert_eq!(
        parse_head_tuning(&written, TunableHeadFamily::Laya).unwrap(),
        scorer_bias_tuning(0.75)
    );
    // Without a calibration to add, none is written: an empty one would not load.
    assert!(
        !output_directory
            .join(DECISION_CALIBRATION_FILE_NAME)
            .exists()
    );
}

#[test]
fn should_keep_an_earlier_trained_head_when_only_calibrating() {
    let work_directory = tempfile::tempdir().unwrap();
    let source_directory = source_model_folder(work_directory.path());
    let earlier_tuning = head_tuning_bytes(&scorer_bias_tuning(0.5)).unwrap();
    fs::write(
        source_directory.join(DECISION_HEAD_TUNING_FILE_NAME),
        &earlier_tuning,
    )
    .unwrap();
    let calibrated_directory = work_directory.path().join("laya-calibrated");
    write_calibrated_model_folder(
        &whole_folder(&source_directory),
        &calibrated_directory,
        calibration_only(&noul_calibration()),
        &mut RecordedProgress::default(),
    )
    .unwrap();
    assert_eq!(
        fs::read(calibrated_directory.join(DECISION_HEAD_TUNING_FILE_NAME)).unwrap(),
        earlier_tuning
    );
    let retrained_directory = work_directory.path().join("laya-retrained");
    write_calibrated_model_folder(
        &whole_folder(&source_directory),
        &retrained_directory,
        FolderAdditions {
            calibration_override: None,
            head_tuning: Some((&scorer_bias_tuning(2.0), TunableHeadFamily::Laya)),
            decision_adapters: None,
        },
        &mut RecordedProgress::default(),
    )
    .unwrap();
    let retrained = fs::read(retrained_directory.join(DECISION_HEAD_TUNING_FILE_NAME)).unwrap();
    assert_eq!(
        parse_head_tuning(&retrained, TunableHeadFamily::Laya).unwrap(),
        scorer_bias_tuning(2.0)
    );
}

#[test]
fn should_refuse_a_trained_head_that_cannot_be_written_and_leave_nothing() {
    let work_directory = tempfile::tempdir().unwrap();
    let source_directory = source_model_folder(work_directory.path());
    let output_directory = work_directory.path().join("laya-broken");
    let short_tensor = HeadTuning {
        tensors: vec![TunedTensor {
            name: "scorer.layers.1.weight".to_owned(),
            shape: vec![2, 2],
            values: vec![1.0],
        }],
    };
    let write_outcome = write_calibrated_model_folder(
        &whole_folder(&source_directory),
        &output_directory,
        FolderAdditions {
            calibration_override: None,
            head_tuning: Some((&short_tensor, TunableHeadFamily::Laya)),
            decision_adapters: None,
        },
        &mut RecordedProgress::default(),
    );
    assert!(matches!(
        write_outcome,
        Err(CalibratedFolderError::TuningUnwritable(_))
    ));
    assert!(!output_directory.exists());
    assert!(!partial_directory_for(&output_directory).exists());
}

/// Two add-ons, given out of name order, as a caller might build them.
fn two_layer_adapters() -> DecisionAdapters {
    DecisionAdapters {
        adapters: vec![
            LinearAdapter {
                target: "head.layers.0.linear1".to_owned(),
                rank: 1,
                input_width: 2,
                output_width: 2,
                down_values: vec![0.5, -0.5],
                up_values: vec![1.0e-3, -2.0e-3],
            },
            LinearAdapter {
                target: "encoder.layers.0.attn.Wo".to_owned(),
                rank: 1,
                input_width: 2,
                output_width: 1,
                down_values: vec![0.25, 0.75],
                up_values: vec![0.0],
            },
        ],
    }
}

#[test]
fn should_write_add_ons_beside_the_trained_head_that_read_back_exactly() {
    let work_directory = tempfile::tempdir().unwrap();
    let source_directory = source_model_folder(work_directory.path());
    let output_directory = work_directory.path().join("laya-whole");

    write_calibrated_model_folder(
        &whole_folder(&source_directory),
        &output_directory,
        FolderAdditions {
            calibration_override: None,
            head_tuning: Some((&scorer_bias_tuning(0.75), TunableHeadFamily::Laya)),
            decision_adapters: Some(&two_layer_adapters()),
        },
        &mut RecordedProgress::default(),
    )
    .unwrap();

    let written = fs::read(output_directory.join(DECISION_ADAPTERS_FILE_NAME)).unwrap();
    let read_back = parse_decision_adapters(&written).unwrap();
    assert_eq!(read_back.adapters.len(), 2);
    assert_eq!(read_back.adapters[0].target, "encoder.layers.0.attn.Wo");
    assert_eq!(read_back.adapters[1].up_values, vec![1.0e-3, -2.0e-3]);
    assert!(
        output_directory
            .join(DECISION_HEAD_TUNING_FILE_NAME)
            .exists()
    );
}

#[test]
fn should_replace_earlier_add_ons_and_keep_them_when_only_calibrating() {
    let work_directory = tempfile::tempdir().unwrap();
    let source_directory = source_model_folder(work_directory.path());
    let earlier_adapters = decision_adapters_bytes(&two_layer_adapters()).unwrap();
    fs::write(
        source_directory.join(DECISION_ADAPTERS_FILE_NAME),
        &earlier_adapters,
    )
    .unwrap();
    let calibrated_directory = work_directory.path().join("laya-calibrated");
    write_calibrated_model_folder(
        &whole_folder(&source_directory),
        &calibrated_directory,
        calibration_only(&noul_calibration()),
        &mut RecordedProgress::default(),
    )
    .unwrap();
    assert_eq!(
        fs::read(calibrated_directory.join(DECISION_ADAPTERS_FILE_NAME)).unwrap(),
        earlier_adapters
    );

    let mut retrained_adapters = two_layer_adapters();
    retrained_adapters.adapters.truncate(1);
    let retrained_directory = work_directory.path().join("laya-retrained");
    write_calibrated_model_folder(
        &whole_folder(&source_directory),
        &retrained_directory,
        FolderAdditions {
            calibration_override: None,
            head_tuning: None,
            decision_adapters: Some(&retrained_adapters),
        },
        &mut RecordedProgress::default(),
    )
    .unwrap();
    let written = fs::read(retrained_directory.join(DECISION_ADAPTERS_FILE_NAME)).unwrap();
    assert_eq!(
        parse_decision_adapters(&written).unwrap(),
        retrained_adapters
    );
}
