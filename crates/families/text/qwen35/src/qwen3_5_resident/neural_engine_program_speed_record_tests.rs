use std::fs;

use super::{PROGRAM_SPEED_RECORD_FILE_NAME, ProgramSpeedRecord};
use crate::qwen3_5_resident::neural_engine_chunk_comparison::ChunkComparisonVerdict;

const MEASURED_FOR: &str = "macOS build; method 1";

#[test]
fn should_return_a_recorded_gain_as_its_verdict() {
    let set_directory = tempfile::tempdir().expect("a temporary directory");
    let record = ProgramSpeedRecord::new(set_directory.path(), MEASURED_FOR.to_owned());
    record.record_gain(1.18);
    assert_eq!(
        record.recorded_verdict(),
        Some(ChunkComparisonVerdict::Worthwhile { gain: 1.18 })
    );
    record.record_gain(0.91);
    assert_eq!(
        record.recorded_verdict(),
        Some(ChunkComparisonVerdict::NotWorthwhile { gain: 0.91 })
    );
}

#[test]
fn should_have_no_verdict_before_any_is_recorded() {
    let set_directory = tempfile::tempdir().expect("a temporary directory");
    let record = ProgramSpeedRecord::new(set_directory.path(), MEASURED_FOR.to_owned());
    assert_eq!(record.recorded_verdict(), None);
}

#[test]
fn should_ask_again_after_the_environment_changes() {
    let set_directory = tempfile::tempdir().expect("a temporary directory");
    ProgramSpeedRecord::new(set_directory.path(), MEASURED_FOR.to_owned()).record_gain(0.91);
    let after_update = ProgramSpeedRecord::new(set_directory.path(), "another build".to_owned());
    assert_eq!(after_update.recorded_verdict(), None);
}

#[test]
fn should_treat_a_damaged_record_as_no_record() {
    let set_directory = tempfile::tempdir().expect("a temporary directory");
    let record = ProgramSpeedRecord::new(set_directory.path(), MEASURED_FOR.to_owned());
    let record_path = set_directory.path().join(PROGRAM_SPEED_RECORD_FILE_NAME);
    for damaged_gain in ["not a number", "0", "-1.2", "inf", "NaN", ""] {
        fs::write(&record_path, format!("{MEASURED_FOR}\n{damaged_gain}\n")).expect("it writes");
        assert_eq!(record.recorded_verdict(), None, "{damaged_gain}");
    }
    fs::write(&record_path, MEASURED_FOR).expect("it writes");
    assert_eq!(record.recorded_verdict(), None);
}

#[test]
fn should_survive_a_folder_that_cannot_be_written() {
    let set_directory = tempfile::tempdir().expect("a temporary directory");
    let missing_folder = set_directory.path().join("missing");
    let record = ProgramSpeedRecord::new(&missing_folder, MEASURED_FOR.to_owned());
    record.record_gain(1.2);
    assert_eq!(record.recorded_verdict(), None);
}
