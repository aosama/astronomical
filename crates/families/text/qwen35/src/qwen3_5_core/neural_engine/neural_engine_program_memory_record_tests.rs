use std::fs;

use super::{
    PROGRAM_MEMORY_RECORD_FILE_NAME, record_program_memory_bytes, recorded_program_memory_bytes,
};

const ENVIRONMENT: &str = "macOS 27.0.1 (26A434) on Mac99,1";
const WEIGHTS: u64 = 1000;

fn set_directory() -> tempfile::TempDir {
    tempfile::tempdir().expect("a scratch directory for the set")
}

/// Two readings that agree, which is what makes a record count.
fn record_agreed(set_directory: &std::path::Path, wired_bytes: u64) {
    record_program_memory_bytes(set_directory, ENVIRONMENT, WEIGHTS, wired_bytes);
    record_program_memory_bytes(set_directory, ENVIRONMENT, WEIGHTS, wired_bytes);
}

fn recorded(set_directory: &std::path::Path) -> Option<u64> {
    recorded_program_memory_bytes(set_directory, ENVIRONMENT, WEIGHTS)
}

#[test]
fn should_not_trust_a_first_reading() {
    let set_directory = set_directory();

    record_program_memory_bytes(set_directory.path(), ENVIRONMENT, WEIGHTS, 1300);

    assert_eq!(recorded(set_directory.path()), None);
}

#[test]
fn should_trust_two_readings_that_agree_and_keep_the_lower() {
    let set_directory = set_directory();

    record_program_memory_bytes(set_directory.path(), ENVIRONMENT, WEIGHTS, 1310);
    record_program_memory_bytes(set_directory.path(), ENVIRONMENT, WEIGHTS, 1300);

    assert_eq!(recorded(set_directory.path()), Some(1300));
}

#[test]
fn should_agree_at_exactly_the_largest_disagreement() {
    let set_directory = set_directory();

    record_program_memory_bytes(set_directory.path(), ENVIRONMENT, WEIGHTS, 1700);
    record_program_memory_bytes(set_directory.path(), ENVIRONMENT, WEIGHTS, 1600);

    assert_eq!(recorded(set_directory.path()), Some(1600));
}

#[test]
fn should_not_trust_readings_just_past_the_largest_disagreement() {
    let set_directory = set_directory();

    record_program_memory_bytes(set_directory.path(), ENVIRONMENT, WEIGHTS, 1701);
    record_program_memory_bytes(set_directory.path(), ENVIRONMENT, WEIGHTS, 1600);

    assert_eq!(recorded(set_directory.path()), None);
}

#[test]
fn should_recover_from_an_inflated_reading_once_two_clean_ones_agree() {
    let set_directory = set_directory();

    record_program_memory_bytes(set_directory.path(), ENVIRONMENT, WEIGHTS, 9000);
    record_program_memory_bytes(set_directory.path(), ENVIRONMENT, WEIGHTS, 1300);
    assert_eq!(recorded(set_directory.path()), None);

    record_program_memory_bytes(set_directory.path(), ENVIRONMENT, WEIGHTS, 1310);
    assert_eq!(recorded(set_directory.path()), Some(1300));
}

#[test]
fn should_accept_a_reading_equal_to_the_weights() {
    let set_directory = set_directory();

    record_agreed(set_directory.path(), WEIGHTS);

    assert_eq!(recorded(set_directory.path()), Some(WEIGHTS));
}

#[test]
fn should_leave_no_incomplete_file_behind() {
    let set_directory = set_directory();

    record_agreed(set_directory.path(), 1300);

    let file_count = fs::read_dir(set_directory.path())
        .expect("the set directory reads")
        .count();
    assert_eq!(file_count, 1);
}

#[test]
fn should_measure_again_with_no_record() {
    let set_directory = set_directory();

    assert_eq!(recorded(set_directory.path()), None);
}

#[test]
fn should_measure_again_on_another_mac_or_macos_build() {
    let set_directory = set_directory();
    record_agreed(set_directory.path(), 1300);

    assert_eq!(
        recorded_program_memory_bytes(
            set_directory.path(),
            "macOS 28.0 (27A1) on Mac99,1",
            WEIGHTS
        ),
        None
    );
}

#[test]
fn should_measure_again_when_the_set_covers_other_weights() {
    let set_directory = set_directory();
    record_agreed(set_directory.path(), 1300);

    assert_eq!(
        recorded_program_memory_bytes(set_directory.path(), ENVIRONMENT, WEIGHTS - 1),
        None
    );
}

#[test]
fn should_measure_again_when_a_record_is_below_the_weights() {
    let set_directory = set_directory();
    record_agreed(set_directory.path(), WEIGHTS - 1);

    assert_eq!(recorded(set_directory.path()), None);
}

#[test]
fn should_measure_again_when_a_record_is_damaged() {
    let set_directory = set_directory();
    for damaged_record in [
        String::new(),
        "not a record".to_owned(),
        format!("{ENVIRONMENT}\n{WEIGHTS}\n"),
        format!("{ENVIRONMENT}\n{WEIGHTS}\nthirteen\nagreed\n"),
        format!("{ENVIRONMENT}\n{WEIGHTS}\n1300\nmaybe\n"),
    ] {
        fs::write(
            set_directory.path().join(PROGRAM_MEMORY_RECORD_FILE_NAME),
            damaged_record,
        )
        .expect("the damaged record writes");

        assert_eq!(recorded(set_directory.path()), None);
    }
}

#[test]
fn should_not_fail_when_the_set_directory_is_gone() {
    let set_directory = set_directory();
    let missing_directory = set_directory.path().join("gone");

    record_program_memory_bytes(&missing_directory, ENVIRONMENT, WEIGHTS, 1300);

    assert_eq!(recorded(&missing_directory), None);
}
