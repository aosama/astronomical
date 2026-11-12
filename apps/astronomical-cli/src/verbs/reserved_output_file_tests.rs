use super::{OutputReservationError, ReservedOutputFile, timestamped_file_name};

// Positive: a claimed path is filled and its absolute path returned.
#[test]
fn should_claim_then_fill_a_new_file() {
    let output_directory = tempfile::tempdir().expect("a temporary directory");
    let output_path = output_directory.path().join("romeo.wav");
    let reserved_file = ReservedOutputFile::claim(&output_path).expect("a new path is claimed");
    let saved_path = reserved_file.fill(b"RIFF").expect("the claim is written");
    assert_eq!(
        std::fs::read(&saved_path).expect("the file is readable"),
        b"RIFF"
    );
}

// Negative: an existing file is never overwritten.
#[test]
fn should_refuse_an_existing_file() {
    let output_directory = tempfile::tempdir().expect("a temporary directory");
    let output_path = output_directory.path().join("juliet.wav");
    std::fs::write(&output_path, b"keep me").expect("an existing file");
    assert!(matches!(
        ReservedOutputFile::claim(&output_path),
        Err(OutputReservationError::AlreadyExists)
    ));
    assert_eq!(std::fs::read(&output_path).expect("readable"), b"keep me");
}

// Edge: a discarded claim leaves nothing behind, and a missing folder fails
// at once.
#[test]
fn should_leave_nothing_when_discarded_and_refuse_a_missing_folder() {
    let output_directory = tempfile::tempdir().expect("a temporary directory");
    let output_path = output_directory.path().join("mercutio.wav");
    ReservedOutputFile::claim(&output_path)
        .expect("a new path is claimed")
        .discard();
    assert!(!output_path.exists());
    assert!(matches!(
        ReservedOutputFile::claim(&output_directory.path().join("no/such/folder.wav")),
        Err(OutputReservationError::Unwritable { .. })
    ));
    assert!(timestamped_file_name("speech", "wav").starts_with("astronomical-speech-"));
}
