use std::fs;

use super::*;

fn fixture_with_files() -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("config.json"), b"0123456789").unwrap();
    fs::create_dir(directory.path().join("optiq")).unwrap();
    fs::write(directory.path().join("optiq/tower.safetensors"), b"xy").unwrap();
    fs::create_dir(directory.path().join(".cache")).unwrap();
    fs::write(directory.path().join(".cache/skip.safetensors"), b"z").unwrap();
    directory
}

#[test]
fn should_list_nested_files_and_skip_hidden_bookkeeping() {
    let directory = fixture_with_files();
    let source = LocalDirectorySource::open(directory.path()).unwrap();
    let mut relative_paths = Vec::new();
    for artifact_file in &source.listing().files {
        relative_paths.push(artifact_file.relative_path.as_str());
    }
    assert_eq!(relative_paths, ["config.json", "optiq/tower.safetensors"]);
}

#[test]
fn should_not_descend_past_the_depth_limit() {
    let directory = tempfile::tempdir().unwrap();
    fs::create_dir_all(directory.path().join("one/two/three")).unwrap();
    fs::write(directory.path().join("one/two/kept.txt"), b"k").unwrap();
    fs::write(directory.path().join("one/two/three/deep.txt"), b"d").unwrap();
    let source = LocalDirectorySource::open(directory.path()).unwrap();
    assert_eq!(source.listing().files.len(), 1);
    assert_eq!(source.listing().files[0].relative_path, "one/two/kept.txt");
}

#[test]
fn should_read_an_exact_byte_range() {
    let directory = fixture_with_files();
    let source = LocalDirectorySource::open(directory.path()).unwrap();
    assert_eq!(
        source.read_byte_range("config.json", 3, 4).unwrap(),
        b"3456"
    );
}

#[test]
fn should_fail_when_a_range_runs_past_the_end_of_the_file() {
    let directory = fixture_with_files();
    let source = LocalDirectorySource::open(directory.path()).unwrap();
    assert!(source.read_byte_range("config.json", 8, 4).is_err());
}

#[test]
fn should_refuse_a_whole_file_over_its_limit_and_accept_one_at_it() {
    let directory = fixture_with_files();
    let source = LocalDirectorySource::open(directory.path()).unwrap();
    assert!(source.read_whole_file("config.json", 10).is_ok());
    assert!(matches!(
        source.read_whole_file("config.json", 9),
        Err(VisionTowerError::FileTooLarge { .. })
    ));
}

#[test]
fn should_fail_to_open_a_missing_directory() {
    let directory = tempfile::tempdir().unwrap();
    assert!(LocalDirectorySource::open(&directory.path().join("absent")).is_err());
}

#[test]
fn should_recover_provenance_from_a_cache_snapshot_path() {
    let directory = tempfile::tempdir().unwrap();
    let snapshot = directory
        .path()
        .join("models--example-org--Model-1.5-4bit/snapshots/abc123");
    fs::create_dir_all(&snapshot).unwrap();
    let source = LocalDirectorySource::open(&snapshot).unwrap();
    assert_eq!(
        source.listing().origin_repository.as_deref(),
        Some("example-org/Model-1.5-4bit")
    );
    assert_eq!(source.listing().pinned_revision.as_deref(), Some("abc123"));
}

#[test]
fn should_not_invent_provenance_for_an_ordinary_folder() {
    let directory = tempfile::tempdir().unwrap();
    let source = LocalDirectorySource::open(directory.path()).unwrap();
    assert_eq!(source.listing().origin_repository, None);
    assert_eq!(source.listing().pinned_revision, None);

    let lookalike = directory
        .path()
        .join("models--no-separator/snapshots/abc123");
    fs::create_dir_all(&lookalike).unwrap();
    let source = LocalDirectorySource::open(&lookalike).unwrap();
    assert_eq!(source.listing().origin_repository, None);
}
