//! Crash-safety contracts for the instance-owned model download job.

use std::fs;

use astronomical_supervisor::{
    DownloadJob, DownloadJobError, DownloadJobState, DownloadJobStore, DownloadJobStoreError,
};
use tempfile::TempDir;

const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

#[test]
fn should_report_idle_without_creating_the_instance_models_directory() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let models_directory = test_directory.path().join("models");
    let job_store = DownloadJobStore::new(models_directory.clone());

    assert!(
        job_store
            .load()
            .expect("idle load should succeed")
            .is_none()
    );
    assert!(!models_directory.exists());
}

#[test]
fn should_persist_one_strict_job_atomically_and_reload_it() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let models_directory = test_directory.path().join("models");
    let job_store = DownloadJobStore::new(models_directory.clone());
    let expected_job = parse_job(&job_json("downloading", 4, 12, 4, "null", 100));

    job_store
        .create(&expected_job)
        .expect("valid job should persist");

    let loaded_job = job_store
        .load()
        .expect("persisted job should load")
        .expect("persisted job should be present");
    assert_eq!(loaded_job, expected_job);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(job_store.job_file_path())
                .expect("job metadata should be readable")
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
    let temporary_entries = fs::read_dir(&models_directory)
        .expect("models directory should be readable")
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().contains(".tmp."))
        .count();
    assert_eq!(temporary_entries, 0);
}

#[test]
fn should_recover_every_interrupted_state_as_durably_paused_from_actual_file_lengths() {
    for (interrupted_job_json, staged_bytes, expected_completed_bytes) in [
        (premanifest_job_json("checking_disk", "null", 100), None, 0),
        (
            premanifest_job_json("fetching_manifest", "null", 100),
            None,
            0,
        ),
        (
            job_json("downloading", 4, 12, 4, "null", 100),
            Some(&b"Romeo"[..]),
            5,
        ),
        (
            job_json("verifying", 12, 12, 12, "null", 100),
            Some(&b"RomeoJuliet!"[..]),
            12,
        ),
    ] {
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let models_directory = test_directory.path().join("models");
        let job_store = DownloadJobStore::new(models_directory.clone());
        let interrupted_job = parse_job(&interrupted_job_json);
        job_store
            .create(&interrupted_job)
            .expect("interrupted job should persist");
        if let Some(staged_bytes) = staged_bytes {
            let staged_file = models_directory
                .join(".incomplete/astronomical-test/example-qwen/weights/model.safetensors");
            fs::create_dir_all(
                staged_file
                    .parent()
                    .expect("staged file should have a parent"),
            )
            .expect("staging directory should be created");
            fs::write(&staged_file, staged_bytes).expect("staged fixture should be written");
        }

        let recovered_job = job_store
            .load_recovering_interrupted_job(200)
            .expect("interrupted job should recover")
            .expect("recovered job should be present");

        assert_eq!(recovered_job.state(), DownloadJobState::Paused);
        assert_eq!(recovered_job.bytes_completed(), expected_completed_bytes);
        if let Some(recovered_file) = recovered_job.files().first() {
            assert_eq!(recovered_file.bytes_on_disk(), expected_completed_bytes);
        }
        assert_eq!(recovered_job.current_file_relative_path(), None);
        assert_eq!(recovered_job.updated_at_unix_millis(), 200);
        assert_eq!(
            job_store
                .load()
                .expect("recovered job should reload")
                .expect("recovered job should remain present"),
            recovered_job
        );
    }
}

#[test]
fn should_reconcile_paused_and_failed_jobs_without_changing_their_state() {
    for (state, error_code) in [("paused", "null"), ("failed", "\"checksum_mismatch\"")] {
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let models_directory = test_directory.path().join("models");
        let job_store = DownloadJobStore::new(models_directory.clone());
        job_store
            .create(&parse_job(&job_json(state, 4, 12, 4, error_code, 100)))
            .expect("job should persist");

        let recovered_job = job_store
            .load_recovering_interrupted_job(200)
            .expect("job should reconcile")
            .expect("job should remain present");

        assert_eq!(recovered_job.state().as_str(), state);
        assert_eq!(recovered_job.bytes_completed(), 0);
        assert_eq!(recovered_job.updated_at_unix_millis(), 200);
        if state == "failed" {
            assert_eq!(
                recovered_job.error_code(),
                Some(astronomical_supervisor::DownloadJobPublicErrorCode::ChecksumMismatch)
            );
        }
        assert_eq!(
            job_store
                .load()
                .expect("reconciled job should reload")
                .expect("reconciled job should remain present"),
            recovered_job
        );
    }
}

#[test]
fn should_pause_the_current_job_only_after_reconciling_and_persisting_progress() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let models_directory = test_directory.path().join("models");
    let job_store = DownloadJobStore::new(models_directory.clone());
    job_store
        .create(&parse_job(&job_json("downloading", 4, 12, 4, "null", 100)))
        .expect("job should persist");
    let staged_file = models_directory
        .join(".incomplete/astronomical-test/example-qwen/weights/model.safetensors");
    fs::create_dir_all(
        staged_file
            .parent()
            .expect("staged file should have a parent"),
    )
    .expect("staging directory should be created");
    fs::write(&staged_file, b"Juliet").expect("staged fixture should be written");

    let paused_job = job_store
        .pause_current_job(300)
        .expect("pause should succeed")
        .expect("paused job should be returned");

    assert_eq!(paused_job.state(), DownloadJobState::Paused);
    assert_eq!(paused_job.bytes_completed(), 6);
    assert_eq!(
        job_store
            .load()
            .expect("paused job should reload")
            .expect("paused job should remain present"),
        paused_job
    );
}

#[test]
fn should_destructively_cancel_ten_times_without_touching_a_published_destination() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let models_directory = test_directory.path().join("models");
    let job_store = DownloadJobStore::new(models_directory.clone());
    job_store
        .create(&parse_job(&job_json("paused", 0, 12, 0, "null", 100)))
        .expect("job should persist");
    let staging_directory = models_directory.join(".incomplete/astronomical-test/example-qwen");
    fs::create_dir_all(&staging_directory).expect("staging directory should be created");
    fs::write(staging_directory.join("partial"), b"Romeo")
        .expect("partial fixture should be written");
    let published_directory = models_directory.join("astronomical-test/example-qwen");
    fs::create_dir_all(&published_directory).expect("published directory should be created");
    fs::write(published_directory.join("config.json"), b"Juliet")
        .expect("published fixture should be written");

    assert!(
        job_store
            .cancel_current_job()
            .expect("first cancel should succeed")
    );
    for _repetition in 1..10 {
        assert!(
            !job_store
                .cancel_current_job()
                .expect("repeated cancel should remain successful")
        );
    }

    assert!(!staging_directory.exists());
    assert!(!job_store.job_file_path().exists());
    assert_eq!(
        fs::read(published_directory.join("config.json"))
            .expect("published destination should remain readable"),
        b"Juliet"
    );
}

#[test]
fn should_reject_malformed_or_internally_inconsistent_job_documents() {
    let invalid_documents = [
        job_json("downloading", 5, 12, 4, "null", 100),
        job_json("downloading", 4, 11, 4, "null", 100),
        job_json("failed", 4, 12, 4, "null", 100),
        job_json("paused", 4, 12, 4, "\"checksum_mismatch\"", 100),
        job_json("unknown", 4, 12, 4, "null", 100),
        job_json("downloading", 4, 12, 13, "null", 100),
        job_json("downloading", 4, 12, 4, "null", 100).replace(
            "weights/model.safetensors",
            "../published/model.safetensors",
        ),
        job_json("downloading", 4, 12, 4, "null", 100).replace(DIGEST, "ABC"),
        job_json("downloading", 4, 12, 4, "null", 100)
            .replace("\"updated_at\":100", "\"updated_at\":100,\"extra\":true"),
    ];

    for invalid_document in invalid_documents {
        assert!(DownloadJob::parse_json(&invalid_document).is_err());
    }
}

#[test]
fn should_reject_duplicate_case_colliding_file_paths_and_unknown_error_codes() {
    let duplicate_file = format!(
        "{{\"relative_path\":\"Weights/Model.safetensors\",\"expected_bytes\":12,\"expected_digest\":{{\"algorithm\":\"sha256\",\"hex\":\"{DIGEST}\"}},\"bytes_on_disk\":0}}"
    );
    let duplicate_document = job_json("paused", 0, 24, 0, "null", 100).replace(
        "}],\"error_code\"",
        &format!("}},{duplicate_file}],\"error_code\""),
    );
    assert!(matches!(
        DownloadJob::parse_json(&duplicate_document),
        Err(DownloadJobError::DuplicateFilePath)
    ));

    let unknown_error = job_json("failed", 4, 12, 4, "\"private_path_error\"", 100);
    assert!(matches!(
        DownloadJob::parse_json(&unknown_error),
        Err(DownloadJobError::Parse(_))
    ));
}

#[test]
fn should_reject_file_and_descendant_manifest_paths() {
    let descendant_file = format!(
        "{{\"relative_path\":\"weights/model.safetensors/part\",\"expected_bytes\":12,\"expected_digest\":{{\"algorithm\":\"sha256\",\"hex\":\"{DIGEST}\"}},\"bytes_on_disk\":0}}"
    );
    let conflicting_document = job_json("paused", 0, 24, 0, "null", 100).replace(
        "}],\"error_code\"",
        &format!("}},{descendant_file}],\"error_code\""),
    );

    assert!(matches!(
        DownloadJob::parse_json(&conflicting_document),
        Err(DownloadJobError::FilePathHierarchyConflict)
    ));
}

#[test]
fn should_reject_noncanonical_or_non_ascii_manifest_paths() {
    for unsafe_path in [
        "weights//model.safetensors",
        "weights/./model.safetensors",
        "weights/model.safetensors/",
        "weights/modél.safetensors",
    ] {
        let unsafe_document = job_json("paused", 0, 12, 0, "null", 100)
            .replace("weights/model.safetensors", unsafe_path);
        assert!(matches!(
            DownloadJob::parse_json(&unsafe_document),
            Err(DownloadJobError::UnsafeRelativePath { .. })
        ));
    }
}

#[test]
fn should_reject_a_second_job_without_replacing_the_first_job() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let job_store = DownloadJobStore::new(test_directory.path().join("models"));
    let first_job = parse_job(&job_json("paused", 0, 12, 0, "null", 100));
    let second_job = parse_job(
        &job_json("paused", 0, 12, 0, "null", 200).replace("example-qwen", "another-qwen"),
    );
    job_store
        .create(&first_job)
        .expect("first job should persist");

    assert!(matches!(
        job_store.create(&second_job),
        Err(DownloadJobStoreError::JobAlreadyExists)
    ));
    assert_eq!(
        job_store
            .load()
            .expect("first job should reload")
            .expect("first job should remain present"),
        first_job
    );
}

#[test]
fn should_fail_closed_without_mutating_an_interrupted_publishing_job() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let job_store = DownloadJobStore::new(test_directory.path().join("models"));
    let publishing_job = parse_job(&job_json("publishing", 12, 12, 12, "null", 100));
    job_store
        .create(&publishing_job)
        .expect("publishing job should persist");

    assert!(matches!(
        job_store.load_recovering_interrupted_job(200),
        Err(DownloadJobStoreError::PublishingRecoveryRequired)
    ));
    assert_eq!(
        job_store
            .load()
            .expect("publishing job should reload")
            .expect("publishing job should remain present"),
        publishing_job
    );
}

#[test]
fn should_reject_an_oversized_job_file_before_allocating_its_complete_contents() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let models_directory = test_directory.path().join("models");
    fs::create_dir(&models_directory).expect("models directory should be created");
    fs::write(
        models_directory.join(".download-job.json"),
        vec![b' '; 8_000_001],
    )
    .expect("oversized job should be written");
    let job_store = DownloadJobStore::new(models_directory);

    assert!(matches!(
        job_store.load(),
        Err(DownloadJobStoreError::JobDocumentTooLarge)
    ));
}

#[test]
fn should_reject_oversized_job_metadata_before_json_parsing() {
    let oversized_document = " ".repeat(8_000_001);
    assert!(matches!(
        DownloadJob::parse_json(&oversized_document),
        Err(DownloadJobError::DocumentTooLarge)
    ));
}

#[cfg(unix)]
#[test]
fn should_fail_closed_for_symlinked_job_staging_or_staged_file_paths() {
    use std::os::unix::fs::symlink;

    let job_symlink_directory = TempDir::new().expect("temporary directory should be available");
    let external_job = job_symlink_directory.path().join("external-job.json");
    fs::write(&external_job, job_json("paused", 0, 12, 0, "null", 100))
        .expect("external job should be written");
    let models_directory = job_symlink_directory.path().join("models");
    fs::create_dir(&models_directory).expect("models directory should be created");
    symlink(&external_job, models_directory.join(".download-job.json"))
        .expect("job symlink should be created");
    let job_store = DownloadJobStore::new(models_directory);
    assert!(matches!(
        job_store.load(),
        Err(DownloadJobStoreError::UnsafeFilesystemObject { .. })
    ));

    for symlink_target in ["staging", "file"] {
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let models_directory = test_directory.path().join("models");
        let job_store = DownloadJobStore::new(models_directory.clone());
        job_store
            .create(&parse_job(&job_json("downloading", 0, 12, 0, "null", 100)))
            .expect("job should persist");
        let external_path = test_directory.path().join("external");
        if symlink_target == "staging" {
            fs::create_dir(&external_path).expect("external directory should be created");
            fs::create_dir(models_directory.join(".incomplete"))
                .expect("incomplete directory should be created");
            symlink(
                &external_path,
                models_directory.join(".incomplete/astronomical-test"),
            )
            .expect("staging symlink should be created");
        } else {
            fs::write(&external_path, b"Romeo").expect("external file should be written");
            let staged_parent =
                models_directory.join(".incomplete/astronomical-test/example-qwen/weights");
            fs::create_dir_all(&staged_parent).expect("staging parent should be created");
            symlink(&external_path, staged_parent.join("model.safetensors"))
                .expect("staged file symlink should be created");
        }

        assert!(matches!(
            job_store.load_recovering_interrupted_job(200),
            Err(DownloadJobStoreError::UnsafeFilesystemObject { .. })
        ));
    }
}

#[test]
fn should_fail_recovery_when_staged_bytes_exceed_the_manifest_expectation() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let models_directory = test_directory.path().join("models");
    let job_store = DownloadJobStore::new(models_directory.clone());
    job_store
        .create(&parse_job(&job_json("downloading", 0, 12, 0, "null", 100)))
        .expect("job should persist");
    let staged_file = models_directory
        .join(".incomplete/astronomical-test/example-qwen/weights/model.safetensors");
    fs::create_dir_all(
        staged_file
            .parent()
            .expect("staged file should have a parent"),
    )
    .expect("staging directory should be created");
    fs::write(&staged_file, b"Romeo & Juliet").expect("oversized staged fixture should be written");

    assert!(matches!(
        job_store.load_recovering_interrupted_job(200),
        Err(DownloadJobStoreError::StagedFileTooLarge { .. })
    ));
}

fn parse_job(job_document: &str) -> DownloadJob {
    DownloadJob::parse_json(job_document).expect("fixture job should be valid")
}

fn job_json(
    state: &str,
    bytes_completed: u64,
    bytes_total: u64,
    bytes_on_disk: u64,
    error_code: &str,
    updated_at_unix_millis: u64,
) -> String {
    let current_file_relative_path = if state == "downloading" {
        "\"weights/model.safetensors\""
    } else {
        "null"
    };
    format!(
        "{{\
            \"huggingface_id\":\"astronomical-test/example-qwen\",\
            \"revision\":\"{REVISION}\",\
            \"state\":\"{state}\",\
            \"bytes_completed\":{bytes_completed},\
            \"bytes_total\":{bytes_total},\
            \"current_file_relative_path\":{current_file_relative_path},\
            \"files\":[{{\
                \"relative_path\":\"weights/model.safetensors\",\
                \"expected_bytes\":12,\
                \"expected_digest\":{{\"algorithm\":\"sha256\",\"hex\":\"{DIGEST}\"}},\
                \"bytes_on_disk\":{bytes_on_disk}\
            }}],\
            \"error_code\":{error_code},\
            \"updated_at\":{updated_at_unix_millis}\
        }}"
    )
}

fn premanifest_job_json(state: &str, error_code: &str, updated_at_unix_millis: u64) -> String {
    format!(
        "{{\
            \"huggingface_id\":\"astronomical-test/example-qwen\",\
            \"revision\":\"{REVISION}\",\
            \"state\":\"{state}\",\
            \"bytes_completed\":0,\
            \"bytes_total\":12,\
            \"current_file_relative_path\":null,\
            \"files\":[],\
            \"error_code\":{error_code},\
            \"updated_at\":{updated_at_unix_millis}\
        }}"
    )
}
