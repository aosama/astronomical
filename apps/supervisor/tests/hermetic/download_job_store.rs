//! Concurrency and destructive-boundary contracts for the durable job store.

use std::fs;
use std::sync::{Arc, Barrier};
use std::time::Duration;

use astronomical_supervisor::{DownloadJob, DownloadJobStore, DownloadJobStoreError};
use tempfile::TempDir;

const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const GIT_BLOB_SHA1: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

#[test]
fn should_retain_an_ordinary_file_git_blob_sha1_in_durable_state() {
    let sha1_job_document = format!(
        "{{\"huggingface_id\":\"astronomical-test/example-qwen\",\"revision\":\"{REVISION}\",\"state\":\"paused\",\"bytes_completed\":0,\"bytes_total\":12,\"current_file_relative_path\":null,\"files\":[{{\"relative_path\":\"config.json\",\"expected_bytes\":12,\"expected_digest\":{{\"algorithm\":\"git_blob_sha1\",\"hex\":\"{GIT_BLOB_SHA1}\"}},\"bytes_on_disk\":0}}],\"error_code\":null,\"updated_at\":100}}"
    );

    let download_job = DownloadJob::parse_json(&sha1_job_document)
        .expect("ordinary Hub file SHA-1 evidence should be durable");

    assert_eq!(
        download_job.files()[0].expected_digest().hex(),
        GIT_BLOB_SHA1
    );
}

#[test]
fn should_replace_only_the_current_matching_download_identity() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let job_store = DownloadJobStore::new(test_directory.path().join("models"));
    let mut current_job =
        DownloadJob::new_checking_disk("astronomical-test/example-qwen", REVISION, 12, 100)
            .expect("initial preflight job should be valid");
    job_store
        .create(&current_job)
        .expect("initial job should persist");
    current_job
        .mark_fetching_manifest(100)
        .expect("successful initial preflight should advance to manifest retrieval");
    job_store
        .replace_current(&current_job)
        .expect("manifest retrieval state should persist before Hub I/O");
    let replacement_job = parse_job("example-qwen", "paused");

    job_store
        .replace_current(&replacement_job)
        .expect("the exact same immutable identity should replace atomically");

    assert_eq!(
        job_store
            .load()
            .expect("replacement should load")
            .expect("replacement should remain present"),
        replacement_job
    );
    let mismatched_job = parse_job("different-qwen", "paused");
    assert!(matches!(
        job_store.replace_current(&mismatched_job),
        Err(DownloadJobStoreError::JobIdentityMismatch)
    ));
    let stale_premanifest_job =
        DownloadJob::new_checking_disk("astronomical-test/example-qwen", REVISION, 12, 99)
            .expect("stale premanifest fixture should be valid");
    assert!(matches!(
        job_store.replace_current(&stale_premanifest_job),
        Err(DownloadJobStoreError::InvalidJobReplacement)
    ));
}

#[test]
fn should_allow_only_one_atomic_creator_across_independent_store_instances() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let models_directory = test_directory.path().join("models");
    let start_barrier = Arc::new(Barrier::new(3));
    let mut creation_threads = Vec::new();
    for model_name in ["first-qwen", "second-qwen"] {
        let thread_models_directory = models_directory.clone();
        let thread_start_barrier = Arc::clone(&start_barrier);
        let download_job = parse_job(model_name, "paused");
        creation_threads.push(std::thread::spawn(move || {
            let job_store = DownloadJobStore::new(thread_models_directory);
            thread_start_barrier.wait();
            job_store.create(&download_job).map(|()| download_job)
        }));
    }
    start_barrier.wait();

    let creation_outcomes = creation_threads
        .into_iter()
        .map(|creation_thread| {
            creation_thread
                .join()
                .expect("job creation thread should complete")
        })
        .collect::<Vec<_>>();
    assert_eq!(
        creation_outcomes
            .iter()
            .filter(|creation_outcome| creation_outcome.is_ok())
            .count(),
        1
    );
    assert_eq!(
        creation_outcomes
            .iter()
            .filter(|creation_outcome| {
                matches!(
                    creation_outcome,
                    Err(DownloadJobStoreError::JobAlreadyExists)
                )
            })
            .count(),
        1
    );
    let persisted_job = DownloadJobStore::new(models_directory)
        .load()
        .expect("winning job should load")
        .expect("winning job should be present");
    assert!(creation_outcomes.iter().any(|creation_outcome| {
        creation_outcome
            .as_ref()
            .is_ok_and(|created_job| created_job == &persisted_job)
    }));
}

#[test]
fn should_serialize_pause_and_cancel_across_independent_store_instances() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let models_directory = test_directory.path().join("models");
    let initial_store = DownloadJobStore::new(models_directory.clone());
    initial_store
        .create(&parse_job("example-qwen", "paused"))
        .expect("paused job should persist");
    let transaction_lock = fs::File::options()
        .read(true)
        .write(true)
        .create(true)
        .open(models_directory.join(".download-job.lock"))
        .expect("transaction lock fixture should open");
    transaction_lock
        .lock()
        .expect("transaction lock fixture should lock");
    let (completion_sender, completion_receiver) = std::sync::mpsc::channel();

    let pause_store = DownloadJobStore::new(models_directory.clone());
    let pause_completion_sender = completion_sender.clone();
    let pause_thread = std::thread::spawn(move || {
        let pause_outcome = pause_store.pause_current_job(200);
        pause_completion_sender
            .send(())
            .expect("pause completion should be observable");
        pause_outcome
    });
    let cancel_store = DownloadJobStore::new(models_directory.clone());
    let cancel_thread = std::thread::spawn(move || {
        let cancel_outcome = cancel_store.cancel_current_job();
        completion_sender
            .send(())
            .expect("cancel completion should be observable");
        cancel_outcome
    });

    assert!(
        completion_receiver
            .recv_timeout(Duration::from_millis(100))
            .is_err(),
        "both transactions must wait for the shared filesystem lock"
    );
    transaction_lock
        .unlock()
        .expect("transaction lock fixture should unlock");
    pause_thread
        .join()
        .expect("pause thread should complete")
        .expect("pause transaction should succeed");
    cancel_thread
        .join()
        .expect("cancel thread should complete")
        .expect("cancel transaction should succeed");

    assert!(
        DownloadJobStore::new(models_directory)
            .load()
            .expect("serialized final state should load")
            .is_none()
    );
}

#[test]
fn should_reject_pause_and_cancel_while_publication_requires_reconciliation() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let job_store = DownloadJobStore::new(test_directory.path().join("models"));
    let publishing_job = parse_job("example-qwen", "publishing");
    job_store
        .create(&publishing_job)
        .expect("publishing job should persist");

    assert!(matches!(
        job_store.pause_current_job(200),
        Err(DownloadJobStoreError::PublishingRecoveryRequired)
    ));
    assert!(matches!(
        job_store.cancel_current_job(),
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

#[cfg(unix)]
#[test]
fn should_leave_external_content_and_the_job_untouched_when_cancel_finds_a_symlink() {
    use std::os::unix::fs::symlink;

    let test_directory = TempDir::new().expect("temporary directory should be available");
    let models_directory = test_directory.path().join("models");
    let job_store = DownloadJobStore::new(models_directory.clone());
    job_store
        .create(&parse_job("example-qwen", "paused"))
        .expect("paused job should persist");
    let external_directory = test_directory.path().join("external");
    fs::create_dir(&external_directory).expect("external directory should be created");
    let external_file = external_directory.join("keep.txt");
    fs::write(&external_file, b"Romeo and Juliet").expect("external fixture should be written");
    fs::create_dir(models_directory.join(".incomplete"))
        .expect("incomplete directory should be created");
    symlink(
        &external_directory,
        models_directory.join(".incomplete/astronomical-test"),
    )
    .expect("staging symlink should be created");

    assert!(matches!(
        job_store.cancel_current_job(),
        Err(DownloadJobStoreError::UnsafeFilesystemObject { .. })
    ));
    assert_eq!(
        fs::read(&external_file).expect("external fixture should remain readable"),
        b"Romeo and Juliet"
    );
    assert!(job_store.job_file_path().exists());
}

#[test]
fn should_preserve_the_io_cause_when_the_models_parent_is_missing() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let models_directory = test_directory.path().join("missing-parent/models");
    let job_store = DownloadJobStore::new(models_directory.clone());

    let creation_error = job_store
        .create(&parse_job("example-qwen", "paused"))
        .expect_err("missing parent should reject job creation");

    match creation_error {
        DownloadJobStoreError::CreateDirectory { path, source } => {
            assert_eq!(path, models_directory);
            assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
        }
        unexpected_error => panic!("unexpected store error: {unexpected_error}"),
    }
}

fn parse_job(model_name: &str, state: &str) -> DownloadJob {
    let is_publishing = state == "publishing";
    let bytes_completed = if is_publishing { 12 } else { 0 };
    let bytes_on_disk = bytes_completed;
    DownloadJob::parse_json(&format!(
        "{{\
            \"huggingface_id\":\"astronomical-test/{model_name}\",\
            \"revision\":\"{REVISION}\",\
            \"state\":\"{state}\",\
            \"bytes_completed\":{bytes_completed},\
            \"bytes_total\":12,\
            \"current_file_relative_path\":null,\
            \"files\":[{{\
                \"relative_path\":\"weights/model.safetensors\",\
                \"expected_bytes\":12,\
                \"expected_digest\":{{\"algorithm\":\"sha256\",\"hex\":\"{DIGEST}\"}},\
                \"bytes_on_disk\":{bytes_on_disk}\
            }}],\
            \"error_code\":null,\
            \"updated_at\":100\
        }}"
    ))
    .expect("fixture job should be valid")
}
