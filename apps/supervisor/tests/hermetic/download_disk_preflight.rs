//! Hermetic admission contracts for Library download disk capacity.

use std::{
    io,
    path::{Path, PathBuf},
    sync::Mutex,
};

use astronomical_supervisor::{
    DiskCapacityQuery, DownloadDiskPreflight, DownloadDiskPreflightError, DownloadJob,
    Fs4DiskCapacityQuery,
};

const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
const SHA256: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

#[test]
fn should_admit_initial_download_when_catalog_bytes_and_one_percent_margin_fit() {
    let capacity_query = FakeDiskCapacityQuery::with_available_bytes(4_040_000_000);
    let preflight = DownloadDiskPreflight::new(capacity_query);

    let capacity_check = preflight
        .check_initial_download(Path::new("fictional-library"), 4_000_000_000)
        .expect("catalog bytes plus the decimal-byte margin should fit");

    assert_eq!(capacity_check.required_bytes(), 4_040_000_000);
    assert_eq!(capacity_check.available_bytes(), 4_040_000_000);
    assert_eq!(
        preflight.capacity_query().queried_paths(),
        vec![PathBuf::from("fictional-library")]
    );
}

#[test]
fn should_apply_one_percent_with_a_minimum_one_byte_margin() {
    for (catalog_approximate_bytes, expected_required_bytes) in [(1, 2), (100, 101), (101, 103)] {
        let preflight = DownloadDiskPreflight::new(FakeDiskCapacityQuery::with_available_bytes(
            expected_required_bytes,
        ));

        let capacity_check = preflight
            .check_initial_download(Path::new("fictional-library"), catalog_approximate_bytes)
            .expect("the minimum one-percent margin should fit");

        assert_eq!(capacity_check.required_bytes(), expected_required_bytes);
    }
}

#[test]
fn should_reject_initial_download_with_required_and_available_bytes() {
    let preflight = DownloadDiskPreflight::new(FakeDiskCapacityQuery::with_available_bytes(1_009));

    let preflight_error = preflight
        .check_initial_download(Path::new("fictional-library"), 1_000)
        .expect_err("capacity below the catalog estimate and margin must fail closed");

    assert!(matches!(
        preflight_error,
        DownloadDiskPreflightError::InsufficientSpace {
            required_bytes: 1_010,
            available_bytes: 1_009,
        }
    ));
}

#[test]
fn should_admit_exact_manifest_remaining_bytes_without_an_extra_margin() {
    let preflight = DownloadDiskPreflight::new(FakeDiskCapacityQuery::with_available_bytes(101));

    let capacity_check = preflight
        .check_job_remaining_bytes(Path::new("fictional-library"), &job_with_progress(12, 113))
        .expect("exact remaining manifest bytes should not receive another margin");

    assert_eq!(capacity_check.required_bytes(), 101);
    assert_eq!(capacity_check.available_bytes(), 101);
}

#[test]
fn should_reject_exact_manifest_remaining_bytes_with_typed_capacity_evidence() {
    let preflight = DownloadDiskPreflight::new(FakeDiskCapacityQuery::with_available_bytes(100));

    let preflight_error = preflight
        .check_job_remaining_bytes(Path::new("fictional-library"), &job_with_progress(12, 113))
        .expect_err("exact remaining bytes beyond capacity must fail closed");

    assert!(matches!(
        preflight_error,
        DownloadDiskPreflightError::InsufficientSpace {
            required_bytes: 101,
            available_bytes: 100,
        }
    ));
}

#[test]
fn should_reject_remaining_byte_check_before_an_exact_manifest_exists() {
    let preflight = DownloadDiskPreflight::new(FakeDiskCapacityQuery::with_available_bytes(1_000));
    let premanifest_job =
        DownloadJob::new_checking_disk("astronomical-test/example-qwen", REVISION, 100, 100)
            .expect("premanifest job should be valid");

    assert!(matches!(
        preflight.check_job_remaining_bytes(Path::new("fictional-library"), &premanifest_job),
        Err(DownloadDiskPreflightError::ExactManifestRequired)
    ));
    assert!(preflight.capacity_query().queried_paths().is_empty());
}

#[test]
fn should_report_checked_initial_requirement_overflow_before_querying_capacity() {
    let preflight =
        DownloadDiskPreflight::new(FakeDiskCapacityQuery::with_available_bytes(u64::MAX));

    let preflight_error = preflight
        .check_initial_download(Path::new("fictional-library"), u64::MAX)
        .expect_err("an unrepresentable catalog requirement must fail closed");

    assert!(matches!(
        preflight_error,
        DownloadDiskPreflightError::RequiredBytesOverflow {
            catalog_approximate_bytes: u64::MAX,
            margin_bytes: 184_467_440_737_095_517,
        }
    ));
    assert!(preflight.capacity_query().queried_paths().is_empty());
}

#[test]
fn should_preserve_capacity_query_path_required_bytes_and_io_cause() {
    let preflight = DownloadDiskPreflight::new(FakeDiskCapacityQuery::with_failure(
        io::ErrorKind::PermissionDenied,
    ));

    let preflight_error = preflight
        .check_job_remaining_bytes(
            Path::new("fictional-library"),
            &job_with_progress(250, 1_000),
        )
        .expect_err("capacity query failures must remain typed");

    match preflight_error {
        DownloadDiskPreflightError::QueryCapacity {
            path,
            required_bytes,
            source,
        } => {
            assert_eq!(path, PathBuf::from("fictional-library"));
            assert_eq!(required_bytes, 750);
            assert_eq!(source.kind(), io::ErrorKind::PermissionDenied);
        }
        unexpected_error => panic!("unexpected preflight error: {unexpected_error}"),
    }
}

fn job_with_progress(bytes_completed: u64, bytes_total: u64) -> DownloadJob {
    DownloadJob::parse_json(&format!(
        "{{\"huggingface_id\":\"astronomical-test/example-qwen\",\"revision\":\"{REVISION}\",\"state\":\"paused\",\"bytes_completed\":{bytes_completed},\"bytes_total\":{bytes_total},\"current_file_relative_path\":null,\"files\":[{{\"relative_path\":\"model.safetensors\",\"expected_bytes\":{bytes_total},\"expected_digest\":{{\"algorithm\":\"sha256\",\"hex\":\"{SHA256}\"}},\"bytes_on_disk\":{bytes_completed}}}],\"error_code\":null,\"updated_at\":100}}"
    ))
    .expect("disk preflight job fixture should be valid")
}

#[test]
fn should_allow_capacity_query_and_preflight_to_cross_thread_boundaries() {
    fn assert_send_and_sync<T: Send + Sync>() {}

    assert_send_and_sync::<FakeDiskCapacityQuery>();
    assert_send_and_sync::<Fs4DiskCapacityQuery>();
    assert_send_and_sync::<DownloadDiskPreflight>();
    assert_send_and_sync::<DownloadDiskPreflight<FakeDiskCapacityQuery>>();
    let _production_preflight = DownloadDiskPreflight::production();
}

struct FakeDiskCapacityQuery {
    available_bytes: Result<u64, io::ErrorKind>,
    queried_paths: Mutex<Vec<PathBuf>>,
}

impl FakeDiskCapacityQuery {
    fn with_available_bytes(available_bytes: u64) -> Self {
        Self {
            available_bytes: Ok(available_bytes),
            queried_paths: Mutex::new(Vec::new()),
        }
    }

    fn with_failure(error_kind: io::ErrorKind) -> Self {
        Self {
            available_bytes: Err(error_kind),
            queried_paths: Mutex::new(Vec::new()),
        }
    }

    fn queried_paths(&self) -> Vec<PathBuf> {
        self.queried_paths
            .lock()
            .expect("fake query path lock should remain available")
            .clone()
    }
}

impl DiskCapacityQuery for FakeDiskCapacityQuery {
    fn available_space_bytes(&self, existing_same_volume_path: &Path) -> io::Result<u64> {
        self.queried_paths
            .lock()
            .expect("fake query path lock should remain available")
            .push(existing_same_volume_path.to_path_buf());
        self.available_bytes.map_err(|error_kind| {
            io::Error::new(error_kind, "intentional fake capacity query failure")
        })
    }
}
