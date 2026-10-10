use std::fs;
use std::io::Read;
use std::os::unix::fs::FileExt;

use astronomical_runtime_integration::weights_file_cache_retention::{
    self, WeightsFileCacheRetention, WeightsFileCacheRetentionError,
};

const WEIGHT_FILE_CONTENTS: &[u8] = b"weights payload for cache retention coverage";

fn write_weights_fixture(sandbox_directory: &std::path::Path) -> std::path::PathBuf {
    let weights_file_path = sandbox_directory.join("model-00001-of-00002.safetensors");
    fs::write(&weights_file_path, WEIGHT_FILE_CONTENTS)
        .expect("the test should write a weights fixture file");
    weights_file_path
}

#[test]
fn should_open_a_materialize_once_weights_file_and_read_its_bytes() {
    let sandbox_directory =
        tempfile::tempdir().expect("the test should create a weights file sandbox");
    let weights_file_path = write_weights_fixture(sandbox_directory.path());

    let mut weights_file = weights_file_cache_retention::open_weights_file(
        &weights_file_path,
        WeightsFileCacheRetention::MaterializeOnce,
    )
    .expect("a materialize-once weights file should open");

    let mut read_buffer = Vec::new();
    weights_file
        .read_to_end(&mut read_buffer)
        .expect("the materialize-once weights file should remain readable");
    assert_eq!(read_buffer, WEIGHT_FILE_CONTENTS);
}

#[test]
fn should_open_a_reuse_across_reads_weights_file_and_read_its_bytes() {
    let sandbox_directory =
        tempfile::tempdir().expect("the test should create a weights file sandbox");
    let weights_file_path = write_weights_fixture(sandbox_directory.path());

    let mut weights_file = weights_file_cache_retention::open_weights_file(
        &weights_file_path,
        WeightsFileCacheRetention::ReuseAcrossReads,
    )
    .expect("a reuse-across-reads weights file should open");

    let mut read_buffer = Vec::new();
    weights_file
        .read_to_end(&mut read_buffer)
        .expect("the reuse-across-reads weights file should remain readable");
    assert_eq!(read_buffer, WEIGHT_FILE_CONTENTS);
}

#[test]
fn should_apply_retention_to_an_exclusively_owned_descriptor_without_breaking_reads() {
    let sandbox_directory =
        tempfile::tempdir().expect("the test should create a weights file sandbox");
    let weights_file_path = write_weights_fixture(sandbox_directory.path());

    let weights_file = fs::File::open(&weights_file_path)
        .expect("the test should open the weights fixture directly");
    weights_file_cache_retention::apply_weights_file_cache_retention(
        &weights_file,
        &weights_file_path,
        WeightsFileCacheRetention::MaterializeOnce,
    );
    weights_file_cache_retention::apply_weights_file_cache_retention(
        &weights_file,
        &weights_file_path,
        WeightsFileCacheRetention::ReuseAcrossReads,
    );

    let mut read_buffer = Vec::new();
    weights_file
        .take(WEIGHT_FILE_CONTENTS.len() as u64)
        .read_to_end(&mut read_buffer)
        .expect("positional reads through a flagged descriptor should keep working");
    assert_eq!(read_buffer, WEIGHT_FILE_CONTENTS);
}

#[test]
fn should_read_positionally_through_a_materialize_once_descriptor() {
    let sandbox_directory =
        tempfile::tempdir().expect("the test should create a weights file sandbox");
    let weights_file_path = write_weights_fixture(sandbox_directory.path());

    let weights_file = weights_file_cache_retention::open_weights_file(
        &weights_file_path,
        WeightsFileCacheRetention::MaterializeOnce,
    )
    .expect("a materialize-once weights file should open");

    let mut read_buffer = vec![0u8; WEIGHT_FILE_CONTENTS.len()];
    weights_file
        .read_exact_at(&mut read_buffer, 0)
        .expect("positional reads are the resident materialization access pattern");
    assert_eq!(read_buffer, WEIGHT_FILE_CONTENTS);
}

#[test]
fn should_fail_with_a_typed_error_when_the_weights_file_is_missing() {
    let sandbox_directory =
        tempfile::tempdir().expect("the test should create a weights file sandbox");
    let missing_weights_file_path = sandbox_directory.path().join("missing.safetensors");

    let retention_error = weights_file_cache_retention::open_weights_file(
        &missing_weights_file_path,
        WeightsFileCacheRetention::MaterializeOnce,
    )
    .expect_err("a missing weights file must fail the materialization open");

    assert!(matches!(
        retention_error,
        WeightsFileCacheRetentionError::OpenFailed { .. }
    ));
    assert!(
        retention_error.to_string().contains("missing.safetensors"),
        "the open failure should name the offending weights file: {retention_error}"
    );
}
