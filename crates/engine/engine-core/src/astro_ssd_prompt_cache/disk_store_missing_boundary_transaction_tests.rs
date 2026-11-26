//! The steps of the missing-boundary transaction that need neither a model
//! nor a GPU: the staging name and the final rename. Staging, quota admission
//! and the whole transaction run on a real runtime in the astro SSD prompt
//! cache disk store acceptance tests.

use std::fs;

use model_primitives::performance_attribution::{PerformanceAttribution, PerformanceOperation};

use super::{missing_boundary_staging_name, rename_staged_boundary_into_block};
use crate::astro_ssd_prompt_cache::disk_store_error::AstroSsdPromptCacheDiskStoreError;
use crate::astro_ssd_prompt_cache::disk_store_file::StagedAstroSsdPromptCacheStateFile;

#[test]
fn should_name_the_staging_directory_after_the_block_hash_with_a_boundary_suffix() {
    let block_hash = [0xab_u8; 32];

    let staging_name = missing_boundary_staging_name(&block_hash);

    assert_eq!(staging_name, format!("{}-boundary", "ab".repeat(32)));
}

#[test]
fn should_keep_every_byte_of_a_hash_made_only_of_zeros() {
    let staging_name = missing_boundary_staging_name(&[0_u8; 32]);

    assert_eq!(staging_name, format!("{}-boundary", "0".repeat(64)));
}

struct StagedBoundaryFixture {
    _root_directory: tempfile::TempDir,
    staging_directory: std::path::PathBuf,
    block_directory: std::path::PathBuf,
    staged_boundary_state: StagedAstroSsdPromptCacheStateFile,
}

fn staged_boundary_fixture() -> StagedBoundaryFixture {
    let root_directory = tempfile::tempdir().expect("a blocks directory");
    let staging_directory = root_directory.path().join("staging");
    let block_directory = root_directory.path().join("block");
    fs::create_dir(&staging_directory).expect("a staging directory");
    fs::create_dir(&block_directory).expect("a committed block directory");
    let staged_file_path = staging_directory.join("boundary_state.safetensors");
    fs::write(
        &staged_file_path,
        b"Rough winds do shake the darling buds of May",
    )
    .expect("a staged boundary file");
    StagedBoundaryFixture {
        _root_directory: root_directory,
        staging_directory,
        block_directory,
        staged_boundary_state: StagedAstroSsdPromptCacheStateFile {
            file_path: staged_file_path,
            file_size_bytes: 44,
        },
    }
}

#[test]
fn should_rename_the_staged_boundary_file_into_the_committed_block_directory() {
    let fixture = staged_boundary_fixture();
    let final_boundary_file_path = fixture.block_directory.join("boundary_state.safetensors");

    rename_staged_boundary_into_block(
        &fixture.staged_boundary_state,
        &final_boundary_file_path,
        &fixture.staging_directory,
        &mut PerformanceAttribution::disabled(),
    )
    .expect("the rename should succeed");

    assert_eq!(
        fs::read(&final_boundary_file_path).expect("the committed boundary file"),
        b"Rough winds do shake the darling buds of May"
    );
    assert!(!fixture.staged_boundary_state.file_path.exists());
    assert!(
        fixture.staging_directory.is_dir(),
        "the caller removes the staging directory after a successful rename"
    );
}

#[test]
fn should_record_the_rename_as_an_atomic_commit_when_attribution_is_enabled() {
    let fixture = staged_boundary_fixture();
    let final_boundary_file_path = fixture.block_directory.join("boundary_state.safetensors");
    let mut performance_attribution = PerformanceAttribution::enabled();

    rename_staged_boundary_into_block(
        &fixture.staged_boundary_state,
        &final_boundary_file_path,
        &fixture.staging_directory,
        &mut performance_attribution,
    )
    .expect("the rename should succeed");

    let atomic_commit_measurement = performance_attribution
        .operation_measurement(PerformanceOperation::AstroSsdPromptCacheAtomicCommit)
        .expect("the rename should be measured");
    assert_eq!(atomic_commit_measurement.occurrence_count(), 1);
}

#[test]
fn should_remove_the_staging_directory_when_the_rename_cannot_happen() {
    let fixture = staged_boundary_fixture();
    let unreachable_final_path = fixture
        .block_directory
        .join("no_such_directory")
        .join("boundary_state.safetensors");

    let failure = rename_staged_boundary_into_block(
        &fixture.staged_boundary_state,
        &unreachable_final_path,
        &fixture.staging_directory,
        &mut PerformanceAttribution::disabled(),
    )
    .expect_err("a rename into a missing directory should fail");

    assert!(matches!(
        failure,
        AstroSsdPromptCacheDiskStoreError::RenameTempFile { .. }
    ));
    assert!(!fixture.staging_directory.exists());
}
