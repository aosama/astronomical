//! The pure decisions publication makes about a block the index already
//! holds, over hand-built index entries and no disk store.

use std::path::PathBuf;

use super::{
    ExistingBlockCompleteness, existing_block_completeness,
    existing_block_matches_requested_topology,
};
use crate::astro_ssd_prompt_cache::disk_store_index::{
    TrackedAstroSsdPromptCacheBlock, TrackedAstroSsdPromptCacheFile,
};

fn tracked_state_file(file_name: &str) -> TrackedAstroSsdPromptCacheFile {
    TrackedAstroSsdPromptCacheFile {
        file_path: PathBuf::from(format!("/blocks/sonnet_18/{file_name}")),
        file_size_bytes: 1_024,
    }
}

fn indexed_block(
    has_sequence_state_file: bool,
    has_boundary_state_file: bool,
) -> TrackedAstroSsdPromptCacheBlock {
    TrackedAstroSsdPromptCacheBlock {
        block_directory_path: PathBuf::from("/blocks/sonnet_18"),
        block_index: 3,
        parent_block_hash: Some([7_u8; 32]),
        sequence_state_file: has_sequence_state_file
            .then(|| tracked_state_file("sequence_state.safetensors")),
        boundary_state_file: has_boundary_state_file
            .then(|| tracked_state_file("boundary_state.safetensors")),
    }
}

#[test]
fn should_call_a_block_complete_when_it_holds_every_state_file_the_model_requires() {
    assert_eq!(
        existing_block_completeness(true, true, &indexed_block(true, true)),
        ExistingBlockCompleteness::Complete
    );
}

#[test]
fn should_not_ask_for_a_state_kind_the_model_does_not_have() {
    assert_eq!(
        existing_block_completeness(true, false, &indexed_block(true, false)),
        ExistingBlockCompleteness::Complete
    );
    assert_eq!(
        existing_block_completeness(false, true, &indexed_block(false, true)),
        ExistingBlockCompleteness::Complete
    );
}

#[test]
fn should_report_a_missing_boundary_snapshot_as_repairable() {
    assert_eq!(
        existing_block_completeness(true, true, &indexed_block(true, false)),
        ExistingBlockCompleteness::BoundaryStateMissing
    );
}

#[test]
fn should_report_a_missing_sequence_state_even_when_the_boundary_is_missing_too() {
    assert_eq!(
        existing_block_completeness(true, true, &indexed_block(false, true)),
        ExistingBlockCompleteness::SequenceStateMissing
    );
    assert_eq!(
        existing_block_completeness(true, true, &indexed_block(false, false)),
        ExistingBlockCompleteness::SequenceStateMissing
    );
}

#[test]
fn should_match_a_request_with_the_stored_block_index_and_parent() {
    assert!(existing_block_matches_requested_topology(
        &indexed_block(true, true),
        3,
        Some([7_u8; 32]),
    ));
}

#[test]
fn should_refuse_a_request_that_names_another_block_index() {
    assert!(!existing_block_matches_requested_topology(
        &indexed_block(true, true),
        4,
        Some([7_u8; 32]),
    ));
}

#[test]
fn should_refuse_a_request_whose_parent_differs_or_is_absent() {
    assert!(!existing_block_matches_requested_topology(
        &indexed_block(true, true),
        3,
        Some([8_u8; 32]),
    ));
    assert!(!existing_block_matches_requested_topology(
        &indexed_block(true, true),
        3,
        None,
    ));
}
