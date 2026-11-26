//! The quota arithmetic and eviction decisions, over hand-built scan
//! candidates and no disk store.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use super::{CommitQuotaDemand, GlobalQuotaTotals, eviction_candidate_was_already_removed};
use crate::astro_ssd_prompt_cache::disk_store_error::AstroSsdPromptCacheDiskStoreError;
use crate::astro_ssd_prompt_cache::disk_store_global_quota_candidate::{
    GlobalAstroSsdPromptCacheBlockSubtree, GlobalAstroSsdPromptCacheEvictionCandidate,
    GlobalAstroSsdPromptCacheFile, GlobalAstroSsdPromptCacheStaleDirectory,
    GlobalAstroSsdPromptCacheStandaloneFileClassification,
};

const ROOT_DIRECTORY: &str = "/astro_ssd_prompt_cache_root";

fn committed_block_subtree(
    root_block_directory: &str,
    file_size_bytes: u64,
) -> GlobalAstroSsdPromptCacheEvictionCandidate {
    GlobalAstroSsdPromptCacheEvictionCandidate::BlockSubtree(
        GlobalAstroSsdPromptCacheBlockSubtree {
            root_block_directory_path: PathBuf::from(root_block_directory),
            file_size_bytes,
            modified_at: SystemTime::UNIX_EPOCH,
            block_directory_paths: vec![PathBuf::from(root_block_directory)],
            tracked_file_paths: Vec::new(),
        },
    )
}

fn visual_embedding_file(file_size_bytes: u64) -> GlobalAstroSsdPromptCacheEvictionCandidate {
    GlobalAstroSsdPromptCacheEvictionCandidate::StandaloneFile(GlobalAstroSsdPromptCacheFile {
        file_path: PathBuf::from("/visual_embeddings/image.safetensors"),
        file_size_bytes,
        modified_at: SystemTime::UNIX_EPOCH,
        is_visual_embedding: true,
        cleanup_classification:
            GlobalAstroSsdPromptCacheStandaloneFileClassification::CommittedArtifact,
    })
}

fn stale_staging_directory() -> GlobalAstroSsdPromptCacheEvictionCandidate {
    GlobalAstroSsdPromptCacheEvictionCandidate::StaleDirectory(
        GlobalAstroSsdPromptCacheStaleDirectory {
            directory_path: PathBuf::from("/blocks/abc.staging-1-2"),
            file_size_bytes: 10,
            modified_at: SystemTime::UNIX_EPOCH,
            tracked_file_paths: Vec::new(),
        },
    )
}

fn demand<'demand>(
    additional_committed_size_bytes: u64,
    post_commit_reclaimable_size_bytes: u64,
    protected_block_directory_paths: &'demand [PathBuf],
) -> CommitQuotaDemand<'demand> {
    CommitQuotaDemand {
        additional_committed_size_bytes,
        post_commit_reclaimable_size_bytes,
        protected_block_directory_paths,
        maximum_size_bytes: 1_000,
        global_astro_ssd_prompt_cache_root_directory: Path::new(ROOT_DIRECTORY),
    }
}

#[test]
fn should_take_a_removed_visual_embedding_out_of_both_totals() {
    let mut totals = GlobalQuotaTotals {
        total_size_bytes: 1_000,
        visual_embedding_total_size_bytes: 300,
    };

    totals.subtract_candidate(&visual_embedding_file(100));

    assert_eq!(totals.total_size_bytes, 900);
    assert_eq!(totals.visual_embedding_total_size_bytes, 200);
}

#[test]
fn should_leave_the_visual_embedding_total_alone_when_a_block_subtree_is_removed() {
    let mut totals = GlobalQuotaTotals {
        total_size_bytes: 1_000,
        visual_embedding_total_size_bytes: 300,
    };

    totals.subtract_candidate(&committed_block_subtree("/blocks/a", 250));

    assert_eq!(totals.total_size_bytes, 750);
    assert_eq!(totals.visual_embedding_total_size_bytes, 300);
}

#[test]
fn should_stop_a_total_at_zero_when_the_removed_candidate_is_larger_than_it() {
    let mut totals = GlobalQuotaTotals {
        total_size_bytes: 50,
        visual_embedding_total_size_bytes: 20,
    };

    totals.subtract_candidate(&visual_embedding_file(100));

    assert_eq!(totals.total_size_bytes, 0);
    assert_eq!(totals.visual_embedding_total_size_bytes, 0);
}

#[test]
fn should_add_the_commit_and_subtract_what_it_makes_redundant() {
    let committed_size_bytes = demand(50, 30, &[]).committed_size_after(100);

    assert_eq!(committed_size_bytes.expect("a small sum"), 120);
}

#[test]
fn should_stop_the_committed_size_at_zero_when_more_is_reclaimed_than_exists() {
    let committed_size_bytes = demand(10, 500, &[]).committed_size_after(100);

    assert_eq!(committed_size_bytes.expect("a saturating difference"), 0);
}

#[test]
fn should_report_the_global_root_when_the_committed_size_overflows() {
    let overflow = demand(1, 0, &[]).committed_size_after(u64::MAX);

    assert!(matches!(
        overflow,
        Err(AstroSsdPromptCacheDiskStoreError::GlobalAstroSsdPromptCacheSizeOverflow {
            global_astro_ssd_prompt_cache_root_directory,
        }) if global_astro_ssd_prompt_cache_root_directory == Path::new(ROOT_DIRECTORY)
    ));
}

#[test]
fn should_keep_a_candidate_whose_path_was_already_removed_with_an_ancestor() {
    let removed_eviction_paths = HashSet::from([PathBuf::from("/blocks/a")]);

    let keeps = demand(0, 0, &[])
        .keeps_eviction_candidate(
            &committed_block_subtree("/blocks/a", 10),
            &removed_eviction_paths,
            5_000,
        )
        .expect("the decision");

    assert!(keeps);
}

#[test]
fn should_keep_a_candidate_when_any_member_directory_was_already_removed() {
    let overlapping_subtree = GlobalAstroSsdPromptCacheEvictionCandidate::BlockSubtree(
        GlobalAstroSsdPromptCacheBlockSubtree {
            root_block_directory_path: PathBuf::from("/blocks/root"),
            file_size_bytes: 10,
            modified_at: SystemTime::UNIX_EPOCH,
            block_directory_paths: vec![
                PathBuf::from("/blocks/root"),
                PathBuf::from("/blocks/child"),
            ],
            tracked_file_paths: Vec::new(),
        },
    );
    let removed_eviction_paths = HashSet::from([PathBuf::from("/blocks/child")]);

    assert!(eviction_candidate_was_already_removed(
        &overlapping_subtree,
        &removed_eviction_paths
    ));
    assert!(!eviction_candidate_was_already_removed(
        &overlapping_subtree,
        &HashSet::new()
    ));
}

#[test]
fn should_remove_an_interrupted_transaction_even_when_the_projection_fits() {
    let keeps = demand(0, 0, &[])
        .keeps_eviction_candidate(&stale_staging_directory(), &HashSet::new(), 10)
        .expect("the decision");

    assert!(!keeps);
}

#[test]
fn should_keep_a_committed_subtree_on_the_chain_being_extended_even_when_over_the_ceiling() {
    let protected_block_directory_paths = [PathBuf::from("/blocks/a")];

    let keeps = demand(0, 0, &protected_block_directory_paths)
        .keeps_eviction_candidate(
            &committed_block_subtree("/blocks/a", 10),
            &HashSet::new(),
            5_000,
        )
        .expect("the decision");

    assert!(keeps);
}

#[test]
fn should_remove_committed_content_only_while_the_projection_exceeds_the_ceiling() {
    let candidate = committed_block_subtree("/blocks/a", 10);
    let at_ceiling = demand(0, 0, &[])
        .keeps_eviction_candidate(&candidate, &HashSet::new(), 1_000)
        .expect("the decision");
    let one_byte_over = demand(0, 0, &[])
        .keeps_eviction_candidate(&candidate, &HashSet::new(), 1_001)
        .expect("the decision");

    assert!(at_ceiling, "a projection exactly at the ceiling fits");
    assert!(!one_byte_over);
}

#[test]
fn should_pass_the_overflow_of_the_projection_through_the_eviction_decision() {
    let decision = demand(1, 0, &[]).keeps_eviction_candidate(
        &committed_block_subtree("/blocks/a", 10),
        &HashSet::new(),
        u64::MAX,
    );

    assert!(matches!(
        decision,
        Err(AstroSsdPromptCacheDiskStoreError::GlobalAstroSsdPromptCacheSizeOverflow { .. })
    ));
}

#[test]
fn should_accept_a_final_size_exactly_at_the_ceiling() {
    assert!(demand(100, 0, &[]).ensure_satisfied_at(900).is_ok());
}

#[test]
fn should_refuse_a_final_size_over_the_ceiling_and_name_both_numbers() {
    let refusal = demand(100, 0, &[]).ensure_satisfied_at(901);

    assert!(matches!(
        refusal,
        Err(
            AstroSsdPromptCacheDiskStoreError::GlobalAstroSsdPromptCacheQuotaNotSatisfied {
                maximum_size_bytes: 1_000,
                remaining_size_bytes: 1_001,
            }
        )
    ));
}

#[test]
fn should_count_reclaimable_bytes_toward_fitting_the_ceiling() {
    assert!(demand(300, 250, &[]).ensure_satisfied_at(950).is_ok());
}
