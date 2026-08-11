//! Proves that route inspection is read-only and exact before cache commit.
//!
//! Three repeated assignments select one expert. Analysis must therefore report
//! one missing page and zero disk reads; only commit may perform that one read.

use std::fs;

use astronomical_runtime_integration::{MlxNativeExpertCache, MlxNativeExpertLayerDescriptor};

use super::native_expert_cache_fixture::{EXPERT_CAPACITY, build_two_expert_source_fixture};
use crate::common::runtime_test_support::runtime;

#[test]
fn should_analyze_exact_missing_route_bytes_before_committing_disk_reads() {
    let runtime = runtime();
    let fixture_directory = tempfile::tempdir().expect("the native cache fixture should exist");
    let source_file_path = fixture_directory.path().join("expert-source.bin");
    let (source_file_bytes, source_descriptors) =
        build_two_expert_source_fixture(&source_file_path);
    fs::write(&source_file_path, &source_file_bytes)
        .expect("the expert source fixture should be writable");
    let bytes_per_expert = source_file_bytes.len() as u64 / EXPERT_CAPACITY as u64;
    let native_expert_cache = MlxNativeExpertCache::new(
        &runtime,
        &[MlxNativeExpertLayerDescriptor::new(
            EXPERT_CAPACITY,
            source_descriptors,
        )],
        bytes_per_expert * 2,
    )
    .expect("the native cache fixture should be valid");
    let repeated_expert_indices = runtime
        .array_from_i32(&[1, 1, 1], &[1, 3])
        .expect("the repeated route should be valid");

    let (route_analysis, analysis_report) = native_expert_cache
        .analyze_layer(&runtime, 0, &repeated_expert_indices, true)
        .expect("exact route analysis should succeed without loading pages");
    assert_eq!(analysis_report.selected_expert_assignment_count(), 3);
    assert_eq!(analysis_report.distinct_route_expert_count(), 1);
    assert_eq!(analysis_report.missing_route_expert_count(), 1);
    assert_eq!(
        analysis_report.missing_route_payload_byte_count(),
        bytes_per_expert
    );
    assert_eq!(analysis_report.disk_page_load_count(), 0);
    assert_eq!(native_expert_cache.statistics().resident_expert_count(), 0);

    let (_snapshot, commit_report) = native_expert_cache
        .commit_layer(&runtime, route_analysis, true)
        .expect("the analyzed route should commit successfully");
    assert_eq!(commit_report.cache_miss_count(), 1);
    assert_eq!(commit_report.disk_page_load_count(), 1);
    assert_eq!(
        commit_report.retention_ceiling_before_byte_count(),
        bytes_per_expert * 2
    );
    assert_eq!(
        commit_report.retention_ceiling_after_byte_count(),
        bytes_per_expert * 2
    );
    assert_eq!(native_expert_cache.statistics().resident_expert_count(), 1);

    let (warm_route_analysis, warm_analysis_report) = native_expert_cache
        .analyze_layer(&runtime, 0, &repeated_expert_indices, true)
        .expect("the warm route should remain exactly analyzable");
    assert_eq!(warm_analysis_report.selected_expert_assignment_count(), 3);
    assert_eq!(warm_analysis_report.distinct_route_expert_count(), 1);
    assert_eq!(warm_analysis_report.missing_route_expert_count(), 0);
    assert_eq!(warm_analysis_report.missing_route_payload_byte_count(), 0);
    assert_eq!(warm_analysis_report.disk_page_load_count(), 0);

    let (_warm_snapshot, warm_commit_report) = native_expert_cache
        .commit_layer(&runtime, warm_route_analysis, true)
        .expect("the exact warm route should commit without storage reads");
    assert_eq!(warm_commit_report.cache_hit_count(), 1);
    assert_eq!(warm_commit_report.cache_miss_count(), 0);
    assert_eq!(warm_commit_report.disk_page_load_count(), 0);
}
