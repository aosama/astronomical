use std::path::{Path, PathBuf};
use std::time::Duration;

use tokio::time::timeout;

#[tokio::test]
async fn should_keep_qwen3_5_imports_directed_through_shared_core() {
    timeout(Duration::from_secs(5), async {
        let source_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
        let qwen3_5_core = source_root.join("qwen3_5_core");
        let qwen3_5_resident = source_root.join("qwen3_5_resident");
        let qwen3_5_streaming = source_root.join("qwen3_5_streaming");
        let qwen3_5_family_facade = source_root.join("qwen3_5.rs");
        let crate_root = source_root.join("lib.rs");

        assert_tree_excludes_source_fragments(
            &qwen3_5_core,
            &[
                "crate::qwen3_5::",
                "crate::qwen3_5_resident::",
                "crate::qwen3_5_streaming::",
            ],
        );
        assert_tree_excludes_source_fragments(
            &qwen3_5_resident,
            &["crate::qwen3_5::", "crate::qwen3_5_streaming::"],
        );
        assert_tree_excludes_source_fragments(
            &qwen3_5_streaming,
            &[
                "crate::qwen3_5::",
                "crate::qwen3_5_resident::",
                "resident_expert_weights",
                "resident_expert_layer_weights",
                "Qwen3_5ResidentExpertLayerWeights",
                "Qwen3_5ResidentGateUpWeights",
                "Qwen3_5ResidentExpertWeights",
                "try_promote_experts_to_resident",
                "demote_resident_experts_to_paging",
                "sparse_experts_are_paged",
            ],
        );
        assert_source_excludes_source_fragments(
            &qwen3_5_resident.join("inference_execution/prompt_processing_chunk_sizer.rs"),
            &["AdaptiveRamGrowthExecutionProfile", "ssd_streaming"],
        );
        assert_source_excludes_source_fragments(
            &qwen3_5_streaming.join("inference_execution/prompt_processing_chunk_sizer.rs"),
            &["AdaptiveRamGrowthExecutionProfile", "Resident"],
        );
        assert_source_excludes_source_fragments(&qwen3_5_family_facade, &["Qwen3_5Engine"]);
        assert_source_excludes_source_fragments(&crate_root, &["Qwen3_5Engine"]);
    })
    .await
    .expect("Qwen3.5 import-direction inspection should finish within five seconds");
}

fn assert_tree_excludes_source_fragments(source_directory: &Path, forbidden_fragments: &[&str]) {
    for directory_entry in
        std::fs::read_dir(source_directory).expect("Qwen3.5 source directory should be readable")
    {
        let directory_entry =
            directory_entry.expect("Qwen3.5 source directory entry should be readable");
        let source_path = directory_entry.path();
        if source_path.is_dir() {
            assert_tree_excludes_source_fragments(&source_path, forbidden_fragments);
        } else if source_path
            .extension()
            .is_some_and(|extension| extension == "rs")
        {
            assert_source_excludes_source_fragments(&source_path, forbidden_fragments);
        }
    }
}

fn assert_source_excludes_source_fragments(source_path: &Path, forbidden_fragments: &[&str]) {
    let source_text =
        std::fs::read_to_string(source_path).expect("Qwen3.5 Rust source should be readable");
    for (line_number, source_line) in source_text.lines().enumerate() {
        if source_line.trim_start().starts_with("//") {
            continue;
        }
        for forbidden_fragment in forbidden_fragments {
            assert!(
                !source_line.contains(forbidden_fragment),
                "{}:{} contains forbidden source fragment {forbidden_fragment}",
                source_path.display(),
                line_number + 1,
            );
        }
    }
}
