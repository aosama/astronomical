//! Unit tests for the pure logic of `cargo xtask scan-retired-native-output`,
//! the Phase 0 probe tracking issue #1486.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::retired_native_scan::{
    allocated_bytes_under, gate_error_message, is_retired_candidate_directory,
    parse_scan_arguments, scan_root_for_retired_candidates, scan_root_is_refused,
    ScanArguments,
};

fn owned(arguments: &[&str]) -> Vec<String> {
    arguments
        .iter()
        .map(|argument| (*argument).to_owned())
        .collect()
}

fn generation_out_directory(root: &Path, generation_suffix: &str) -> PathBuf {
    root.join("debug")
        .join("build")
        .join(format!("astronomical-runtime-integration-{generation_suffix}"))
        .join("out")
}

mod should_parse_scan_arguments {
    use super::*;

    #[test]
    fn should_accept_a_plain_scan() {
        assert_eq!(
            parse_scan_arguments(&owned(&[])),
            ScanArguments::Scan {
                target_directory_override: None,
                extra_scan_roots: Vec::new(),
                fail_on_found: false
            }
        );
    }

    #[test]
    fn should_accept_override_extra_roots_and_fail_on_found() {
        assert_eq!(
            parse_scan_arguments(&owned(&[
                "--target-directory",
                "/tmp/target",
                "--fail-on-found",
                "/tmp/cache-a",
                "/tmp/cache-b"
            ])),
            ScanArguments::Scan {
                target_directory_override: Some("/tmp/target".to_owned()),
                extra_scan_roots: vec!["/tmp/cache-a".to_owned(), "/tmp/cache-b".to_owned()],
                fail_on_found: true
            }
        );
    }

    #[test]
    fn should_reject_a_dangling_target_directory_flag() {
        assert_eq!(
            parse_scan_arguments(&owned(&["--target-directory"])),
            ScanArguments::PrintUsage
        );
    }

    #[test]
    fn should_reject_a_flag_swallowed_as_the_target_directory_value() {
        assert_eq!(
            parse_scan_arguments(&owned(&["--target-directory", "--fail-on-found"])),
            ScanArguments::PrintUsage
        );
    }

    #[test]
    fn should_reject_a_duplicated_target_directory_flag() {
        assert_eq!(
            parse_scan_arguments(&owned(&[
                "--target-directory",
                "/tmp/a",
                "--target-directory",
                "/tmp/b"
            ])),
            ScanArguments::PrintUsage
        );
    }

    #[test]
    fn should_reject_a_duplicated_fail_on_found_flag() {
        assert_eq!(
            parse_scan_arguments(&owned(&["--fail-on-found", "--fail-on-found"])),
            ScanArguments::PrintUsage
        );
    }

    #[test]
    fn should_reject_unknown_flags() {
        assert_eq!(
            parse_scan_arguments(&owned(&["--apply"])),
            ScanArguments::PrintUsage
        );
    }

    #[test]
    fn should_treat_help_as_success_not_as_a_usage_error() {
        assert_eq!(parse_scan_arguments(&owned(&["--help"])), ScanArguments::PrintHelp);
        assert_eq!(parse_scan_arguments(&owned(&["-h"])), ScanArguments::PrintHelp);
    }
}

mod should_classify_retired_candidates {
    use super::*;

    #[test]
    fn should_accept_the_legacy_directory_in_a_generation_layout() {
        let out_directory = PathBuf::from("/fixture")
            .join("debug/build/astronomical-runtime-integration-abc123/out");
        assert!(is_retired_candidate_directory(
            &out_directory,
            "mlx-c-runtime-build".as_ref()
        ));
    }

    #[test]
    fn should_accept_interrupted_removal_handoffs() {
        let out_directory = PathBuf::from("/fixture")
            .join("debug/build/astronomical-runtime-integration-abc123/out");
        assert!(is_retired_candidate_directory(
            &out_directory,
            ".mlx-c-runtime-build.astronomical-removing-4711-0".as_ref()
        ));
    }

    #[test]
    fn should_reject_the_legacy_name_outside_an_out_directory() {
        let misplaced_directory = PathBuf::from("/fixture/debug/mlx-c-runtime-build");
        assert!(!is_retired_candidate_directory(
            &misplaced_directory,
            "mlx-c-runtime-build".as_ref()
        ));
    }

    #[test]
    fn should_reject_candidates_without_a_runtime_generation_grandparent() {
        let out_directory = PathBuf::from("/fixture/debug/build/other-crate-abc/out");
        assert!(!is_retired_candidate_directory(
            &out_directory,
            "mlx-c-runtime-build".as_ref()
        ));
    }

    #[test]
    fn should_reject_candidates_without_a_build_great_grandparent() {
        let out_directory =
            PathBuf::from("/fixture/debug/not-build/astronomical-runtime-integration-abc/out");
        assert!(!is_retired_candidate_directory(
            &out_directory,
            "mlx-c-runtime-build".as_ref()
        ));
    }

    #[test]
    fn should_reject_unrelated_directory_names() {
        let out_directory = PathBuf::from("/fixture")
            .join("debug/build/astronomical-runtime-integration-abc123/out");
        assert!(!is_retired_candidate_directory(
            &out_directory,
            "mlx-c-runtime".as_ref()
        ));
        assert!(!is_retired_candidate_directory(
            &out_directory,
            "something-else".as_ref()
        ));
    }
}

mod should_refuse_dangerous_scan_roots {
    use super::*;

    #[test]
    fn should_refuse_filesystem_root_current_and_parent() {
        assert!(scan_root_is_refused("/"));
        assert!(scan_root_is_refused("."));
        assert!(scan_root_is_refused(".."));
    }

    #[test]
    fn should_accept_ordinary_directories() {
        assert!(!scan_root_is_refused("/private/tmp/target"));
        assert!(!scan_root_is_refused("./target"));
    }
}

mod should_word_the_gate_error {
    use super::*;

    #[test]
    fn should_name_the_count_the_clean_guidance_and_the_tracking_issue() {
        let gate_error_text = gate_error_message(3);
        assert!(gate_error_text.starts_with("Error: 3 retired native-output directories remain"));
        assert!(
            gate_error_text.contains("`cargo clean`"),
            "the only surviving removal path for stale generations is cargo clean: {gate_error_text}"
        );
        assert!(
            gate_error_text.contains("#1486"),
            "the message should point at the tracking issue: {gate_error_text}"
        );
        assert!(
            !gate_error_text.contains("scripts/"),
            "the shell cleaner is retired; no script path may resurface: {gate_error_text}"
        );
        assert!(
            !gate_error_text.contains("automatically"),
            "the build-script auto-clear was retired in #1486 Phase 2; the message \
             must not promise a remedy that no longer exists: {gate_error_text}"
        );
    }
}

mod should_walk_targets_and_collect_candidates {
    use super::*;

    struct ScanFixture {
        root_directory: PathBuf,
        legacy_directory: PathBuf,
        removal_directory: PathBuf,
    }

    impl Drop for ScanFixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root_directory);
        }
    }

    /// A complete fake Cargo target: two genuine retired candidates, a
    /// mis-decoy legacy name at the wrong depth, a symlink decoy inside an
    /// out directory, and ordinary Cargo noise.
    fn install_scan_fixture(fixture_label: &str) -> ScanFixture {
        let root_directory = std::env::temp_dir().join(format!(
            "retired-scan-walk-{fixture_label}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root_directory);
        let out_directory = generation_out_directory(&root_directory, "gen1");
        let legacy_directory = out_directory.join("mlx-c-runtime-build");
        fs::create_dir_all(legacy_directory.join("CMakeFiles"))
            .expect("the legacy fixture should be creatable");
        fs::write(legacy_directory.join("build.ninja"), "fixture\n")
            .expect("the legacy fixture file should be writable");
        let removal_directory = out_directory.join(".mlx-c-runtime-build.astronomical-removing-9-0");
        fs::create_dir_all(&removal_directory).expect("the removal fixture should be creatable");
        let misplaced_directory = root_directory.join("debug/mlx-c-runtime-build");
        fs::create_dir_all(&misplaced_directory).expect("the decoy should be creatable");
        let symlink_decoy_target = root_directory.join("elsewhere");
        fs::create_dir_all(&symlink_decoy_target).expect("the symlink target should be creatable");
        let _ = std::os::unix::fs::symlink(&symlink_decoy_target, out_directory.join("mlx-c-runtime-build-link"));
        fs::create_dir_all(root_directory.join("debug/incremental"))
            .expect("the ordinary noise should be creatable");
        ScanFixture {
            root_directory,
            legacy_directory,
            removal_directory,
        }
    }

    #[test]
    fn should_find_only_structural_candidates_and_skip_decoys() {
        let fixture = install_scan_fixture("decoys");
        let mut far_future_progress_at = Instant::now() + std::time::Duration::from_secs(60);
        let scan_summary = scan_root_for_retired_candidates(
            &fixture.root_directory,
            Instant::now(),
            &mut far_future_progress_at,
        );
        assert_eq!(
            scan_summary.retired_directories,
            vec![fixture.removal_directory.clone(), fixture.legacy_directory.clone()],
            "exactly the two structural candidates should be found, sorted"
        );
        assert!(
            scan_summary.directories_walked >= 6,
            "the ordinary noise directories should be walked: {:?}",
            scan_summary.directories_walked
        );
        assert_eq!(scan_summary.unreadable_directories, 0);
    }

    #[test]
    fn should_report_positive_allocated_bytes_for_the_found_candidates() {
        let fixture = install_scan_fixture("bytes");
        let mut far_future_progress_at = Instant::now() + std::time::Duration::from_secs(60);
        let scan_summary = scan_root_for_retired_candidates(
            &fixture.root_directory,
            Instant::now(),
            &mut far_future_progress_at,
        );
        assert!(
            scan_summary.retired_allocated_bytes > 0,
            "the legacy fixture carries one written file"
        );
    }
}

mod should_measure_allocated_bytes {
    use super::*;

    #[test]
    fn should_measure_a_whole_subtree_above_the_sum_of_its_files() {
        let fixture_root = std::env::temp_dir().join("retired-scan-bytes-subtree");
        let retired_directory =
            generation_out_directory(&fixture_root, "bytes1").join("mlx-c-runtime-build");
        let _ = fs::remove_dir_all(&fixture_root);
        fs::create_dir_all(retired_directory.join("CMakeFiles"))
            .expect("the fixture directory should be creatable");
        fs::write(retired_directory.join("build.ninja"), "fixture contents\n")
            .expect("the fixture file should be writable");
        fs::write(
            retired_directory.join("CMakeFiles/generated.data"),
            "fixture contents\n",
        )
        .expect("the fixture file should be writable");

        // Lower bound from the two files' own blocks: counting only the
        // top level and missing the nested CMakeFiles subtree would fail
        // this assertion.
        use std::os::unix::fs::MetadataExt;
        let expected_file_bytes: u64 = ["build.ninja", "CMakeFiles/generated.data"]
            .into_iter()
            .map(|fixture_relative_path| {
                fs::symlink_metadata(retired_directory.join(fixture_relative_path))
                    .expect("the fixture file metadata")
                    .blocks()
                    * 512
            })
            .sum();
        let measured_bytes = allocated_bytes_under(&retired_directory);
        assert!(
            measured_bytes >= expected_file_bytes,
            "measured {measured_bytes} must cover the file blocks {expected_file_bytes}"
        );

        let _ = fs::remove_dir_all(&fixture_root);
    }

    #[test]
    fn should_measure_an_empty_leaf_directory_as_zero_and_a_populated_one_as_positive() {
        // APFS allocates no blocks for an empty directory, so a leaf with
        // no content measures zero; adding one file must raise the count.
        let fixture_root = std::env::temp_dir().join("retired-scan-bytes-leaf");
        let retired_directory =
            generation_out_directory(&fixture_root, "bytes2").join("mlx-c-runtime-build");
        let _ = fs::remove_dir_all(&fixture_root);
        fs::create_dir_all(&retired_directory).expect("the fixture directory should be creatable");
        assert_eq!(
            allocated_bytes_under(&retired_directory),
            0,
            "an empty directory carries no allocated blocks on APFS"
        );
        fs::write(retired_directory.join("artifact.data"), "fixture contents\n")
            .expect("the fixture file should be writable");
        assert!(
            allocated_bytes_under(&retired_directory) > 0,
            "one file should register allocated blocks"
        );

        let _ = fs::remove_dir_all(&fixture_root);
    }
}
