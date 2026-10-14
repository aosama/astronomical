use astronomical_test_harness::offline_test;
use std::path::Path;

use crate::bootstrap_native_dependencies::{
    ProvisionDecision, decide_provision, has_required_free_space, parse_arguments,
    parse_available_kib_blocks, resolve_cache_directory, validate_downloaded_size,
};

offline_test!(df_output_available_blocks_parse, {
    let macos_style = "Filesystem 1024-blocks Used Available Capacity iused ifree %iused  Mounted on\n/dev/disk1s1 994688456 685980400 308681056 69% 3144123 4872668253    0%   /System/Volumes/Data\n";
    assert_eq!(
        parse_available_kib_blocks(macos_style).unwrap(),
        308_681_056
    );
    let linux_style = "Filesystem 1K-blocks      Used Available Use% Mounted on\n/dev/nvme0n1p2 498433024 185388868 287570860  40% /\n";
    assert_eq!(
        parse_available_kib_blocks(linux_style).unwrap(),
        287_570_860
    );
    assert!(
        parse_available_kib_blocks("Filesystem 1K-blocks Used Available Use% Mounted on\n")
            .is_err()
    );
    assert!(
        parse_available_kib_blocks(
            "Filesystem 1K-blocks Used Available Use% Mounted on\n/dev/disk1s1 a b c d /\n"
        )
        .is_err()
    );
    assert!(parse_available_kib_blocks("").is_err());
});

offline_test!(free_space_boundary_is_two_gibibytes_of_kib_blocks, {
    let minimum_kib_blocks = 2_147_483_648u64 / 1_024;
    assert!(!has_required_free_space(minimum_kib_blocks - 1));
    assert!(has_required_free_space(minimum_kib_blocks));
    // Saturating math: a df output that overflows the multiply must read as
    // insufficient rather than wrapping negative.
    assert!(has_required_free_space(u64::MAX / 1_024));
});

offline_test!(downloaded_size_bound_is_256_mib, {
    assert!(validate_downloaded_size(268_435_456, "MLX").is_ok());
    assert!(validate_downloaded_size(268_435_457, "MLX").is_err());
});

offline_test!(provision_decision_prefers_a_verified_cache, {
    let archive_path = Path::new("/cache/mlx.tar.gz");
    assert_eq!(
        decide_provision(true, true, true, false, "MLX", archive_path).unwrap(),
        ProvisionDecision::VerifiedCached
    );
    assert_eq!(
        decide_provision(true, true, true, true, "MLX", archive_path).unwrap(),
        ProvisionDecision::VerifiedCached
    );
});

offline_test!(verify_only_refuses_missing_or_invalid_cache, {
    let archive_path = Path::new("/cache/mlx.tar.gz");
    assert!(
        decide_provision(false, false, false, true, "MLX", archive_path)
            .unwrap_err()
            .contains("missing or invalid MLX archive")
    );
    assert!(
        decide_provision(true, true, false, true, "MLX", archive_path)
            .unwrap_err()
            .contains("missing or invalid MLX archive")
    );
});

offline_test!(non_regular_cache_entry_is_refused_even_when_downloading, {
    let archive_path = Path::new("/cache/mlx.tar.gz");
    let directory_error =
        decide_provision(true, false, false, false, "MLX", archive_path).unwrap_err();
    assert!(directory_error.contains("not a regular file"));
    let symlink_error =
        decide_provision(true, false, false, true, "MLX", archive_path).unwrap_err();
    assert!(symlink_error.contains("not a regular file"));
});

offline_test!(download_is_chosen_when_cache_is_absent_or_stale, {
    let archive_path = Path::new("/cache/mlx.tar.gz");
    assert_eq!(
        decide_provision(false, false, false, false, "MLX", archive_path).unwrap(),
        ProvisionDecision::Download
    );
    assert_eq!(
        decide_provision(true, true, false, false, "MLX", archive_path).unwrap(),
        ProvisionDecision::Download
    );
});

offline_test!(
    argument_parsing_accepts_cache_dir_verify_and_rejects_unknowns,
    {
        let options = parse_arguments(&[
            "--verify".to_owned(),
            "--cache-dir".to_owned(),
            "/tmp/cache".to_owned(),
        ])
        .unwrap();
        assert_eq!(options.cache_directory.as_deref(), Some("/tmp/cache"));
        assert!(options.is_verify_only);

        let options =
            parse_arguments(&["--cache-dir".to_owned(), "/tmp/cache".to_owned()]).unwrap();
        assert_eq!(options.cache_directory.as_deref(), Some("/tmp/cache"));
        assert!(!options.is_verify_only);

        assert!(parse_arguments(&["--cache-dir".to_owned()]).is_err());
        assert!(parse_arguments(&["--unknown".to_owned()]).is_err());
        assert!(parse_arguments(&[]).unwrap().cache_directory.is_none());
    }
);

offline_test!(cache_directory_resolution_falls_back_to_home, {
    let resolved = resolve_cache_directory(None, Some("/Users/builder")).unwrap();
    assert!(resolved.ends_with("Library/Caches/Astronomical/native-dependencies"));

    let resolved = resolve_cache_directory(Some("/tmp/overlay-cache"), None).unwrap();
    assert_eq!(resolved, Path::new("/tmp/overlay-cache"));

    assert!(resolve_cache_directory(Some("relative/cache"), None).is_err());
    assert!(resolve_cache_directory(None, None).is_err());
});
