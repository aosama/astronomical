use astronomical_test_harness::offline_test;

use crate::sccache_save::{SccacheSaveDecision, directory_size_bytes, save_decision};

offline_test!(fresh_generation_saves_when_no_stored_entry_exists, {
    assert_eq!(
        save_decision(100_000_000, 450_000_000, 0, 0, false),
        SccacheSaveDecision::NewGeneration
    );
});

offline_test!(stored_entry_within_budget_skips_the_save, {
    assert_eq!(
        save_decision(100_000_000, 450_000_000, 1, 0, false),
        SccacheSaveDecision::FreshEntryWithinBudget
    );
});

offline_test!(directory_over_budget_skips_the_save_even_after_deletions, {
    assert_eq!(
        save_decision(500_000_000, 450_000_000, 1, 1, false),
        SccacheSaveDecision::DirectoryOverBudget
    );
    assert_eq!(
        save_decision(500_000_000, 450_000_000, 0, 0, false),
        SccacheSaveDecision::DirectoryOverBudget
    );
});

offline_test!(deleting_an_oversize_entry_forces_a_replacement_save, {
    assert_eq!(
        save_decision(100_000_000, 450_000_000, 1, 1, false),
        SccacheSaveDecision::ReplaceOversizeEntry
    );
});

offline_test!(blocked_deletion_wins_over_every_other_decision, {
    assert_eq!(
        save_decision(100_000_000, 450_000_000, 0, 0, true),
        SccacheSaveDecision::ReplaceBlocked
    );
    assert!(!SccacheSaveDecision::ReplaceBlocked.should_save());
});

offline_test!(only_replacement_decisions_save, {
    assert!(SccacheSaveDecision::NewGeneration.should_save());
    assert!(SccacheSaveDecision::ReplaceOversizeEntry.should_save());
    assert!(!SccacheSaveDecision::FreshEntryWithinBudget.should_save());
    assert!(!SccacheSaveDecision::DirectoryOverBudget.should_save());
    assert!(!SccacheSaveDecision::CacheListUnavailable.should_save());
    for decision in [
        SccacheSaveDecision::NewGeneration,
        SccacheSaveDecision::ReplaceOversizeEntry,
        SccacheSaveDecision::FreshEntryWithinBudget,
        SccacheSaveDecision::DirectoryOverBudget,
        SccacheSaveDecision::ReplaceBlocked,
        SccacheSaveDecision::CacheListUnavailable,
    ] {
        assert!(!decision.as_decision_text().is_empty());
    }
});

offline_test!(
    directory_measurement_sums_apparent_file_sizes_recursively,
    {
        let temporary_directory = std::env::temp_dir().join(format!(
            "astronomical-sccache-save-measure-{}",
            std::process::id()
        ));
        let nested_directory = temporary_directory.join("nested/deeper");
        std::fs::create_dir_all(&nested_directory).unwrap();
        std::fs::write(temporary_directory.join("top"), vec![0u8; 100]).unwrap();
        std::fs::write(nested_directory.join("leaf"), vec![0u8; 30]).unwrap();
        let measured_bytes = directory_size_bytes(&temporary_directory);
        std::fs::remove_dir_all(&temporary_directory).unwrap();
        assert_eq!(measured_bytes, 130);
    }
);

offline_test!(missing_directory_measures_as_zero, {
    let absent_directory = std::env::temp_dir().join(format!(
        "astronomical-sccache-save-absent-{}",
        std::process::id()
    ));
    assert_eq!(directory_size_bytes(&absent_directory), 0);
});
