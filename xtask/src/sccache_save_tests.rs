use crate::sccache_save::{SccacheSaveDecision, directory_size_bytes, save_decision};

#[test]
fn should_save_a_fresh_generation_when_no_stored_entry_exists() {
    assert_eq!(
        save_decision(100_000_000, 0, 0, false),
        SccacheSaveDecision::NewGeneration
    );
}

#[test]
fn should_skip_the_save_when_a_stored_entry_is_within_budget() {
    assert_eq!(
        save_decision(100_000_000, 1, 0, false),
        SccacheSaveDecision::FreshEntryWithinBudget
    );
}

#[test]
fn should_skip_the_save_when_the_directory_is_over_budget_even_after_deletions() {
    assert_eq!(
        save_decision(680_000_001, 1, 1, false),
        SccacheSaveDecision::DirectoryOverBudget
    );
    assert_eq!(
        save_decision(680_000_001, 0, 0, false),
        SccacheSaveDecision::DirectoryOverBudget
    );
}

#[test]
fn should_save_directories_at_the_fixed_budget_boundary() {
    for directory_bytes in [0, 679_999_999, 680_000_000] {
        assert_eq!(
            save_decision(directory_bytes, 0, 0, false),
            SccacheSaveDecision::NewGeneration
        );
    }
}

#[test]
fn should_replace_a_deleted_oversize_entry() {
    assert_eq!(
        save_decision(100_000_000, 1, 1, false),
        SccacheSaveDecision::ReplaceOversizeEntry
    );
}

#[test]
fn should_prioritize_blocked_deletion_over_every_other_decision() {
    assert_eq!(
        save_decision(100_000_000, 0, 0, true),
        SccacheSaveDecision::ReplaceBlocked
    );
    assert!(!SccacheSaveDecision::ReplaceBlocked.should_save());
}

#[test]
fn should_save_only_new_or_replacement_generations() {
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
}

#[test]
fn should_sum_apparent_file_sizes_recursively() {
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

#[test]
fn should_measure_a_missing_directory_as_zero() {
    let absent_directory = std::env::temp_dir().join(format!(
        "astronomical-sccache-save-absent-{}",
        std::process::id()
    ));
    assert_eq!(directory_size_bytes(&absent_directory), 0);
}
