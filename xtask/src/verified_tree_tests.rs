use std::path::Path;

use super::{
    is_tree_already_verified, listing_proves_verified_tree, newest_default_branch_run_commit,
};

const TREE: &str = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";

fn listing(name: &str, expired: bool, repository_id: u64, head_repository_id: u64) -> String {
    serde_json::json!({
        "total_count": 1,
        "artifacts": [{
            "name": name,
            "expired": expired,
            "workflow_run": {
                "repository_id": repository_id,
                "head_repository_id": head_repository_id
            }
        }]
    })
    .to_string()
}

fn name_for_tree() -> String {
    format!("verified-tree-{TREE}")
}

#[test]
fn should_accept_a_live_record_of_this_tree_from_this_repository() {
    assert!(listing_proves_verified_tree(
        &listing(&name_for_tree(), false, 7, 7),
        TREE
    ));
}

#[test]
fn should_refuse_an_expired_record() {
    assert!(!listing_proves_verified_tree(
        &listing(&name_for_tree(), true, 7, 7),
        TREE
    ));
}

#[test]
fn should_refuse_a_record_from_a_fork() {
    assert!(!listing_proves_verified_tree(
        &listing(&name_for_tree(), false, 7, 8),
        TREE
    ));
}

#[test]
fn should_refuse_a_record_of_another_tree() {
    assert!(!listing_proves_verified_tree(
        &listing("verified-tree-0000", false, 7, 7),
        TREE
    ));
}

#[test]
fn should_refuse_a_lookalike_name() {
    assert!(!listing_proves_verified_tree(
        &listing(&format!("{}-extra", name_for_tree()), false, 7, 7),
        TREE
    ));
}

#[test]
fn should_refuse_an_empty_listing() {
    assert!(!listing_proves_verified_tree(r#"{"artifacts":[]}"#, TREE));
}

#[test]
fn should_refuse_a_record_with_no_repository_details() {
    let without_details = serde_json::json!({
        "artifacts": [{"name": name_for_tree(), "expired": false}]
    })
    .to_string();
    assert!(!listing_proves_verified_tree(&without_details, TREE));
}

#[test]
fn should_refuse_a_record_with_no_expiry() {
    let without_expiry = serde_json::json!({
        "artifacts": [{
            "name": name_for_tree(),
            "workflow_run": {"repository_id": 7, "head_repository_id": 7}
        }]
    })
    .to_string();
    assert!(!listing_proves_verified_tree(&without_expiry, TREE));
}

#[test]
fn should_refuse_output_that_is_not_a_listing() {
    assert!(!listing_proves_verified_tree("", TREE));
    assert!(!listing_proves_verified_tree("not json", TREE));
    assert!(!listing_proves_verified_tree(
        r#"{"message":"Not Found"}"#,
        TREE
    ));
}

#[test]
fn should_answer_not_verified_when_the_commit_cannot_be_read() {
    let not_a_repository = tempfile::tempdir().expect("a temporary directory");
    assert!(!is_tree_already_verified(
        not_a_repository.path(),
        "0000000000000000000000000000000000000000"
    ));
    assert!(!is_tree_already_verified(
        Path::new("/nonexistent-directory-for-this-test"),
        "HEAD"
    ));
}

fn default_branch_record(
    created_at: &str,
    commit: &str,
    expired: bool,
    head_repository_id: u64,
) -> serde_json::Value {
    serde_json::json!({
        "name": "default-branch-macos-run",
        "expired": expired,
        "created_at": created_at,
        "workflow_run": {
            "repository_id": 7,
            "head_repository_id": head_repository_id,
            "head_sha": commit
        }
    })
}

fn listing_of(records: &[serde_json::Value]) -> String {
    serde_json::json!({ "artifacts": records }).to_string()
}

#[test]
fn should_find_the_newest_default_branch_run_whatever_the_order() {
    let listing = listing_of(&[
        default_branch_record("2026-11-05T10:00:00Z", "older", false, 7),
        default_branch_record("2026-11-06T10:00:00Z", "newest", false, 7),
        default_branch_record("2026-11-04T10:00:00Z", "oldest", false, 7),
    ]);
    assert_eq!(
        newest_default_branch_run_commit(&listing).as_deref(),
        Some("newest")
    );
}

#[test]
fn should_pass_over_expired_and_foreign_default_branch_records() {
    let listing = listing_of(&[
        default_branch_record("2026-11-06T10:00:00Z", "expired", true, 7),
        default_branch_record("2026-11-06T11:00:00Z", "forked", false, 8),
        default_branch_record("2026-11-01T10:00:00Z", "live", false, 7),
    ]);
    assert_eq!(
        newest_default_branch_run_commit(&listing).as_deref(),
        Some("live")
    );
}

#[test]
fn should_know_no_default_branch_run_without_a_usable_record() {
    assert_eq!(newest_default_branch_run_commit(&listing_of(&[])), None);
    assert_eq!(newest_default_branch_run_commit("not json"), None);
    assert_eq!(
        newest_default_branch_run_commit(r#"{"message":"Not Found"}"#),
        None
    );
    let without_commit = serde_json::json!({"artifacts": [{
        "name": "default-branch-macos-run",
        "expired": false,
        "created_at": "2026-11-06T10:00:00Z",
        "workflow_run": {"repository_id": 7, "head_repository_id": 7}
    }]})
    .to_string();
    assert_eq!(newest_default_branch_run_commit(&without_commit), None);
}

#[test]
fn should_not_take_another_artifact_for_a_default_branch_run() {
    let other_artifact = serde_json::json!({"artifacts": [{
        "name": "verified-tree-abc",
        "expired": false,
        "created_at": "2026-11-06T10:00:00Z",
        "workflow_run": {"repository_id": 7, "head_repository_id": 7, "head_sha": "abc"}
    }]})
    .to_string();
    assert_eq!(newest_default_branch_run_commit(&other_artifact), None);
}
