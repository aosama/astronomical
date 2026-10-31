//! The cache-ownership half of the workflow structural contract: every cache
//! owner restores and saves through the shared composite action with matching
//! paths and keys, the owners stay disjoint, native identity and sccache keys
//! keep their deliberate couplings, and the prune step runs last with bounded
//! authority (issue #1250).

mod contract_support;

use contract_support::{
    composite_action_contract, find_step_by_id, find_step_by_name, is_pinned_action_reference,
    path_lines, required_field, required_job, required_text, required_u64, run_text_of, steps_of,
    workflow_contract,
};
use serde_yaml::Value;

/// Asserts one cache owner's restore step through the shared composite action
/// and accumulates its restored paths for the disjointness check. Returns the
/// restore path text and configuration so the save half can prove it matches.
fn assert_cache_restore_step<'a>(
    job_steps: &[&'a Value],
    owner_id: &str,
    expected_paths: &[&str],
    restored_paths: &mut Vec<String>,
) -> (&'a str, &'a Value) {
    let restore_step = find_step_by_id(job_steps, owner_id);
    let restore_uses = required_text(
        restore_step,
        "uses",
        &format!("cache owner {owner_id} restore"),
    );
    assert_eq!(
        restore_uses, "./.github/actions/astronomical-cache",
        "cache owner {owner_id} must restore through the shared composite action"
    );
    let restore_configuration = required_field(
        restore_step,
        "with",
        &format!("cache owner {owner_id} restore"),
    );
    assert_eq!(
        required_text(
            restore_configuration,
            "operation",
            &format!("cache owner {owner_id} restore")
        ),
        "restore",
        "cache owner {owner_id} must declare the restore operation"
    );
    let path_text = required_text(
        restore_configuration,
        "path",
        &format!("cache owner {owner_id} restore"),
    );
    for expected_path in expected_paths {
        assert!(
            path_text.contains(expected_path),
            "cache owner {owner_id} omits {expected_path}"
        );
    }
    assert!(
        !path_lines(path_text).contains(&"target"),
        "cache owner {owner_id} must not cache target"
    );
    for restored_path in path_lines(path_text) {
        restored_paths.push(restored_path.to_string());
    }
    let expected_report_owner = owner_id
        .strip_suffix("-cache")
        .expect("cache owner ids end in -cache");
    assert_eq!(
        required_text(
            restore_configuration,
            "owner",
            &format!("cache owner {owner_id} restore")
        ),
        expected_report_owner,
        "cache owner {owner_id} has the wrong report owner"
    );
    (path_text, restore_configuration)
}

/// Asserts the matching save half: the same composite action, the save
/// operation, a bounded save condition, and paths and keys identical to the
/// restore half.
fn assert_cache_save_step_matches_restore(
    job_steps: &[&Value],
    owner_id: &str,
    restore_path_text: &str,
    restore_configuration: &Value,
) {
    let save_step_id = format!("{owner_id}-save");
    let save_step = find_step_by_id(job_steps, &save_step_id);
    let save_uses = required_text(
        save_step,
        "uses",
        &format!("cache owner {save_step_id} save"),
    );
    assert_eq!(
        save_uses, "./.github/actions/astronomical-cache",
        "cache owner {owner_id} must save through the shared composite action"
    );
    let save_configuration = required_field(
        save_step,
        "with",
        &format!("cache owner {save_step_id} save"),
    );
    assert_eq!(
        required_text(
            save_configuration,
            "operation",
            &format!("cache owner {save_step_id} save")
        ),
        "save",
        "cache owner {owner_id} must declare the save operation"
    );
    let save_condition =
        required_text(save_step, "if", &format!("cache owner {save_step_id} save"));
    assert!(
        save_condition.contains("success()") || save_condition.contains("!cancelled()"),
        "cache owner {owner_id} save is unconditional"
    );
    assert_eq!(
        required_text(
            save_configuration,
            "path",
            &format!("cache owner {save_step_id} save")
        ),
        restore_path_text,
        "cache owner {owner_id} save path diverges from its restore path"
    );
    assert_eq!(
        required_text(
            save_configuration,
            "key",
            &format!("cache owner {save_step_id} save")
        ),
        required_text(
            restore_configuration,
            "key",
            &format!("cache owner {owner_id} restore")
        ),
        "cache owner {owner_id} save key diverges from its restore key"
    );
}

/// Asserts one cache owner's restore/save pair through the shared composite
/// action and accumulates the restored paths for the disjointness check.
fn assert_composite_cache_owner(
    job_steps: &[&Value],
    owner_id: &str,
    expected_paths: &[&str],
    restored_paths: &mut Vec<String>,
) {
    let (restore_path_text, restore_configuration) =
        assert_cache_restore_step(job_steps, owner_id, expected_paths, restored_paths);
    assert_cache_save_step_matches_restore(
        job_steps,
        owner_id,
        restore_path_text,
        restore_configuration,
    );
}

#[test]
fn cache_owners_are_disjoint_composite_users() {
    let workflow = workflow_contract();
    let verification_steps = steps_of(required_job(&workflow, "rust-core"), "rust-core");
    let swift_steps = steps_of(required_job(&workflow, "swift-node"), "swift-node");

    let mut restored_paths: Vec<String> = Vec::new();
    assert_composite_cache_owner(
        &verification_steps,
        "cargo-downloads-cache",
        &["~/.cargo/registry", "~/.cargo/git"],
        &mut restored_paths,
    );
    assert_composite_cache_owner(
        &verification_steps,
        "native-archives-cache",
        &["~/Library/Caches/Astronomical/native-dependencies"],
        &mut restored_paths,
    );
    assert_composite_cache_owner(
        &verification_steps,
        "native-build-cache",
        &["env.NATIVE_BUILD_ENTRY_DIRECTORY"],
        &mut restored_paths,
    );
    assert_composite_cache_owner(
        &verification_steps,
        "sccache-cache",
        &["~/Library/Caches/Astronomical/sccache"],
        &mut restored_paths,
    );
    assert_composite_cache_owner(
        &verification_steps,
        "aws-lc-sys-cache",
        &[
            "target/debug/build/aws-lc-sys-*",
            "target/debug/.fingerprint/aws-lc-sys-*",
            "target/aarch64-apple-darwin/debug/build/aws-lc-sys-*",
            "target/aarch64-apple-darwin/debug/.fingerprint/aws-lc-sys-*",
        ],
        &mut restored_paths,
    );
    assert_composite_cache_owner(
        &swift_steps,
        "swiftpm-cache",
        &["apps/astronomical-menu/.build"],
        &mut restored_paths,
    );
    assert_composite_cache_owner(
        &swift_steps,
        "thin-talk-cache",
        &["apps/thin-talk/.build"],
        &mut restored_paths,
    );
    let mut deduplicated_paths = restored_paths.clone();
    deduplicated_paths.sort();
    deduplicated_paths.dedup();
    assert_eq!(
        restored_paths.len(),
        deduplicated_paths.len(),
        "cache owners overlap paths"
    );
}

#[test]
fn native_and_sccache_keys_keep_their_identity_couplings() {
    let workflow = workflow_contract();
    let verification_steps = steps_of(required_job(&workflow, "rust-core"), "rust-core");

    let native_configuration = required_field(
        find_step_by_id(&verification_steps, "native-build-cache"),
        "with",
        "native-build-cache restore",
    );
    if let Some(restore_keys) = native_configuration.get("restore-keys") {
        assert!(
            restore_keys
                .as_str()
                .unwrap_or_default()
                .contains("NATIVE_BUILD_IDENTITY"),
            "native build cache must not cross identities"
        );
    }
    assert!(
        required_text(native_configuration, "path", "native-build-cache restore")
            .contains("NATIVE_BUILD_ENTRY_DIRECTORY"),
        "native entry path is not identity-specific"
    );
    assert!(
        required_text(native_configuration, "key", "native-build-cache restore")
            .contains("NATIVE_BUILD_IDENTITY"),
        "native key omits full identity"
    );

    let sccache_configuration = required_field(
        find_step_by_id(&verification_steps, "sccache-cache"),
        "with",
        "sccache-cache restore",
    );
    let sccache_key = required_text(sccache_configuration, "key", "sccache-cache restore");
    assert!(
        sccache_key.contains("Cargo.lock"),
        "sccache key is not stable by dependency graph"
    );
    assert!(
        !sccache_key.contains("NATIVE_BUILD_IDENTITY"),
        "sccache key couples to the native identity; the native build product cache owns that identity"
    );
    let sccache_fallbacks: Vec<&str> = required_text(
        sccache_configuration,
        "restore-keys",
        "sccache-cache restore",
    )
    .split(['\n', '\r'])
    .map(str::trim)
    .filter(|line| !line.is_empty())
    .collect();
    assert!(
        sccache_fallbacks
            .iter()
            .any(|fallback| !fallback.contains("NATIVE_BUILD_IDENTITY")),
        "sccache restore omits an identity-free fallback"
    );

    let swift_configuration = required_field(
        find_step_by_id(
            &steps_of(required_job(&workflow, "swift-node"), "swift-node"),
            "swiftpm-cache",
        ),
        "with",
        "swiftpm-cache restore",
    );
    let swift_key = required_text(swift_configuration, "key", "swiftpm-cache restore");
    assert!(
        !swift_key.contains("Cargo"),
        "Swift state is coupled to Rust"
    );
    assert!(
        swift_key.contains("SWIFT_TOOLCHAIN_IDENTITY"),
        "Swift cache omits toolchain compatibility"
    );

    let thin_talk_configuration = required_field(
        find_step_by_id(
            &steps_of(required_job(&workflow, "swift-node"), "swift-node"),
            "thin-talk-cache",
        ),
        "with",
        "thin-talk-cache restore",
    );
    let thin_talk_key = required_text(thin_talk_configuration, "key", "thin-talk-cache restore");
    assert!(
        !thin_talk_key.contains("Cargo"),
        "Thin Talk Swift state must not be coupled to Rust"
    );
    assert!(
        thin_talk_key.contains("SWIFT_TOOLCHAIN_IDENTITY"),
        "Thin Talk cache omits toolchain compatibility"
    );
    // The thin-talk key must live in its OWN prune family: the prune derives
    // each entry's family from the key prefix before `-macOS` and keeps one
    // entry per family, so a shared `astronomical-v2-swiftpm-` prefix made
    // the prune evict menu and thin-talk in alternation (2026-10-31 hunt).
    assert!(
        thin_talk_key.starts_with("astronomical-v2-swiftpm-thin-talk-"),
        "thin-talk cache shares the swiftpm prune family and will be evicted in alternation with the menu cache"
    );
}

#[test]
fn aws_lc_sys_key_keeps_toolchain_and_lockfile_couplings() {
    let workflow = workflow_contract();
    let verification_steps = steps_of(required_job(&workflow, "rust-core"), "rust-core");
    let aws_lc_sys_configuration = required_field(
        find_step_by_id(&verification_steps, "aws-lc-sys-cache"),
        "with",
        "aws-lc-sys-cache restore",
    );
    let aws_lc_sys_key = required_text(aws_lc_sys_configuration, "key", "aws-lc-sys-cache restore");
    assert!(
        aws_lc_sys_key.contains("Cargo.lock"),
        "aws-lc-sys build-script cache is not stable by dependency graph"
    );
    assert!(
        aws_lc_sys_key.contains("rust-toolchain.toml"),
        "aws-lc-sys build-script cache omits the compiler identity its fingerprints embed"
    );
    assert!(
        !aws_lc_sys_key.contains("NATIVE_BUILD_IDENTITY"),
        "aws-lc-sys build-script cache couples to the native build identity"
    );
}

#[test]
fn cache_prune_runs_last_with_bounded_authority() {
    let workflow = workflow_contract();
    let verification_job = required_job(&workflow, "rust-core");
    let verification_steps = steps_of(verification_job, "rust-core");

    let prune_step = find_step_by_name(&verification_steps, "Prune surplus CI caches");
    assert!(
        required_text(prune_step, "if", "cache prune step").contains("!cancelled()"),
        "cache prune must run even when earlier steps fail"
    );
    assert_eq!(
        required_text(
            required_field(prune_step, "env", "cache prune step"),
            "GH_TOKEN",
            "cache prune env"
        ),
        "${{ github.token }}",
        "cache prune lacks the workflow token"
    );
    assert!(
        run_text_of(prune_step, "prune-caches").contains("prune-caches"),
        "cache prune does not call the xtask prune"
    );
    assert!(
        required_u64(prune_step, "timeout-minutes", "cache prune step") <= 2,
        "cache prune exceeded its bounded timeout"
    );
    let prune_step_position = verification_steps
        .iter()
        .position(|step| {
            step.get("name").and_then(Value::as_str) == Some("Prune surplus CI caches")
        })
        .expect("cache prune step position");
    let final_save_position = verification_steps
        .iter()
        .position(|step| step.get("id").and_then(Value::as_str) == Some("sccache-cache-save"))
        .expect("sccache-cache-save position");
    assert!(
        prune_step_position > final_save_position,
        "cache prune must run after the final cache save"
    );

    let verification_permissions = required_field(verification_job, "permissions", "job rust-core");
    assert_eq!(
        required_text(
            verification_permissions,
            "contents",
            "rust-core permissions"
        ),
        "read",
        "required check must keep contents read-only"
    );
    assert_eq!(
        required_text(verification_permissions, "actions", "rust-core permissions"),
        "write",
        "cache prune needs actions write permission"
    );
}

#[test]
fn shared_cache_action_is_pinned_and_reported() {
    let composite = composite_action_contract();
    let composite_runs = required_field(&composite, "runs", "the shared cache action");
    assert_eq!(
        required_text(composite_runs, "using", "the shared cache action runs"),
        "composite",
        "shared cache action must run as a composite"
    );
    let composite_steps = steps_of(composite_runs, "the shared cache action");
    let composite_restore = find_step_by_id(&composite_steps, "restore");
    assert!(
        is_pinned_action_reference(
            required_text(composite_restore, "uses", "shared cache action restore"),
            "actions/cache/restore",
        ),
        "shared cache action restore is not a pinned actions/cache/restore"
    );
    let composite_save = find_step_by_id(&composite_steps, "save");
    assert!(
        is_pinned_action_reference(
            required_text(composite_save, "uses", "shared cache action save"),
            "actions/cache/save",
        ),
        "shared cache action save is not a pinned actions/cache/save"
    );
    let composite_report = find_step_by_id(&composite_steps, "report");
    let report_environment = required_field(composite_report, "env", "shared cache action report");
    assert!(
        required_text(
            report_environment,
            "CACHE_OWNER",
            "shared cache action report env"
        )
        .contains("inputs.owner"),
        "shared cache action report does not forward the owner"
    );
    let reported_outcome = required_text(
        report_environment,
        "CACHE_STEP_OUTCOME",
        "shared cache action report env",
    );
    assert!(
        reported_outcome.contains("steps.restore.outcome"),
        "shared cache action report does not forward the restore outcome"
    );
    assert!(
        reported_outcome.contains("steps.save.outcome"),
        "shared cache action report does not forward the save outcome"
    );
    assert!(
        run_text_of(composite_report, "report").contains("report-build-cache-restoration.sh"),
        "shared cache action report must call the restoration reporter"
    );
    assert!(
        required_text(
            required_field(
                required_field(&composite, "outputs", "the shared cache action"),
                "cache-hit",
                "shared cache action outputs"
            ),
            "value",
            "shared cache action cache-hit output"
        )
        .contains("steps.restore.outputs.cache-hit"),
        "shared cache action must surface cache hits to callers"
    );
}
