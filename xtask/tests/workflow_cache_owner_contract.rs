//! The cache-ownership half of the workflow structural contract: every cache
//! owner restores and saves through the shared composite action with matching
//! paths and keys, the owners stay disjoint, native identity and sccache keys
//! keep their deliberate couplings, and the prune step runs last with bounded
//! authority (issue #1250).

mod contract_support;

use contract_support::{
    composite_action_contract, find_step_by_id, find_step_by_name, is_pinned_action_reference,
    path_lines, repository_root, required_field, required_job, required_text, required_u64,
    run_text_of, steps_of, workflow_contract,
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
    let neural_engine_steps = steps_of(required_job(&workflow, "neural-engine"), "neural-engine");

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
    assert_composite_cache_owner(
        &neural_engine_steps,
        "neural-engine-sccache-cache",
        &["~/Library/Caches/Astronomical/neural-engine-sccache"],
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

const SCCACHE_KEY_PREFIX: &str = "astronomical-v2-sccache-";
const TOOL_PINS_MANIFEST: &str = "third-party/tool-pins/Cargo.toml";

/// Every line of `step_text` that names an sccache cache key but does not
/// hash the tool-pin manifest into it.
fn sccache_key_lines_missing_tool_pins(step_text: &str) -> Vec<String> {
    let mut uncoupled_key_lines = Vec::new();
    for step_line in step_text.lines() {
        if step_line.contains(SCCACHE_KEY_PREFIX) && !step_line.contains(TOOL_PINS_MANIFEST) {
            uncoupled_key_lines.push(step_line.trim().to_owned());
        }
    }
    uncoupled_key_lines
}

#[test]
fn should_flag_only_sccache_key_lines_without_the_tool_pin_hash() {
    let coupled_key = "key: astronomical-v2-sccache-${{ hashFiles('rust-toolchain.toml', 'third-party/tool-pins/Cargo.toml') }}";
    assert!(sccache_key_lines_missing_tool_pins(coupled_key).is_empty());
    let uncoupled_key = "key: astronomical-v2-sccache-${{ hashFiles('rust-toolchain.toml') }}";
    assert_eq!(sccache_key_lines_missing_tool_pins(uncoupled_key).len(), 1);
    let other_family_key =
        "key: astronomical-v2-cargo-downloads-${{ hashFiles('rust-toolchain.toml') }}";
    assert!(sccache_key_lines_missing_tool_pins(other_family_key).is_empty());
    assert!(sccache_key_lines_missing_tool_pins("").is_empty());
}

// sccache releases change its internal cache version, so an sccache bump
// must rotate every sccache key: restore, fallbacks, budget decision, and
// save. A key that missed the rotation would keep restoring the old,
// unusable entry on an unchanged primary key and skip the save forever.
#[test]
fn should_rotate_every_sccache_key_with_the_tool_pin_manifest() {
    let workflow = workflow_contract();
    let build_steps = steps_of(required_job(&workflow, "rust-core"), "rust-core");
    let mut inspected_key_line_count = 0;
    for step in &build_steps {
        let mut step_texts: Vec<&str> = Vec::new();
        if let Some(run_text) = step.get("run").and_then(Value::as_str) {
            step_texts.push(run_text);
        }
        if let Some(step_inputs) = step.get("with") {
            for input_name in ["key", "restore-keys"] {
                if let Some(input_text) = step_inputs.get(input_name).and_then(Value::as_str) {
                    step_texts.push(input_text);
                }
            }
        }
        for step_text in step_texts {
            inspected_key_line_count += step_text.matches(SCCACHE_KEY_PREFIX).count();
            let uncoupled_key_lines = sccache_key_lines_missing_tool_pins(step_text);
            assert!(
                uncoupled_key_lines.is_empty(),
                "sccache keys do not hash the tool-pin manifest: {uncoupled_key_lines:?}"
            );
        }
    }
    assert_eq!(
        inspected_key_line_count, 5,
        "expected the restore key, two fallbacks, the budget key, and the save key"
    );
}

const NEURAL_ENGINE_SCCACHE_KEY_PREFIX: &str = "astronomical-v2-neural-engine-sccache-";
const SCCACHE_TOOL_NAME: &str = "sccache";

/// The exact version a tool-pin manifest line pins, e.g. `0.19.0` from
/// `sccache = { version = "=0.19.0", ... }`; `None` when the manifest pins
/// no such tool. The name must match the whole key so `sccache-dist` never
/// answers for `sccache`.
fn pinned_tool_version(manifest_text: &str, tool_name: &str) -> Option<String> {
    for manifest_line in manifest_text.lines() {
        let Some((line_key, line_value)) = manifest_line.split_once('=') else {
            continue;
        };
        if line_key.trim() != tool_name {
            continue;
        }
        let Some((_, after_version_marker)) = line_value.split_once("version = \"=") else {
            continue;
        };
        let Some((pinned_version, _)) = after_version_marker.split_once('"') else {
            continue;
        };
        return Some(pinned_version.to_owned());
    }
    None
}

#[test]
fn should_read_the_exact_pinned_tool_version() {
    let manifest_text = "sccache = { version = \"=0.19.0\", default-features = false }";
    assert_eq!(
        pinned_tool_version(manifest_text, SCCACHE_TOOL_NAME),
        Some("0.19.0".to_owned())
    );
}

#[test]
fn should_not_answer_for_a_tool_the_manifest_does_not_pin() {
    let manifest_text = "cargo-about = { version = \"=0.8.0\" }";
    assert_eq!(pinned_tool_version(manifest_text, SCCACHE_TOOL_NAME), None);
    assert_eq!(pinned_tool_version("", SCCACHE_TOOL_NAME), None);
}

#[test]
fn should_not_answer_for_a_version_range_that_is_not_an_exact_pin() {
    let manifest_text = "sccache = { version = \"0.19\" }";
    assert_eq!(pinned_tool_version(manifest_text, SCCACHE_TOOL_NAME), None);
}

#[test]
fn should_not_match_a_tool_whose_name_only_starts_with_the_requested_name() {
    let manifest_text =
        "sccache-dist = { version = \"=9.9.9\" }\nsccache = { version = \"=0.19.0\" }";
    assert_eq!(
        pinned_tool_version(manifest_text, SCCACHE_TOOL_NAME),
        Some("0.19.0".to_owned())
    );
}

// The neural-engine lane's compiler cache must stay its own bounded family:
// a key under the build job's `astronomical-v2-sccache-` prefix would be
// restored by the build job's fallbacks and pruned as a build-job generation;
// a pull-request save would duplicate the default-branch entry; and an
// sccache binary off the tool pin would write a cache format the key does not
// rotate on.
#[test]
fn should_keep_the_neural_engine_compiler_cache_in_its_own_default_branch_family() {
    let workflow = workflow_contract();
    let neural_engine_steps = steps_of(required_job(&workflow, "neural-engine"), "neural-engine");
    let restore_configuration = required_field(
        find_step_by_id(&neural_engine_steps, "neural-engine-sccache-cache"),
        "with",
        "neural-engine-sccache-cache restore",
    );
    let restore_key = required_text(
        restore_configuration,
        "key",
        "neural-engine-sccache-cache restore",
    );
    assert!(
        restore_key.starts_with(NEURAL_ENGINE_SCCACHE_KEY_PREFIX),
        "neural-engine compiler cache left its own prune family: {restore_key}"
    );
    assert!(
        !restore_key.starts_with(SCCACHE_KEY_PREFIX),
        "neural-engine compiler cache shares the build job's sccache family"
    );
    assert!(
        restore_key.contains(TOOL_PINS_MANIFEST) && restore_key.contains("Cargo.lock"),
        "neural-engine compiler cache key does not rotate with the tool pins and dependency graph"
    );
    let save_condition = required_text(
        find_step_by_id(&neural_engine_steps, "neural-engine-sccache-cache-save"),
        "if",
        "neural-engine-sccache-cache save",
    );
    assert!(
        save_condition.contains("github.event.repository.default_branch"),
        "neural-engine compiler cache saves outside the default branch"
    );
    let manifest_text = std::fs::read_to_string(repository_root().join(TOOL_PINS_MANIFEST))
        .expect("the tool-pin manifest should be readable");
    let pinned_sccache_version = pinned_tool_version(&manifest_text, SCCACHE_TOOL_NAME)
        .expect("the tool-pin manifest pins sccache");
    let provision_step = find_step_by_name(&neural_engine_steps, "Provision pinned sccache");
    assert!(
        is_pinned_action_reference(
            required_text(provision_step, "uses", "Provision pinned sccache"),
            "taiki-e/install-action"
        ),
        "neural-engine sccache is not provisioned through the SHA-pinned install action"
    );
    let provision_inputs = required_field(provision_step, "with", "Provision pinned sccache");
    assert_eq!(
        required_text(provision_inputs, "tool", "Provision pinned sccache"),
        format!("{SCCACHE_TOOL_NAME}@{pinned_sccache_version}"),
        "neural-engine sccache drifted from the tool pin"
    );
    assert_eq!(
        required_text(provision_inputs, "fallback", "Provision pinned sccache"),
        "none",
        "neural-engine sccache may silently build from source"
    );
}
