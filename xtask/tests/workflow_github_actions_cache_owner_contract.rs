//! The `github_actions_cache` ownership half of the workflow structural
//! contract: every owner restores and saves through the shared composite
//! action with matching paths and keys, the owners stay disjoint, native
//! identity and `sccache` keys keep their deliberate couplings, and the prune
//! step runs last with bounded authority (issue #1250).

mod contract_support;
mod github_actions_cache_owner_assertions;

use contract_support::{
    composite_action_contract, find_step_by_id, find_step_by_name, is_pinned_action_reference,
    path_lines, repository_root, required_field, required_job, required_text, required_u64,
    run_text_of, steps_of, workflow_contract,
};
use github_actions_cache_owner_assertions::{
    assert_composite_report_forwards_owner_and_classification,
    assert_composite_restore_and_save_steps_are_pinned, assert_composite_surfaces_hits_to_callers,
    assert_native_build_entry_is_identity_specific,
    assert_sccache_key_follows_dependencies_not_native_identity,
    assert_swiftpm_key_is_decoupled_from_rust,
    assert_thin_talk_key_is_decoupled_and_owns_its_prune_family,
};
use serde_yaml::Value;

/// Asserts one `github_actions_cache` owner's restore step through the shared
/// composite action and accumulates its restored paths for the disjointness
/// check. Returns the restore path text and configuration so the save half
/// can prove it matches.
fn assert_github_actions_cache_restore_step<'a>(
    job_steps: &[&'a Value],
    owner_id: &str,
    expected_paths: &[&str],
    restored_paths: &mut Vec<String>,
) -> (&'a str, &'a Value) {
    let restore_step = find_step_by_id(job_steps, owner_id);
    let restore_uses = required_text(
        restore_step,
        "uses",
        &format!("github_actions_cache owner {owner_id} restore"),
    );
    assert_eq!(
        restore_uses, "./.github/actions/astronomical-cache",
        "github_actions_cache owner {owner_id} must restore through the shared composite action"
    );
    let restore_configuration = required_field(
        restore_step,
        "with",
        &format!("github_actions_cache owner {owner_id} restore"),
    );
    assert_eq!(
        required_text(
            restore_configuration,
            "operation",
            &format!("github_actions_cache owner {owner_id} restore")
        ),
        "restore",
        "github_actions_cache owner {owner_id} must declare the restore operation"
    );
    let path_text = required_text(
        restore_configuration,
        "path",
        &format!("github_actions_cache owner {owner_id} restore"),
    );
    for expected_path in expected_paths {
        assert!(
            path_text.contains(expected_path),
            "github_actions_cache owner {owner_id} omits {expected_path}"
        );
    }
    assert!(
        !path_lines(path_text).contains(&"target"),
        "github_actions_cache owner {owner_id} must not save target in the github_actions_cache"
    );
    for restored_path in path_lines(path_text) {
        restored_paths.push(restored_path.to_string());
    }
    let expected_report_owner = owner_id
        .strip_suffix("-cache")
        .expect("github_actions_cache owner ids end in -cache");
    assert_eq!(
        required_text(
            restore_configuration,
            "owner",
            &format!("github_actions_cache owner {owner_id} restore")
        ),
        expected_report_owner,
        "github_actions_cache owner {owner_id} has the wrong report owner"
    );
    (path_text, restore_configuration)
}

/// Asserts the matching save half: the same composite action, the save
/// operation, a bounded save condition, and paths and keys identical to the
/// restore half.
fn assert_github_actions_cache_save_step_matches_restore(
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
        &format!("github_actions_cache owner {save_step_id} save"),
    );
    assert_eq!(
        save_uses, "./.github/actions/astronomical-cache",
        "github_actions_cache owner {owner_id} must save through the shared composite action"
    );
    let save_configuration = required_field(
        save_step,
        "with",
        &format!("github_actions_cache owner {save_step_id} save"),
    );
    assert_eq!(
        required_text(
            save_configuration,
            "operation",
            &format!("github_actions_cache owner {save_step_id} save")
        ),
        "save",
        "github_actions_cache owner {owner_id} must declare the save operation"
    );
    let save_condition = required_text(
        save_step,
        "if",
        &format!("github_actions_cache owner {save_step_id} save"),
    );
    assert!(
        save_condition.contains("success()") || save_condition.contains("!cancelled()"),
        "github_actions_cache owner {owner_id} save is unconditional"
    );
    assert_eq!(
        required_text(
            save_configuration,
            "path",
            &format!("github_actions_cache owner {save_step_id} save")
        ),
        restore_path_text,
        "github_actions_cache owner {owner_id} save path diverges from its restore path"
    );
    assert_eq!(
        required_text(
            save_configuration,
            "key",
            &format!("github_actions_cache owner {save_step_id} save")
        ),
        required_text(
            restore_configuration,
            "key",
            &format!("github_actions_cache owner {owner_id} restore")
        ),
        "github_actions_cache owner {owner_id} save key diverges from its restore key"
    );
}

/// Asserts one `github_actions_cache` owner's restore/save pair through the
/// shared composite action and accumulates the restored paths for the
/// disjointness check.
fn assert_composite_github_actions_cache_owner(
    job_steps: &[&Value],
    owner_id: &str,
    expected_paths: &[&str],
    restored_paths: &mut Vec<String>,
) {
    let (restore_path_text, restore_configuration) = assert_github_actions_cache_restore_step(
        job_steps,
        owner_id,
        expected_paths,
        restored_paths,
    );
    assert_github_actions_cache_save_step_matches_restore(
        job_steps,
        owner_id,
        restore_path_text,
        restore_configuration,
    );
}

/// How the shared astronomical-cache action classifies an operation, as the
/// one expression its report step evaluates (issue #2635 replaced the shell
/// reporter that decided it): a save is published or an error; a failed
/// restore is an error; an exact hit on the primary key is primary; a hit on a
/// restore-keys prefix is fallback; anything else is a miss. Branch order is
/// the rule, so the whole expression is pinned.
const EXPECTED_CACHE_CLASSIFICATION: &str = "${{ inputs.operation == 'save' && (steps.save.outcome == 'success' && 'published' || 'error') || steps.restore.outcome != 'success' && 'error' || steps.restore.outputs.cache-hit == 'true' && 'primary' || steps.restore.outputs.cache-matched-key != '' && 'fallback' || 'miss' }}";

#[test]
fn github_actions_cache_owners_are_disjoint_composite_users() {
    let workflow = workflow_contract();
    let verification_steps = steps_of(required_job(&workflow, "rust-core"), "rust-core");
    let swift_steps = steps_of(required_job(&workflow, "swift-node"), "swift-node");
    let neural_engine_steps = steps_of(required_job(&workflow, "neural-engine"), "neural-engine");

    let mut restored_paths: Vec<String> = Vec::new();
    assert_composite_github_actions_cache_owner(
        &verification_steps,
        "cargo-downloads-cache",
        &["~/.cargo/registry", "~/.cargo/git"],
        &mut restored_paths,
    );
    assert_composite_github_actions_cache_owner(
        &verification_steps,
        "native-archives-cache",
        &["~/Library/Caches/Astronomical/native-dependencies"],
        &mut restored_paths,
    );
    assert_composite_github_actions_cache_owner(
        &verification_steps,
        "native-build-cache",
        &["env.NATIVE_BUILD_ENTRY_DIRECTORY"],
        &mut restored_paths,
    );
    assert_composite_github_actions_cache_owner(
        &verification_steps,
        "sccache-cache",
        &["~/Library/Caches/Astronomical/sccache"],
        &mut restored_paths,
    );
    assert_composite_github_actions_cache_owner(
        &verification_steps,
        "verification-tools-cache",
        &["~/Library/Caches/Astronomical/verification-tools"],
        &mut restored_paths,
    );
    assert_composite_github_actions_cache_owner(
        &swift_steps,
        "swiftpm-cache",
        &["apps/astronomical-menu/.build"],
        &mut restored_paths,
    );
    assert_composite_github_actions_cache_owner(
        &swift_steps,
        "thin-talk-cache",
        &["apps/thin-talk/.build"],
        &mut restored_paths,
    );
    assert_composite_github_actions_cache_owner(
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
        "github_actions_cache owners overlap paths"
    );
}

#[test]
fn native_and_sccache_keys_keep_their_identity_couplings() {
    let workflow = workflow_contract();
    let verification_steps = steps_of(required_job(&workflow, "rust-core"), "rust-core");

    assert_native_build_entry_is_identity_specific(&verification_steps);
    assert_sccache_key_follows_dependencies_not_native_identity(&verification_steps);
    assert_swiftpm_key_is_decoupled_from_rust(&workflow);
    assert_thin_talk_key_is_decoupled_and_owns_its_prune_family(&workflow);
}

#[test]
fn github_actions_cache_prune_runs_last_with_bounded_authority() {
    let workflow = workflow_contract();
    let verification_job = required_job(&workflow, "rust-core");
    let verification_steps = steps_of(verification_job, "rust-core");

    let prune_step = find_step_by_name(
        &verification_steps,
        "Prune surplus github_actions_cache entries",
    );
    assert!(
        required_text(prune_step, "if", "github_actions_cache prunestep").contains("!cancelled()"),
        "github_actions_cache prunemust run even when earlier steps fail"
    );
    assert_eq!(
        required_text(
            required_field(prune_step, "env", "github_actions_cache prunestep"),
            "GH_TOKEN",
            "github_actions_cache pruneenv"
        ),
        "${{ github.token }}",
        "github_actions_cache prunelacks the workflow token"
    );
    assert!(
        run_text_of(prune_step, "prune-caches").contains("prune-caches"),
        "github_actions_cache prunedoes not call the xtask prune"
    );
    assert!(
        required_u64(
            prune_step,
            "timeout-minutes",
            "github_actions_cache prunestep"
        ) <= 2,
        "github_actions_cache pruneexceeded its bounded timeout"
    );
    let prune_step_position = verification_steps
        .iter()
        .position(|step| {
            step.get("name").and_then(Value::as_str)
                == Some("Prune surplus github_actions_cache entries")
        })
        .expect("github_actions_cache prunestep position");
    let final_save_position = verification_steps
        .iter()
        .position(|step| step.get("id").and_then(Value::as_str) == Some("sccache-cache-save"))
        .expect("sccache-cache-save position");
    assert!(
        prune_step_position > final_save_position,
        "github_actions_cache prunemust run after the final github_actions_cache save"
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
        "github_actions_cache pruneneeds actions write permission"
    );
}

#[test]
fn shared_astronomical_cache_action_is_pinned_and_reported() {
    let composite = composite_action_contract();
    let composite_runs = required_field(&composite, "runs", "the shared astronomical-cache action");
    assert_eq!(
        required_text(
            composite_runs,
            "using",
            "the shared astronomical-cache action runs"
        ),
        "composite",
        "shared astronomical-cache action must run as a composite"
    );
    let composite_steps = steps_of(composite_runs, "the shared astronomical-cache action");
    assert_composite_restore_and_save_steps_are_pinned(&composite_steps);
    assert_composite_report_forwards_owner_and_classification(
        &composite_steps,
        EXPECTED_CACHE_CLASSIFICATION,
    );
    assert_composite_surfaces_hits_to_callers(&composite);
}

const SCCACHE_KEY_PREFIX: &str = "astronomical-v2-sccache-";
const TOOL_PINS_MANIFEST: &str = "third-party/tool-pins/Cargo.toml";

/// Every line of `step_text` that names an `sccache` key of the
/// `github_actions_cache` but does not
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

// sccache releases change the format version of its own storage, so an sccache bump
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

// The neural-engine lane's `sccache` must stay its own bounded family:
// a key under the build job's `astronomical-v2-sccache-` prefix would be
// restored by the build job's fallbacks and pruned as a build-job generation;
// a pull-request save would duplicate the default-branch entry; and an
// sccache binary off the tool pin would write a storage format the key does
// not rotate on.
#[test]
fn should_keep_the_neural_engine_sccache_in_its_own_default_branch_family() {
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
        "neural-engine sccache left its own prune family: {restore_key}"
    );
    assert!(
        !restore_key.starts_with(SCCACHE_KEY_PREFIX),
        "neural-engine sccache shares the build job's sccache family"
    );
    assert!(
        restore_key.contains(TOOL_PINS_MANIFEST) && restore_key.contains("Cargo.lock"),
        "neural-engine sccache key does not rotate with the tool pins and dependency graph"
    );
    let save_condition = required_text(
        find_step_by_id(&neural_engine_steps, "neural-engine-sccache-cache-save"),
        "if",
        "neural-engine-sccache-cache save",
    );
    assert!(
        save_condition.contains("github.event.repository.default_branch"),
        "neural-engine sccache saves outside the default branch"
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
