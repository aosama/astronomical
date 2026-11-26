//! Assertion groups split out of the `github_actions_cache` ownership
//! contract so each contract test reads as a short list of the facts it pins
//! and the contract file stays within the source-size limit.

use crate::contract_support::{
    find_step_by_id, is_pinned_action_reference, required_field, required_job, required_text,
    run_text_of, steps_of,
};
use serde_yaml::Value;

pub(crate) fn assert_native_build_entry_is_identity_specific(verification_steps: &[&Value]) {
    let native_configuration = required_field(
        find_step_by_id(verification_steps, "native-build-cache"),
        "with",
        "native-build-cache restore",
    );
    if let Some(restore_keys) = native_configuration.get("restore-keys") {
        assert!(
            restore_keys
                .as_str()
                .unwrap_or_default()
                .contains("NATIVE_BUILD_IDENTITY"),
            "the native-build entry of the github_actions_cache must not cross identities"
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
}

pub(crate) fn assert_sccache_key_follows_dependencies_not_native_identity(
    verification_steps: &[&Value],
) {
    let sccache_configuration = required_field(
        find_step_by_id(verification_steps, "sccache-cache"),
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
        "sccache key couples to the native identity; the native build product entry of the github_actions_cache owns that identity"
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
}

pub(crate) fn assert_swiftpm_key_is_decoupled_from_rust(workflow: &Value) {
    let swift_configuration = required_field(
        find_step_by_id(
            &steps_of(required_job(workflow, "swift-node"), "swift-node"),
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
        "the SwiftPM entry of the github_actions_cache omits toolchain compatibility"
    );
}

pub(crate) fn assert_thin_talk_key_is_decoupled_and_owns_its_prune_family(workflow: &Value) {
    let thin_talk_configuration = required_field(
        find_step_by_id(
            &steps_of(required_job(workflow, "swift-node"), "swift-node"),
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
        "the Thin Talk entry of the github_actions_cache omits toolchain compatibility"
    );
    // The thin-talk key must live in its OWN prune family: the prune derives
    // each entry's family from the key prefix before `-macOS` and keeps one
    // entry per family, so a shared `astronomical-v2-swiftpm-` prefix made
    // the prune evict menu and thin-talk in alternation (2026-10-31 hunt).
    assert!(
        thin_talk_key.starts_with("astronomical-v2-swiftpm-thin-talk-"),
        "the thin-talk entry of the github_actions_cache shares the swiftpm prune family and will be evicted in alternation with the menu entry"
    );
}

pub(crate) fn assert_composite_restore_and_save_steps_are_pinned(composite_steps: &[&Value]) {
    let composite_restore = find_step_by_id(composite_steps, "restore");
    assert!(
        is_pinned_action_reference(
            required_text(
                composite_restore,
                "uses",
                "shared astronomical-cache action restore"
            ),
            "actions/cache/restore",
        ),
        "shared astronomical-cache action restore is not a pinned actions/cache/restore"
    );
    let composite_save = find_step_by_id(composite_steps, "save");
    assert!(
        is_pinned_action_reference(
            required_text(
                composite_save,
                "uses",
                "shared astronomical-cache action save"
            ),
            "actions/cache/save",
        ),
        "shared astronomical-cache action save is not a pinned actions/cache/save"
    );
}

pub(crate) fn assert_composite_report_forwards_owner_and_classification(
    composite_steps: &[&Value],
    expected_classification: &str,
) {
    let composite_report = find_step_by_id(composite_steps, "report");
    let report_environment = required_field(
        composite_report,
        "env",
        "shared astronomical-cache action report",
    );
    assert!(
        required_text(
            report_environment,
            "CACHE_OWNER",
            "shared astronomical-cache action report env"
        )
        .contains("inputs.owner"),
        "shared astronomical-cache action report does not forward the owner"
    );
    assert_eq!(
        required_text(
            report_environment,
            "CACHE_CLASSIFICATION",
            "shared astronomical-cache action report env"
        ),
        expected_classification,
        "the github_actions_cache classification rules changed; update this pin only on purpose"
    );
    assert!(
        run_text_of(composite_report, "report").contains(">> \"$GITHUB_STEP_SUMMARY\""),
        "shared astronomical-cache action report must write its classification to the job summary"
    );
}

pub(crate) fn assert_composite_surfaces_hits_to_callers(composite: &Value) {
    assert!(
        required_text(
            required_field(
                required_field(composite, "outputs", "the shared astronomical-cache action"),
                "cache-hit",
                "shared astronomical-cache action outputs"
            ),
            "value",
            "shared astronomical-cache action cache-hit output"
        )
        .contains("steps.restore.outputs.cache-hit"),
        "shared astronomical-cache action must surface github_actions_cache hits to callers"
    );
}
