//! Assertion groups for the required jobs of `ci.yml`, split out of the
//! workflow structural contract so each contract test reads as a short list
//! of the facts it pins and the contract file stays within the source-size
//! limit.

use crate::contract_support::{
    EXPECTED_MACOS_AUTHORITY, EXPECTED_MACOS_CONCURRENCY_GROUP, EXPECTED_RUST_CORE_AUTHORITY,
    EXPECTED_SWIFT_CONCURRENCY_GROUP, EXPECTED_SWIFT_NODE_AUTHORITY, required_bool, required_field,
    required_job, required_text, required_u64,
};
use serde_yaml::Value;

pub(crate) fn assert_macos_jobs_keep_hard_caps_and_fail_closed_authority(workflow: &Value) {
    for (macos_job_id, hard_cap_minutes, expected_authority) in [
        ("rust-core", 15, EXPECTED_RUST_CORE_AUTHORITY),
        ("swift-node", 12, EXPECTED_SWIFT_NODE_AUTHORITY),
        ("neural-engine", 6, EXPECTED_MACOS_AUTHORITY),
    ] {
        let job_context = format!("job {macos_job_id}");
        let macos_job = required_job(workflow, macos_job_id);
        assert_eq!(
            required_u64(macos_job, "timeout-minutes", &job_context),
            hard_cap_minutes,
            "{macos_job_id} exceeded its hard cap"
        );
        assert_eq!(
            required_text(macos_job, "if", &job_context),
            expected_authority,
            "{macos_job_id} authority does not fail closed"
        );
    }
}

pub(crate) fn assert_macos_concurrency_never_cancels_in_flight_runs(workflow: &Value) {
    let verification_job = required_job(workflow, "rust-core");
    let swift_node_job = required_job(workflow, "swift-node");
    let verification_concurrency = required_field(verification_job, "concurrency", "job rust-core");
    assert_eq!(
        required_text(verification_concurrency, "group", "rust-core concurrency"),
        EXPECTED_MACOS_CONCURRENCY_GROUP,
        "macOS concurrency is not event-and-ref scoped"
    );
    assert!(
        !required_bool(
            verification_concurrency,
            "cancel-in-progress",
            "rust-core concurrency"
        ),
        "required macOS verification must not cancel in-flight runs"
    );
    let swift_concurrency = required_field(swift_node_job, "concurrency", "job swift-node");
    assert_eq!(
        required_text(swift_concurrency, "group", "swift-node concurrency"),
        EXPECTED_SWIFT_CONCURRENCY_GROUP,
        "swift-node concurrency is not event-and-ref scoped"
    );
    assert!(
        !required_bool(
            swift_concurrency,
            "cancel-in-progress",
            "swift-node concurrency"
        ),
        "swift-node must not cancel in-flight runs"
    );
}

pub(crate) fn assert_neural_engine_static_and_config_contract_lanes_are_gated(workflow: &Value) {
    let static_job = required_job(workflow, "static");
    let neural_engine_job = required_job(workflow, "neural-engine");
    assert_eq!(
        required_text(neural_engine_job, "name", "job neural-engine"),
        "Neural Engine, xtask and standalone builds",
        "the neural-engine lane name changed"
    );
    assert!(
        required_text(static_job, "if", "job static").contains("needs.detect-changes.result =="),
        "static checks must require successful classification"
    );
    let config_contracts_job = required_job(workflow, "config-contracts");
    assert_eq!(
        required_text(config_contracts_job, "name", "job config-contracts"),
        "Configuration contracts",
        "the config-contracts lane name changed"
    );
    assert_eq!(
        required_u64(
            config_contracts_job,
            "timeout-minutes",
            "job config-contracts"
        ),
        8,
        "config-contracts exceeded its hard cap"
    );
    assert!(
        required_text(config_contracts_job, "if", "job config-contracts")
            .contains("needs.detect-changes.result =="),
        "config contracts must require successful classification"
    );
}
