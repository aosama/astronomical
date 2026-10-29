//! Unit contracts for the pure decision units of the config reload endpoint:
//! the live-ceiling validation predicate and the memory-effective generation
//! derivation.

use crate::config::config_reload::ConfigReloadDecision;
use crate::rest::config_reload::{
    effective_ceiling_violates_worker_limit, memory_effective_generation_for_reload,
};

fn rest_api_restart_required_decision() -> ConfigReloadDecision {
    ConfigReloadDecision::RestApiRestartRequired {
        reloaded_fields: vec![String::from("listener_port")],
        restart_required_fields: vec![String::from("listener_port")],
        discovered_model_count: 2,
    }
}

fn no_worker_restart_decision() -> ConfigReloadDecision {
    ConfigReloadDecision::NoWorkerRestart {
        reloaded_fields: vec![String::from("maximum_mlx_memory_gb")],
        discovered_model_count: 2,
    }
}

#[test]
fn should_reject_a_zero_effective_ceiling() {
    assert!(
        effective_ceiling_violates_worker_limit(0, 18_446_744_073_709_551_615),
        "zero means no limit is set at all and the worker treats it as unset"
    );
}

#[test]
fn should_reject_a_ceiling_above_the_machine_limit() {
    assert!(
        effective_ceiling_violates_worker_limit(37, 36),
        "a ceiling above the machine's reported limit is not allocatable"
    );
}

#[test]
fn should_accept_a_ceiling_equal_to_the_machine_limit() {
    assert!(
        !effective_ceiling_violates_worker_limit(36, 36),
        "the machine limit itself is the boundary and is accepted"
    );
}

#[test]
fn should_accept_a_ceiling_below_the_machine_limit() {
    assert!(
        !effective_ceiling_violates_worker_limit(18, 36),
        "any ceiling up to the machine limit is accepted"
    );
}

#[test]
fn should_derive_a_memory_only_generation_for_a_changed_limit_under_restart_required() {
    let effective_generation = memory_effective_generation_for_reload(
        true,
        &rest_api_restart_required_decision(),
        "gen-current",
        Some(37_398_417_028),
        "gen-candidate",
    );
    assert_ne!(
        effective_generation, "gen-candidate",
        "the restart-required candidate is not wholly effective yet, so the live generation must be a memory-only transition"
    );
    assert_eq!(
        effective_generation,
        crate::config::resolved_configuration_generation::ResolvedConfigurationGeneration::derive_memory_only_transition(
            "gen-current",
            Some(37_398_417_028),
        ),
        "the transition derives from the current generation and the candidate ceiling"
    );
}

#[test]
fn should_keep_the_candidate_generation_when_no_restart_is_required() {
    let effective_generation = memory_effective_generation_for_reload(
        true,
        &no_worker_restart_decision(),
        "gen-current",
        Some(37_398_417_028),
        "gen-candidate",
    );
    assert_eq!(
        effective_generation, "gen-candidate",
        "a no-restart reload takes effect wholly under the candidate generation"
    );
}

#[test]
fn should_keep_the_candidate_generation_when_the_limit_did_not_change() {
    let effective_generation = memory_effective_generation_for_reload(
        false,
        &rest_api_restart_required_decision(),
        "gen-current",
        Some(37_398_417_028),
        "gen-candidate",
    );
    assert_eq!(
        effective_generation, "gen-candidate",
        "no memory change means no memory-only transition, even under restart-required"
    );
}
