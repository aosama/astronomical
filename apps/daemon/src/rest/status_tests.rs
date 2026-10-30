//! Beside-source coverage for the pure status-label units. The JSON
//! attachment units are thin field mappings exercised by the REST
//! integration journeys.

use super::{expert_memory_mode_label, state_directory_label};
use astronomical_config::AstronomicalRuntimeInstance;
use astronomical_ipc_protocol::ExpertMemoryMode;

#[test]
fn should_label_the_standard_stable_state_directory() {
    assert_eq!(
        state_directory_label(true, AstronomicalRuntimeInstance::Stable),
        "~/.astronomical"
    );
}

#[test]
fn should_label_the_standard_development_state_directory() {
    assert_eq!(
        state_directory_label(true, AstronomicalRuntimeInstance::Development),
        "~/.astronomical-dev"
    );
}

#[test]
fn should_label_a_custom_directory_regardless_of_channel() {
    assert_eq!(
        state_directory_label(false, AstronomicalRuntimeInstance::Stable),
        "custom"
    );
    assert_eq!(
        state_directory_label(false, AstronomicalRuntimeInstance::Development),
        "custom"
    );
}

#[test]
fn should_label_every_expert_memory_mode() {
    assert_eq!(
        expert_memory_mode_label(ExpertMemoryMode::Resident),
        "resident"
    );
    assert_eq!(expert_memory_mode_label(ExpertMemoryMode::Hybrid), "hybrid");
    assert_eq!(expert_memory_mode_label(ExpertMemoryMode::Paged), "paged");
}
