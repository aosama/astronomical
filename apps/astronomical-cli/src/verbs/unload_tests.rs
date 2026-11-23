use ipc_protocol::DaemonModelUnloadStatus;

use super::{UnloadError, unload_outcome_line};

#[test]
fn should_name_the_model_it_unloaded() {
    let outcome_line = unload_outcome_line(
        DaemonModelUnloadStatus::Unloaded,
        Some("Ornith-1.5-9B-MLX-4bit-vision"),
    )
    .expect("an unload succeeds");
    assert_eq!(
        outcome_line,
        "Unloaded Ornith-1.5-9B-MLX-4bit-vision. Its memory is free."
    );
}

#[test]
fn should_succeed_when_nothing_was_loaded() {
    assert_eq!(
        unload_outcome_line(DaemonModelUnloadStatus::NothingLoaded, None)
            .expect("nothing to unload is not a failure"),
        "No model was loaded."
    );
}

#[test]
fn should_count_an_unload_whose_engine_is_restarting_as_done() {
    let outcome_line = unload_outcome_line(DaemonModelUnloadStatus::UnloadedEngineRestarting, None)
        .expect("the memory is free even while the engine restarts");
    assert!(outcome_line.starts_with("Unloaded The model."));
}

#[test]
fn should_fail_while_a_request_holds_the_model() {
    assert!(matches!(
        unload_outcome_line(DaemonModelUnloadStatus::Busy, Some("any")),
        Err(UnloadError::Busy)
    ));
}

#[test]
fn should_fail_when_the_engine_is_not_running() {
    assert!(matches!(
        unload_outcome_line(DaemonModelUnloadStatus::Unavailable, None),
        Err(UnloadError::EngineNotRunning)
    ));
}
