use ipc_protocol::{DaemonModelUnloadStatus, DaemonResponse};

use super::model_unloaded_response;
use crate::ModelUnloadOutcome;

#[test]
fn should_name_the_unloaded_model() {
    assert_eq!(
        model_unloaded_response(ModelUnloadOutcome::Unloaded {
            model_id: "Ornith-1.5-9B-MLX-4bit-vision".to_owned()
        }),
        DaemonResponse::ModelUnloaded {
            status: DaemonModelUnloadStatus::Unloaded,
            model_id: Some("Ornith-1.5-9B-MLX-4bit-vision".to_owned()),
        }
    );
}

#[test]
fn should_report_a_busy_worker_without_a_model_id() {
    assert_eq!(
        model_unloaded_response(ModelUnloadOutcome::Busy),
        DaemonResponse::ModelUnloaded {
            status: DaemonModelUnloadStatus::Busy,
            model_id: None,
        }
    );
}

#[test]
fn should_map_every_outcome_to_its_own_status() {
    let unload_statuses = [
        ModelUnloadOutcome::Unloaded {
            model_id: String::new(),
        },
        ModelUnloadOutcome::NothingLoaded,
        ModelUnloadOutcome::Busy,
        ModelUnloadOutcome::Unavailable,
        ModelUnloadOutcome::UnloadedWithoutReadyWorker {
            model_id: String::new(),
        },
    ]
    .map(
        |unload_outcome| match model_unloaded_response(unload_outcome) {
            DaemonResponse::ModelUnloaded { status, .. } => status,
            unexpected_response => panic!("unexpected frame {unexpected_response:?}"),
        },
    );
    for (status_index, unload_status) in unload_statuses.iter().enumerate() {
        assert_eq!(
            unload_statuses
                .iter()
                .filter(|other| *other == unload_status)
                .count(),
            1,
            "status {status_index} is shared with another outcome"
        );
    }
}
