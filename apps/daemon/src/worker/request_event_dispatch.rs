//! Routes the events of image, speech, decisions and embeddings requests to
//! the module that owns each kind, so the main event dispatch keeps one arm
//! per family of request.

use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::WorkerEvent;

use crate::worker::loop_types::ActiveWorkerRequest;
use crate::{GenerationPerformanceLog, WorkerControlError, WorkerHealthSnapshot};

/// Routes the events of a request that runs a step at a time: an image or
/// speech. Each kind checks its own order of progress then completion or
/// failure, then finalization.
pub(super) fn dispatch_stepped_event(
    worker_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_request: &mut Option<ActiveWorkerRequest>,
    performance_log: &mut GenerationPerformanceLog,
) -> Result<(), WorkerControlError> {
    if matches!(
        worker_event,
        WorkerEvent::SpeechSegmentSpoken { .. }
            | WorkerEvent::SpeechCompleted { .. }
            | WorkerEvent::SpeechFailed { .. }
            | WorkerEvent::SpeechFinalized { .. }
    ) {
        return crate::worker::speech_event::handle_worker_speech_event(
            worker_event,
            health_snapshot,
            active_request,
        );
    }
    crate::worker::image_event::handle_worker_image_event(
        worker_event,
        health_snapshot,
        active_request,
        performance_log,
    )
}

/// Routes the events of a request that answers in one step: decisions and
/// embeddings. Both follow the same completed-or-failed-then-finalized
/// order, so one arm of the dispatch match hands them here.
pub(super) fn dispatch_single_shot_worker_event(
    worker_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_request: &mut Option<ActiveWorkerRequest>,
) -> Result<(), WorkerControlError> {
    if matches!(
        worker_event,
        WorkerEvent::DecisionsCompleted { .. }
            | WorkerEvent::DecisionsFailed { .. }
            | WorkerEvent::DecisionsFinalized { .. }
    ) {
        return crate::worker::decisions_event::handle_worker_decisions_event(
            worker_event,
            health_snapshot,
            active_request,
        );
    }
    crate::worker::embeddings_event::handle_worker_embeddings_event(
        worker_event,
        health_snapshot,
        active_request,
    )
}
