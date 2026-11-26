//! Unit contracts for the pure steps of chat generation admission: mapping
//! the end of a model-swap wait and assembling the active request. Admission
//! itself needs a live worker process and runs under the daemon integration
//! journeys.

use std::sync::Arc;

use ipc_protocol::RequestId;
use tokio::sync::{Semaphore, mpsc};

use super::{
    ModelSelectionOutcome, assemble_active_chat_generation, model_selection_after_swap_wait,
};
use crate::worker::loop_types::{ActiveWorkerRequest, ChatWork};
use crate::worker::process::model_swap::ModelSwapWaitOutcome;
use crate::{GenerationStartError, WorkerControlError};

#[test]
fn should_let_the_request_proceed_once_the_swapped_model_is_loaded() {
    let model_selection_outcome = model_selection_after_swap_wait(Ok(ModelSwapWaitOutcome::Loaded));

    assert!(matches!(
        model_selection_outcome,
        ModelSelectionOutcome::Proceed
    ));
}

#[test]
fn should_reject_the_request_with_the_reason_the_worker_refused_the_load() {
    let model_selection_outcome =
        model_selection_after_swap_wait(Ok(ModelSwapWaitOutcome::Rejected {
            model_load_failure_reason: "the quantized pack is unreadable".to_owned(),
        }));

    assert!(matches!(
        model_selection_outcome,
        ModelSelectionOutcome::Rejected(GenerationStartError::ModelLoadFailed {
            model_load_failure_reason
        }) if model_load_failure_reason == "the quantized pack is unreadable"
    ));
}

#[test]
fn should_treat_a_failed_swap_wait_as_a_failed_swap() {
    let model_selection_outcome =
        model_selection_after_swap_wait(Err(WorkerControlError::ModelLoadTimeout {
            model_load_timeout_millis: 60_000,
        }));

    assert!(matches!(
        model_selection_outcome,
        ModelSelectionOutcome::SwapFailed(WorkerControlError::ModelLoadTimeout {
            model_load_timeout_millis: 60_000
        })
    ));
}

#[test]
fn should_assemble_a_fresh_active_chat_request_for_a_started_generation() {
    let active_generation_permit = Arc::new(Semaphore::new(1))
        .try_acquire_owned()
        .expect("the test semaphore permit should be acquirable");
    let (stream_event_sender, _stream_event_receiver) = mpsc::channel(4);

    let active_request = assemble_active_chat_generation(
        active_generation_permit,
        RequestId::new(41),
        256,
        stream_event_sender,
        ChatWork::Generate,
    );

    let ActiveWorkerRequest::Chat(active_chat_request) = active_request else {
        panic!("a started chat generation should be an active chat request");
    };
    assert_eq!(active_chat_request.request_id, RequestId::new(41));
    assert_eq!(active_chat_request.max_output_tokens, 256);
    assert_eq!(active_chat_request.generated_token_count, 0);
    assert_eq!(active_chat_request.next_sequence_number, 0);
}
