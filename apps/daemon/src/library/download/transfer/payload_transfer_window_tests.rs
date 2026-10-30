//! Beside-source coverage for the pure units of the bounded payload
//! transfer window: the terminal stop-outcome mapping. The scheduling loop
//! itself is exercised by the download journey suites.

use super::transfer_window_stop;
use crate::library::download::transfer::DownloadPayloadTransferError;
use crate::library::download::transfer::payload_transfer_window::TransferWindowStop;

#[test]
fn should_fail_the_window_with_the_deferred_transfer_error() {
    let outcome = transfer_window_stop(Some(DownloadPayloadTransferError::InvalidJobState), true);

    assert!(matches!(
        outcome,
        Err(TransferWindowStop::TransferFailed(
            DownloadPayloadTransferError::InvalidJobState
        ))
    ));
}

#[test]
fn should_defer_the_failure_even_when_a_pause_was_requested() {
    // The deferred error wins over the pause: the journey must surface the
    // transfer failure, not a soft pause, after bytes failed to move.
    let outcome = transfer_window_stop(Some(DownloadPayloadTransferError::InvalidJobState), true);

    assert!(matches!(
        outcome,
        Err(TransferWindowStop::TransferFailed(
            DownloadPayloadTransferError::InvalidJobState
        ))
    ));
}

#[test]
fn should_stop_paused_when_drained_with_a_pause_request() {
    let outcome = transfer_window_stop(None, true);

    assert!(matches!(outcome, Err(TransferWindowStop::Paused)));
}

#[test]
fn should_complete_when_drained_without_a_pause_request() {
    let outcome = transfer_window_stop(None, false);

    assert!(outcome.is_ok());
}
