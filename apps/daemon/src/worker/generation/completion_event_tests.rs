//! Unit contracts for completion-event validation, the stable completion
//! reason labels, and the derived completion timings.

use std::sync::Arc;
use std::time::Duration;

use astronomical_ipc_protocol::{ChatGenerationCompletionReason, RequestId};
use tokio::sync::{Semaphore, mpsc};

use crate::worker::generation::completion_event::{
    CompletionMeasurements, completion_reason_label, measure_completion_timings,
    validate_chat_completion_event,
};
use crate::worker::loop_types::ActiveGeneration;

fn completion_measurements(
    request_id: RequestId,
    generated_token_count: u16,
    reason: ChatGenerationCompletionReason,
) -> CompletionMeasurements {
    CompletionMeasurements {
        request_id,
        prompt_token_count: 100,
        generated_token_count,
        reasoning_token_count: 0,
        cached_token_count: 0,
        reason,
    }
}

fn active_chat_request(
    request_id: RequestId,
    generated_token_count: u16,
    latest_generation_progress_token_count: u16,
    max_output_tokens: u16,
    next_tool_call_index: u16,
) -> ActiveGeneration {
    let active_generation_permit = Arc::new(Semaphore::new(1))
        .try_acquire_owned()
        .expect("the test semaphore permit should be acquirable");
    let (stream_event_sender, _stream_event_receiver) = mpsc::channel(16);
    ActiveGeneration {
        _active_generation_permit: active_generation_permit,
        generated_token_count,
        generation_started_at: None,
        generation_preparation_started_at: None,
        generation_preparation_elapsed_millis: None,
        first_decode_forward_elapsed_millis: None,
        time_to_first_output_millis: None,
        final_resident_expert_count: None,
        final_resident_expert_payload_bytes: None,
        latest_generation_progress_token_count,
        max_output_tokens,
        next_sequence_number: 0,
        next_tool_call_index,
        completed_tool_calls: Vec::new(),
        request_started_at: tokio::time::Instant::now(),
        prefill_elapsed_millis: 0,
        maximum_mlx_peak_memory_bytes: None,
        last_mlx_active_memory_bytes: None,
        request_id,
        stream_event_sender,
    }
}

#[test]
fn should_accept_a_correlated_end_of_sequence_completion_within_the_output_budget() {
    let completion = completion_measurements(
        RequestId::new(1),
        5,
        ChatGenerationCompletionReason::EndOfSequence,
    );
    let active_request = active_chat_request(RequestId::new(1), 4, 5, 200, 0);

    validate_chat_completion_event(&completion, &active_request)
        .expect("a correlated completion within the output budget should be accepted");
}

#[test]
fn should_reject_a_completion_for_a_different_request() {
    let completion = completion_measurements(
        RequestId::new(2),
        5,
        ChatGenerationCompletionReason::EndOfSequence,
    );
    let active_request = active_chat_request(RequestId::new(1), 4, 5, 200, 0);

    let validation_error = validate_chat_completion_event(&completion, &active_request)
        .expect_err("a mismatched request id should be rejected");
    assert!(matches!(
        validation_error,
        crate::WorkerControlError::WorkerProtocolViolation {
            description: "completion correlation or count mismatch",
        }
    ));
}

#[test]
fn should_reject_a_generated_token_count_regression() {
    let completion = completion_measurements(
        RequestId::new(1),
        4,
        ChatGenerationCompletionReason::EndOfSequence,
    );
    let active_request = active_chat_request(RequestId::new(1), 4, 5, 200, 0);

    let validation_error = validate_chat_completion_event(&completion, &active_request)
        .expect_err("a generated count below the latest known count should be rejected");
    assert!(matches!(
        validation_error,
        crate::WorkerControlError::WorkerProtocolViolation {
            description: "completion correlation or count mismatch",
        }
    ));
}

#[test]
fn should_reject_a_completion_over_the_output_budget() {
    let completion = completion_measurements(
        RequestId::new(1),
        201,
        ChatGenerationCompletionReason::EndOfSequence,
    );
    let active_request = active_chat_request(RequestId::new(1), 4, 5, 200, 0);

    let validation_error = validate_chat_completion_event(&completion, &active_request)
        .expect_err("a completion over the output budget should be rejected");
    assert!(matches!(
        validation_error,
        crate::WorkerControlError::WorkerProtocolViolation {
            description: "completion correlation or count mismatch",
        }
    ));
}

#[test]
fn should_reject_a_cancelled_completion_reason() {
    let completion = completion_measurements(
        RequestId::new(1),
        5,
        ChatGenerationCompletionReason::Cancelled,
    );
    let active_request = active_chat_request(RequestId::new(1), 4, 5, 200, 1);

    let validation_error = validate_chat_completion_event(&completion, &active_request)
        .expect_err("a cancelled completion reason should never be accepted from the worker");
    assert!(matches!(
        validation_error,
        crate::WorkerControlError::WorkerProtocolViolation {
            description: "completion correlation or count mismatch",
        }
    ));
}

#[test]
fn should_reject_a_tool_calls_completion_without_an_emitted_tool_call() {
    let completion = completion_measurements(
        RequestId::new(1),
        5,
        ChatGenerationCompletionReason::ToolCalls,
    );
    let active_request = active_chat_request(RequestId::new(1), 4, 5, 200, 0);

    let validation_error = validate_chat_completion_event(&completion, &active_request)
        .expect_err("a tool-calls completion with no emitted tool call should be rejected");
    assert!(matches!(
        validation_error,
        crate::WorkerControlError::WorkerProtocolViolation {
            description: "completion correlation or count mismatch",
        }
    ));
}

#[test]
fn should_reject_a_maximum_output_tokens_completion_that_undershoots_the_budget() {
    let completion = completion_measurements(
        RequestId::new(1),
        199,
        ChatGenerationCompletionReason::MaximumOutputTokens,
    );
    let active_request = active_chat_request(RequestId::new(1), 4, 5, 200, 0);

    let validation_error = validate_chat_completion_event(&completion, &active_request)
        .expect_err("a maximum-output-tokens completion must land exactly on the budget");
    assert!(matches!(
        validation_error,
        crate::WorkerControlError::WorkerProtocolViolation {
            description: "completion correlation or count mismatch",
        }
    ));
}

#[test]
fn should_accept_an_end_of_sequence_completion_at_exactly_the_output_budget() {
    let completion = completion_measurements(
        RequestId::new(1),
        200,
        ChatGenerationCompletionReason::EndOfSequence,
    );
    let active_request = active_chat_request(RequestId::new(1), 199, 200, 200, 0);

    validate_chat_completion_event(&completion, &active_request)
        .expect("end-of-sequence at exactly the output budget is the inclusive boundary");
}

#[test]
fn should_accept_a_maximum_output_tokens_completion_at_exactly_the_output_budget() {
    let completion = completion_measurements(
        RequestId::new(1),
        200,
        ChatGenerationCompletionReason::MaximumOutputTokens,
    );
    let active_request = active_chat_request(RequestId::new(1), 199, 200, 200, 0);

    validate_chat_completion_event(&completion, &active_request)
        .expect("the budget boundary itself is the valid maximum-output-tokens completion");
}

#[test]
fn should_accept_a_generated_count_equal_to_the_latest_known_count() {
    let completion = completion_measurements(
        RequestId::new(1),
        7,
        ChatGenerationCompletionReason::EndOfSequence,
    );
    let active_request = active_chat_request(RequestId::new(1), 7, 7, 200, 0);

    validate_chat_completion_event(&completion, &active_request)
        .expect("equality with the latest known count is not a regression");
}

#[test]
fn should_label_every_completion_reason_with_its_stable_snake_case_name() {
    assert_eq!(
        completion_reason_label(ChatGenerationCompletionReason::EndOfSequence),
        "end_of_sequence"
    );
    assert_eq!(
        completion_reason_label(ChatGenerationCompletionReason::MaximumOutputTokens),
        "maximum_output_tokens"
    );
    assert_eq!(
        completion_reason_label(ChatGenerationCompletionReason::ToolCalls),
        "tool_calls"
    );
    assert_eq!(
        completion_reason_label(ChatGenerationCompletionReason::Cancelled),
        "cancelled"
    );
}

#[test]
fn should_produce_distinct_non_empty_labels_for_every_completion_reason() {
    let completion_labels = [
        completion_reason_label(ChatGenerationCompletionReason::EndOfSequence),
        completion_reason_label(ChatGenerationCompletionReason::MaximumOutputTokens),
        completion_reason_label(ChatGenerationCompletionReason::ToolCalls),
        completion_reason_label(ChatGenerationCompletionReason::Cancelled),
    ];
    for completion_label in completion_labels {
        assert!(
            !completion_label.is_empty(),
            "every completion reason must have a label"
        );
    }
    for (position, completion_label) in completion_labels.iter().enumerate() {
        for other_completion_label in completion_labels.iter().skip(position + 1) {
            assert_ne!(
                completion_label, other_completion_label,
                "completion reason labels must stay distinct so log readers can tell them apart"
            );
        }
    }
}

fn timed_out_chat_request(
    prompt_token_count: u32,
    cached_token_count: u32,
    generated_token_count: u16,
    prefill_elapsed_millis: u64,
    request_started_ago: Duration,
    generation_started_ago: Option<Duration>,
) -> (CompletionMeasurements, ActiveGeneration) {
    let (stream_event_sender, _stream_event_receiver) = mpsc::channel(16);
    let active_generation_permit = Arc::new(Semaphore::new(1))
        .try_acquire_owned()
        .expect("the test semaphore permit should be acquirable");
    let measurements = CompletionMeasurements {
        request_id: RequestId::new(1),
        prompt_token_count,
        generated_token_count,
        reasoning_token_count: 0,
        cached_token_count,
        reason: ChatGenerationCompletionReason::EndOfSequence,
    };
    let completed_request = ActiveGeneration {
        _active_generation_permit: active_generation_permit,
        generated_token_count,
        generation_started_at: generation_started_ago
            .map(|generation_ago| tokio::time::Instant::now() - generation_ago),
        generation_preparation_started_at: None,
        generation_preparation_elapsed_millis: None,
        first_decode_forward_elapsed_millis: None,
        time_to_first_output_millis: None,
        final_resident_expert_count: None,
        final_resident_expert_payload_bytes: None,
        latest_generation_progress_token_count: generated_token_count,
        max_output_tokens: 200,
        next_sequence_number: 0,
        next_tool_call_index: 0,
        completed_tool_calls: Vec::new(),
        request_started_at: tokio::time::Instant::now() - request_started_ago,
        prefill_elapsed_millis,
        maximum_mlx_peak_memory_bytes: None,
        last_mlx_active_memory_bytes: None,
        request_id: RequestId::new(1),
        stream_event_sender,
    };
    (measurements, completed_request)
}

#[test]
fn should_measure_positive_timings_for_a_generation_that_started() {
    let (measurements, completed_request) = timed_out_chat_request(
        100,
        0,
        50,
        250,
        Duration::from_secs(2),
        Some(Duration::from_secs(1)),
    );

    let completion_timings = measure_completion_timings(&measurements, &completed_request);

    assert!(
        completion_timings.total_elapsed_millis >= 1_000,
        "a request started two seconds ago must measure at least one second of total latency"
    );
    assert!(
        completion_timings.generation_elapsed_millis >= 500,
        "a generation started one second ago must measure at least half a second of decode time"
    );
    // Prefill elapsed is carried through unchanged, which pins the prefill
    // throughput to the deterministic 100 uncached tokens over 250 ms.
    assert_eq!(completion_timings.prefill_elapsed_millis, 250);
    assert_eq!(completion_timings.prefill_tok_per_second, Some(400.0));
    assert!(completion_timings.generation_tok_per_second.is_some());
}

#[test]
fn should_not_report_generation_throughput_when_no_tokens_were_generated() {
    let (measurements, completed_request) = timed_out_chat_request(
        100,
        0,
        0,
        250,
        Duration::from_secs(2),
        Some(Duration::from_secs(1)),
    );

    let completion_timings = measure_completion_timings(&measurements, &completed_request);

    assert_eq!(completion_timings.generation_tok_per_second, None);
}

#[test]
fn should_report_zero_generation_elapsed_and_no_throughput_when_generation_never_started() {
    // A fully cached prompt (prompt == cached) exercises the other
    // zero-throughput boundary at the same time.
    let (measurements, completed_request) =
        timed_out_chat_request(100, 100, 50, 0, Duration::from_millis(5), None);

    let completion_timings = measure_completion_timings(&measurements, &completed_request);

    assert_eq!(completion_timings.generation_elapsed_millis, 0);
    assert_eq!(completion_timings.generation_tok_per_second, None);
    assert_eq!(completion_timings.prefill_elapsed_millis, 0);
    assert_eq!(completion_timings.prefill_tok_per_second, None);
    assert!(completion_timings.total_elapsed_millis < 1_000);
}
