//! Driving one completion to its terminal event and reading its numbers back.
//!
//! The worker's performance log is the authority for the measured rates: it is
//! written by the serving process itself, so a client wall clock can never
//! contaminate the record. Everything here exists to make that one round trip —
//! launch a completion, wait for its terminal event, parse the sample the worker
//! wrote — and to keep the diagnostic output that explains a failure.

use std::fs;
use std::path::Path;
use std::time::Duration;

use tokio::time::sleep;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatImageInput, ChatMessage, ChatToolChoice,
    RequestId,
};
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, WorkerHandle, WorkerHealthStatus,
};
use serde::Deserialize;

/// Values from the single measured completion, after the warmup is discarded.
#[derive(Clone, Debug)]
pub(crate) struct ThroughputMeasurement {
    pub(crate) model_id: String,
    pub(crate) total_input_tokens: u32,
    pub(crate) total_output_tokens: u16,
    pub(crate) cached_tokens: u32,
    pub(crate) prefill_time_seconds: f64,
    pub(crate) decode_time_seconds: f64,
    pub(crate) prefill_tokens_per_second: f64,
    pub(crate) decode_tokens_per_second: f64,
}

/// One line of the worker performance log, re-parsed for the fields the report needs.
#[derive(Debug, Deserialize)]
struct ThroughputSample {
    prompt_token_count: u32,
    cached_token_count: u32,
    generated_token_count: u16,
    prefill_tok_per_second: Option<f64>,
    generation_tok_per_second: Option<f64>,
    prefill_elapsed_millis: u64,
    generation_elapsed_millis: u64,
}

/// Drives one completion to its terminal event, returning when the worker reports
/// completion so its performance record is written.
pub(crate) async fn drive_completion(
    worker_handle: &WorkerHandle,
    request_id: RequestId,
    model_id: &str,
    prompt: &str,
    images: Vec<ChatImageInput>,
    output_tokens: u16,
    temperature_thousandths: u16,
) -> Result<(), String> {
    let command = ChatGenerationCommand {
        request_id,
        model: model_id.to_owned(),
        messages: vec![ChatMessage::User {
            content: prompt.to_owned(),
            images,
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: output_tokens,
            temperature_thousandths: Some(temperature_thousandths),
            top_p_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
        structured_generation: None,
    };
    let mut receiver = match worker_handle.start_chat_generation(command).await {
        Ok(receiver) => receiver,
        Err(error) => {
            return Err(format!(
                "the worker rejected the completion request: {error:?}"
            ));
        }
    };
    while let Some(event) = receiver.recv().await {
        match event {
            ChatGenerationStreamEvent::Completed { .. } => return Ok(()),
            ChatGenerationStreamEvent::Failed { reason } => {
                return Err(format!(
                    "the worker failed the completion request: {reason:?}"
                ));
            }
            ChatGenerationStreamEvent::Error(error_code) => {
                return Err(format!(
                    "the worker stream reported a completion error: {error_code:?}"
                ));
            }
            _ => {}
        }
    }
    Err("the worker stream closed without a terminal completion event".to_owned())
}

/// Prints the tail of every file in the worker's persistent logging directory
/// so the last lines a failed worker wrote stay in the journey output.
///
/// Only a driver's own failure path calls this: the tail can be thousands of
/// lines, so a passing journey never pays for it.
pub(crate) fn dump_worker_logging_directory(
    logging_directory: &Path,
    worker_log_dump_line_limit: usize,
) {
    let Ok(entries) = fs::read_dir(logging_directory) else {
        eprintln!(
            "[performance-throughput] no worker logging directory at {}",
            logging_directory.display()
        );
        return;
    };
    for entry in entries.flatten().filter(|entry| entry.path().is_file()) {
        let log_file_path = entry.path();
        let Ok(contents) = fs::read_to_string(&log_file_path) else {
            continue;
        };
        eprintln!(
            "[performance-throughput] worker log {} (tail):",
            log_file_path.display()
        );
        for log_line in contents.lines().rev().take(worker_log_dump_line_limit) {
            eprintln!("[performance-throughput]   {log_line}");
        }
    }
}

/// Waits for the worker to finish loading and release the model, so the warmup
/// completion never races model loading and the measured run never starts on a
/// cold worker.
pub(crate) async fn wait_until_idle(worker_handle: &WorkerHandle, ready_attempt_limit: u8) {
    for _ in 1..=ready_attempt_limit {
        let snapshot = worker_handle.worker_health_snapshot();
        if snapshot.status == WorkerHealthStatus::Ready && snapshot.ready_model_id.is_none() {
            return;
        }
        sleep(Duration::from_secs(1)).await;
    }
    panic!("the real worker did not become idle before the throughput deadline");
}

/// Reads the worker's performance log and returns the measured completion.
///
/// The parsed sample type never leaves this module: callers want the
/// summarized measurement, and exposing the raw sample would let a journey
/// read a field the driver never meant to publish.
pub(crate) fn read_measured_completion(
    log_path: &Path,
    model_id: &str,
) -> Option<ThroughputMeasurement> {
    summarize(model_id, &read_throughput_samples(log_path))
}

fn read_throughput_samples(log_path: &Path) -> Vec<ThroughputSample> {
    let text = fs::read_to_string(log_path).unwrap_or_default();
    text.lines()
        .filter_map(|line| serde_json::from_str::<ThroughputSample>(line).ok())
        .collect()
}

/// Summarizes the measured completion, which is the last performance sample the
/// worker wrote (the warmup sample precedes it), so JIT warmup never enters the
/// reported values. A budget-exhausted completion never wrote a sample, so its
/// rates are zero and the caller records the partial evidence instead.
fn summarize(model_id: &str, samples: &[ThroughputSample]) -> Option<ThroughputMeasurement> {
    let measured = samples.last()?;
    Some(ThroughputMeasurement {
        model_id: model_id.to_owned(),
        total_input_tokens: measured.prompt_token_count,
        total_output_tokens: measured.generated_token_count,
        cached_tokens: measured.cached_token_count,
        prefill_time_seconds: measured.prefill_elapsed_millis as f64 / 1_000.0,
        decode_time_seconds: measured.generation_elapsed_millis as f64 / 1_000.0,
        prefill_tokens_per_second: measured.prefill_tok_per_second.unwrap_or(0.0),
        decode_tokens_per_second: measured.generation_tok_per_second.unwrap_or(0.0),
    })
}
