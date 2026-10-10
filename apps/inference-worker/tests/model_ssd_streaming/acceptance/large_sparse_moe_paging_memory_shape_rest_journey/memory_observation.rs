//! Observation layer for the paging memory-shape journey.
//!
//! The journey's subject is the SHAPE of memory over a request: what the
//! device held at each moment, how much of it was expert payload, which
//! residency mode was active, and how fast each phase progressed. Collecting
//! that is a separate job from deciding what it means, so it lives here with
//! its own types rather than inside the journey that asserts on it.

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::time::{Instant as TokioInstant, sleep};

use crate::support::openai_client::ChatCompletionStream;
use crate::support::serving_rest::get_json_endpoint;

use super::{MEMORY_SAMPLE_INTERVAL, STATUS_LOG_INTERVAL};

pub(super) fn progress_tokens_per_second(progress: &PhaseProgress) -> f64 {
    let elapsed_seconds = progress.elapsed_seconds();
    if elapsed_seconds <= 0.0 {
        0.0
    } else {
        progress.processed_tokens as f64 / elapsed_seconds
    }
}

pub(super) fn print_memory_shape_timeline(samples: &[MemorySample]) {
    eprintln!(
        "[paging-memory-shape] status=timeline samples={} offset_s activity phase processed total active_gb cache_gb peak_gb experts_gb mode",
        samples.len()
    );
    for sample in samples {
        eprintln!(
            "[paging-memory-shape] timeline offset={:.1} activity={} phase={} processed={} total={} active_gb={:.2} cache_gb={:.2} peak_gb={:.2} experts_gb={:.2} mode={}",
            sample.offset_seconds,
            sample.activity,
            sample.phase,
            sample.processed_tokens,
            sample.total_tokens,
            sample.active_memory_bytes as f64 / 1_000_000_000.0,
            sample.allocator_cache_memory_bytes as f64 / 1_000_000_000.0,
            sample.peak_memory_bytes as f64 / 1_000_000_000.0,
            sample.expert_payload_bytes as f64 / 1_000_000_000.0,
            sample.expert_memory_mode,
        );
    }
}

#[derive(Debug, Clone)]
pub(super) struct PhaseProgress {
    pub(super) processed_tokens: u64,
    pub(super) elapsed_millis: u64,
}

impl PhaseProgress {
    pub(super) fn elapsed_seconds(&self) -> f64 {
        self.elapsed_millis as f64 / 1_000.0
    }
}

#[derive(Debug, Clone)]
pub(super) struct MemorySample {
    pub(super) offset_seconds: f64,
    pub(super) activity: String,
    pub(super) phase: String,
    pub(super) processed_tokens: u64,
    pub(super) total_tokens: u64,
    pub(super) elapsed_millis: u64,
    pub(super) active_memory_bytes: u64,
    pub(super) allocator_cache_memory_bytes: u64,
    pub(super) peak_memory_bytes: u64,
    pub(super) expert_payload_bytes: u64,
    pub(super) expert_memory_mode: String,
}

impl From<&MemorySample> for Value {
    fn from(sample: &MemorySample) -> Value {
        json!({
            "offset_seconds": sample.offset_seconds,
            "activity": sample.activity,
            "phase": sample.phase,
            "processed_tokens": sample.processed_tokens,
            "total_tokens": sample.total_tokens,
            "elapsed_millis": sample.elapsed_millis,
            "active_memory_bytes": sample.active_memory_bytes,
            "allocator_cache_memory_bytes": sample.allocator_cache_memory_bytes,
            "peak_memory_bytes": sample.peak_memory_bytes,
            "expert_payload_bytes": sample.expert_payload_bytes,
            "expert_memory_mode": sample.expert_memory_mode,
        })
    }
}

pub(super) struct MemoryShapeEvidence {
    pub(super) samples: Vec<MemorySample>,
    pub(super) last_prompt_processing_progress: Option<PhaseProgress>,
    pub(super) last_generation_progress: Option<PhaseProgress>,
    pub(super) final_status: Value,
}

pub(super) async fn observe_memory_shape(server_address: SocketAddr) -> MemoryShapeEvidence {
    let observation_started_at = Instant::now();
    let mut samples = Vec::new();
    let mut last_prompt_processing_progress = None;
    let mut last_generation_progress = None;
    let mut observed_generation = false;
    let mut last_status_log_at = Instant::now() - STATUS_LOG_INTERVAL;
    loop {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        let activity = status_document["activity"]
            .as_str()
            .unwrap_or("unknown")
            .to_owned();
        let progress = &status_document["progress"];
        let phase = progress["phase"].as_str().unwrap_or("idle").to_owned();
        let processed_tokens = progress["processed_tokens"].as_u64().unwrap_or(0);
        let total_tokens = progress["total_tokens"].as_u64().unwrap_or(0);
        let elapsed_millis = progress["elapsed_ms"].as_u64().unwrap_or(0);
        // A single-chunk prefill publishes processed_tokens only at its end,
        // so the phase's live elapsed is tracked from every prompt sample and
        // the rate divides the authoritative usage total by that elapsed.
        if activity == "prompt_processing" {
            last_prompt_processing_progress = Some(PhaseProgress {
                processed_tokens,
                elapsed_millis,
            });
        }
        if activity == "generating" {
            observed_generation = true;
            if processed_tokens > 0 {
                last_generation_progress = Some(PhaseProgress {
                    processed_tokens,
                    elapsed_millis,
                });
            }
        }
        let snapshot = &status_document["mlx_memory_snapshot"];
        let sample = MemorySample {
            offset_seconds: observation_started_at.elapsed().as_secs_f64(),
            activity: activity.clone(),
            phase: phase.clone(),
            processed_tokens,
            total_tokens,
            elapsed_millis,
            active_memory_bytes: snapshot["active_memory_bytes"].as_u64().unwrap_or(0),
            allocator_cache_memory_bytes: snapshot["allocator_cache_memory_bytes"]
                .as_u64()
                .unwrap_or(0),
            peak_memory_bytes: snapshot["peak_memory_bytes"].as_u64().unwrap_or(0),
            expert_payload_bytes: snapshot["expert_payload_bytes"].as_u64().unwrap_or(0),
            expert_memory_mode: status_document["expert_memory_mode"]
                .as_str()
                .unwrap_or("unpublished")
                .to_owned(),
        };
        let snapshot_source = snapshot["source"].as_str();
        let request_finished = observed_generation
            && activity == "idle"
            && matches!(snapshot_source, Some("finalized" | "idle_poll"));
        if request_finished || last_status_log_at.elapsed() >= STATUS_LOG_INTERVAL {
            log_sample(&sample, request_finished);
            last_status_log_at = Instant::now();
        }
        if request_finished {
            samples.push(sample);
            return MemoryShapeEvidence {
                samples,
                last_prompt_processing_progress,
                last_generation_progress,
                final_status: status_document,
            };
        }
        samples.push(sample);
        sleep(MEMORY_SAMPLE_INTERVAL).await;
    }
}

pub(super) fn log_sample(sample: &MemorySample, request_finished: bool) {
    eprintln!(
        "[paging-memory-shape] status=progress{} phase={} processed={}/{} elapsed_seconds={:.1} observed_tokens_per_second={:.2} active_gb={:.2} experts_gb={:.2} mode={}",
        if request_finished { " finalized" } else { "" },
        sample.phase,
        sample.processed_tokens,
        sample.total_tokens,
        sample.offset_seconds,
        phase_tokens_per_second(sample),
        sample.active_memory_bytes as f64 / 1_000_000_000.0,
        sample.expert_payload_bytes as f64 / 1_000_000_000.0,
        sample.expert_memory_mode,
    );
}

pub(super) fn phase_tokens_per_second(sample: &MemorySample) -> f64 {
    sample.processed_tokens as f64 * 1_000.0 / sample.elapsed_millis.max(1) as f64
}

pub(super) struct StreamMeasurement {
    pub(super) model_text: String,
    pub(super) finish_reason: Option<String>,
    pub(super) usage: Option<Value>,
    pub(super) first_token_elapsed: Duration,
    pub(super) last_token_elapsed: Duration,
}

pub(super) async fn consume_stream_with_timing(
    mut streamed_completion: ChatCompletionStream,
    request_started_at: TokioInstant,
) -> StreamMeasurement {
    let mut streamed_model_text = String::new();
    let mut finish_reason = None;
    let mut usage = None;
    let mut first_token_elapsed = None;
    let mut last_token_elapsed = request_started_at.elapsed();
    while let Some(stream_item) = streamed_completion.next().await {
        let stream_chunk = stream_item.expect("the public REST stream should remain healthy");
        if let Some(chunk_usage) = stream_chunk.get("usage")
            && chunk_usage.is_object()
        {
            usage = Some(chunk_usage.clone());
        }
        for choice in stream_chunk["choices"].as_array().into_iter().flatten() {
            let delta = &choice["delta"];
            let content_fragment = delta["content"].as_str().unwrap_or_default();
            let reasoning_fragment = delta["reasoning_content"].as_str().unwrap_or_default();
            if !content_fragment.is_empty() || !reasoning_fragment.is_empty() {
                if first_token_elapsed.is_none() {
                    first_token_elapsed = Some(request_started_at.elapsed());
                }
                last_token_elapsed = request_started_at.elapsed();
                streamed_model_text.push_str(content_fragment);
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                finish_reason = Some(reason.to_owned());
            }
        }
    }
    StreamMeasurement {
        model_text: streamed_model_text.trim().to_owned(),
        finish_reason,
        usage,
        first_token_elapsed: first_token_elapsed.unwrap_or_else(|| request_started_at.elapsed()),
        last_token_elapsed,
    }
}
