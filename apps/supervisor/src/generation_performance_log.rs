use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::SystemTime;

use serde::Serialize;

/// One row in the generation performance log.
///
/// Each completed generation request appends one JSON line to
/// `~/.astronomical/logs/performance.jsonl`. Fields are chosen to
/// answer "how well is the model performing?" at a glance:
///
/// - Throughput: `prefill_tok_per_second` and `generation_tok_per_second`
/// - Latency: `total_elapsed_millis`, `prefill_elapsed_millis`, `generation_elapsed_millis`
/// - Scale: `prompt_token_count`, `cached_token_count`, `generated_token_count`
/// - Resources: `mlx_peak_memory_bytes`, `mlx_active_memory_bytes`
/// - Identity: `request_id`, `model_id`, `completion_reason`
#[derive(Clone, Debug, Serialize)]
pub struct GenerationPerformanceRecord {
    /// Unix epoch milliseconds when the record was written (request completion time).
    ///
    /// This is a single `u64` rather than an ISO 8601 string because:
    /// - No external time library dependency
    /// - Trivially sortable and delta-computable for trend analysis
    /// - Unambiguous and timezone-free
    /// - Easily converted to any human-readable format by consumers: `date -r $((ts/1000))`
    pub timestamp_millis: u64,
    /// Supervisor-local monotonic request identifier.
    pub request_id: u64,
    /// The model that produced this generation.
    pub model_id: String,
    /// Total prompt tokens (including cached tokens).
    pub prompt_token_count: u32,
    /// Prompt tokens restored from the persistent SSD cache.
    pub cached_token_count: u32,
    /// Output tokens the model produced.
    pub generated_token_count: u16,
    /// Why the generation stopped: `end_of_sequence`, `tool_calls`, or `maximum_output_tokens`.
    pub completion_reason: String,
    /// Accumulated prompt-processing time across all prefill chunks, in milliseconds.
    pub prefill_elapsed_millis: u64,
    /// Wall-clock decode time from the first output token to completion, in milliseconds.
    pub generation_elapsed_millis: u64,
    /// Wall-clock time from request arrival to completion, in milliseconds.
    pub total_elapsed_millis: u64,
    /// Prefill throughput: `(prompt_token_count - cached_token_count) / (prefill_elapsed_millis / 1000)`.
    /// `None` when the entire prompt was cached (0 ms prefill).
    pub prefill_tok_per_second: Option<f64>,
    /// Decode throughput: `generated_token_count / (generation_elapsed_millis / 1000)`.
    /// `None` when generation took 0 ms.
    pub generation_tok_per_second: Option<f64>,
    /// Peak MLX GPU memory observed during prefill, in bytes.
    pub mlx_peak_memory_bytes: Option<u64>,
    /// Active MLX GPU memory at last prefill progress event, in bytes.
    pub mlx_active_memory_bytes: Option<u64>,
}

impl GenerationPerformanceRecord {
    /// Computes the throughput fields from raw counters and elapsed times.
    ///
    /// - `prefill_tok_per_second` is `None` when `prefill_elapsed_millis == 0` (fully cached).
    /// - `generation_tok_per_second` is `None` when `generation_elapsed_millis == 0`.
    pub fn compute_throughput(
        prompt_token_count: u32,
        cached_token_count: u32,
        generated_token_count: u16,
        prefill_elapsed_millis: u64,
        generation_elapsed_millis: u64,
    ) -> (Option<f64>, Option<f64>) {
        let uncached_prompt_tokens = prompt_token_count.saturating_sub(cached_token_count);
        let prefill_tok_per_second = if prefill_elapsed_millis > 0 && uncached_prompt_tokens > 0 {
            let prefill_seconds = prefill_elapsed_millis as f64 / 1000.0;
            Some(uncached_prompt_tokens as f64 / prefill_seconds)
        } else {
            None
        };
        let generation_tok_per_second =
            if generation_elapsed_millis > 0 && generated_token_count > 0 {
                let generation_seconds = generation_elapsed_millis as f64 / 1000.0;
                Some(generated_token_count as f64 / generation_seconds)
            } else {
                None
            };
        (prefill_tok_per_second, generation_tok_per_second)
    }
}

/// Append-only JSONL writer for generation performance records.
///
/// Opens `performance.jsonl` in the configured log directory at creation
/// and appends one line per completed generation. Uses a synchronous
/// `BufWriter` because the volume is low (one line per request, typically
/// one line every 10–30 seconds) and the writer must never block inference
/// on a slow disk; individual lines end with `\n` so partial writes are
/// bounded. A lossy in-flight buffer is not needed here.
pub struct GenerationPerformanceLog {
    writer: BufWriter<File>,
}

impl GenerationPerformanceLog {
    /// Opens (or creates) the performance log file in the given directory.
    ///
    /// The file is named `performance.jsonl` and opened in append mode.
    /// Creates the file if it does not exist.
    pub fn open(log_directory: &Path) -> std::io::Result<Self> {
        let performance_log_path = log_directory.join("performance.jsonl");
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&performance_log_path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    /// Appends one performance record as a JSON line to the log file.
    ///
    /// Flushes the buffer after each write so records are durable
    /// immediately. This is correct for the low-volume performance log
    /// (one write per request) and avoids data loss on an unclean shutdown.
    pub fn record(&mut self, record: &GenerationPerformanceRecord) {
        let json_line = match serde_json::to_string(record) {
            Ok(json) => json,
            Err(serialization_error) => {
                tracing::warn!(
                    error = %serialization_error,
                    "failed to serialize generation performance record"
                );
                return;
            }
        };
        if let Err(write_error) = writeln!(self.writer, "{json_line}") {
            tracing::warn!(
                error = %write_error,
                "failed to write generation performance record"
            );
            return;
        }
        if let Err(flush_error) = self.writer.flush() {
            tracing::warn!(
                error = %flush_error,
                "failed to flush generation performance log"
            );
        }
    }
}

/// Returns the current time as milliseconds since Unix epoch.
pub fn unix_epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(0)
}
