//! Switchable completion attribution log.
//!
//! The completion attribution log captures *what* a chat generation emitted —
//! each tool call's function name and arguments JSON plus the completion
//! reason — so tool-call argument pollution, foreign-dialect regressions, and
//! fail-closed retry loops are diagnosable instead of invisible. It is the
//! symmetric sibling of the generation performance log: performance attribution
//! answers "how well did it run?" (timing); completion attribution answers
//! "what did it emit?" (the tool calls and arguments).
//!
//! The log is off by default. An operator enables it through the
//! `diagnostics.completion_attribution_enabled` configuration flag, mirroring
//! `diagnostics.performance_attribution_enabled`. When disabled, the sink is
//! `None` and `record` is a no-op with no allocation on the generation path.

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::Serialize;

use crate::generation_performance_log::unix_epoch_millis;

/// Maximum number of bytes of arguments JSON recorded verbatim. Arguments at or
/// under this cap are written in full so polluted keys (`content`, `hash`,
/// duplicates) are visible. Larger arguments are truncated and hashed so the
/// log never grows without bound while still correlating identical payloads.
const COMPLETION_ARGUMENTS_FULL_LIMIT_BYTES: usize = 8_192;

/// One row in the completion attribution log, written when a chat generation
/// completes.
///
/// Fields are chosen to answer "what did the model emit?" at a glance:
/// - Identity: `request_id`, `model_id`
/// - Outcome: `completion_reason`
/// - Content: `tool_calls` (names + bounded arguments)
#[derive(Clone, Debug, Serialize)]
pub struct CompletionAttributionRecord {
    /// Unix epoch milliseconds when the record was written (request completion time).
    pub timestamp_millis: u64,
    /// Supervisor-local monotonic request identifier.
    pub request_id: u64,
    /// The model that produced this generation.
    pub model_id: String,
    /// Why the generation stopped: `end_of_sequence`, `tool_calls`,
    /// `maximum_output_tokens`, or `cancelled`.
    pub completion_reason: String,
    /// The tool calls the model emitted, in emission order. Empty for
    /// non-tool-call completions.
    pub tool_calls: Vec<CompletionToolCallRecord>,
}

/// One emitted tool call, with arguments bounded for safe logging.
#[derive(Clone, Debug, Serialize)]
pub struct CompletionToolCallRecord {
    /// Emission order of this tool call within the generation.
    pub tool_call_index: u16,
    /// The function name the model emitted.
    pub function_name: String,
    /// The arguments JSON, bounded so the log never grows without bound.
    pub arguments: CompletionArgumentsRecord,
}

/// Bounded arguments payload for one tool call.
///
/// `size_bytes` and `sha256` are always present so identical payloads correlate
/// regardless of truncation. `json` is the full arguments string when at or
/// under the cap, or a truncated preview when over the cap; `truncated`
/// distinguishes the two.
#[derive(Clone, Debug, Serialize)]
pub struct CompletionArgumentsRecord {
    /// Original argument length in bytes, before any truncation.
    pub size_bytes: usize,
    /// SHA-256 of the full original arguments, never of the truncation.
    pub sha256: String,
    /// The arguments JSON: full when `truncated` is false, a bounded preview
    /// when `truncated` is true.
    pub json: String,
    /// Whether `json` is a truncation rather than the full arguments.
    pub truncated: bool,
}

impl CompletionToolCallRecord {
    /// Builds one bounded tool-call record from the raw arguments JSON.
    ///
    /// Arguments at or under the cap are recorded verbatim. Larger arguments
    /// are truncated to the cap and the full original is hashed so identical
    /// payloads correlate regardless of truncation.
    #[must_use]
    pub fn from_arguments(tool_call_index: u16, function_name: &str, arguments_json: &str) -> Self {
        let size_bytes = arguments_json.len();
        let sha256 = sha256_hex(arguments_json.as_bytes());
        let (json, truncated) = if size_bytes <= COMPLETION_ARGUMENTS_FULL_LIMIT_BYTES {
            (arguments_json.to_owned(), false)
        } else {
            let truncated_at =
                arguments_json.floor_char_boundary(COMPLETION_ARGUMENTS_FULL_LIMIT_BYTES);
            (arguments_json[..truncated_at].to_owned(), true)
        };
        Self {
            tool_call_index,
            function_name: function_name.to_owned(),
            arguments: CompletionArgumentsRecord {
                size_bytes,
                sha256,
                json,
                truncated,
            },
        }
    }
}

/// Append-only JSONL writer for completion attribution records, gated by
/// configuration.
///
/// Opens `completion.jsonl` in the configured log directory when enabled.
/// When disabled, the sink is `None` and `record` is a no-op so the
/// generation path pays no attribution cost. Uses a synchronous `BufWriter`
/// because the volume is low (one line per completed request).
pub struct CompletionAttributionLog {
    sink: Option<BufWriter<Box<dyn Write + Send>>>,
}

impl CompletionAttributionLog {
    /// Creates a no-overhead log for the disabled configuration.
    #[must_use]
    pub const fn disabled() -> Self {
        Self { sink: None }
    }

    /// Opens (or creates) the completion log when enabled.
    ///
    /// Returns a disabled log (no file, no writes) when the flag is false so
    /// normal inference never touches the attribution path.
    pub fn open(
        log_directory: &Path,
        completion_attribution_enabled: bool,
    ) -> std::io::Result<Self> {
        if !completion_attribution_enabled {
            return Ok(Self::disabled());
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_directory.join("completion.jsonl"))?;
        Ok(Self {
            sink: Some(BufWriter::new(Box::new(file))),
        })
    }

    /// Appends one completion record as a JSON line when enabled.
    ///
    /// No-op when disabled. Flushes after each write so records are durable
    /// immediately, matching the generation performance log contract.
    pub fn record(&mut self, record: &CompletionAttributionRecord) {
        let Some(writer) = self.sink.as_mut() else {
            return;
        };
        let json_line = match serde_json::to_string(record) {
            Ok(json) => json,
            Err(serialization_error) => {
                tracing::warn!(
                    error = %serialization_error,
                    "failed to serialize completion attribution record"
                );
                return;
            }
        };
        if let Err(write_error) = writeln!(writer, "{json_line}") {
            tracing::warn!(
                error = %write_error,
                "failed to write completion attribution record"
            );
            return;
        }
        if let Err(flush_error) = writer.flush() {
            tracing::warn!(
                error = %flush_error,
                "failed to flush completion attribution log"
            );
        }
    }

    /// Records one completion from the raw emitted tool calls.
    ///
    /// Binds the accumulated raw tool calls into bounded records and writes the
    /// row when enabled. The caller supplies the timestamp so the completion
    /// event owns request-correlation time, mirroring the performance log.
    pub fn record_completion(
        &mut self,
        timestamp_millis: u64,
        request_id: u64,
        model_id: &str,
        completion_reason: &str,
        completed_tool_calls: &[CompletedToolCall],
    ) {
        let tool_calls = completed_tool_calls
            .iter()
            .map(|completed| {
                CompletionToolCallRecord::from_arguments(
                    completed.tool_call_index,
                    &completed.function_name,
                    &completed.arguments_json,
                )
            })
            .collect();
        self.record(&CompletionAttributionRecord {
            timestamp_millis,
            request_id,
            model_id: model_id.to_owned(),
            completion_reason: completion_reason.to_owned(),
            tool_calls,
        });
    }
}

/// Raw accumulated tool call, captured at emission time and bounded at write
/// time. Stored on the active request so the completion event can attribute the
/// emitted content without re-reading the stream.
#[derive(Clone, Debug)]
pub struct CompletedToolCall {
    /// Emission order of this tool call within the generation.
    pub tool_call_index: u16,
    /// The function name the model emitted.
    pub function_name: String,
    /// The raw arguments JSON exactly as the model emitted it.
    pub arguments_json: String,
}

/// Convenience for the completion event to record at the request's wall-clock
/// time without each caller repeating the clock call.
pub(crate) fn record_completion_at_now(
    log: &mut CompletionAttributionLog,
    request_id: u64,
    model_id: &str,
    completion_reason: &str,
    completed_tool_calls: &[CompletedToolCall],
) {
    log.record_completion(
        unix_epoch_millis(),
        request_id,
        model_id,
        completion_reason,
        completed_tool_calls,
    );
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    let mut hex_string = String::with_capacity(digest.len() * 2);
    for digest_byte in digest {
        hex_string.push_str(&format!("{digest_byte:02x}"));
    }
    hex_string
}
