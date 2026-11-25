//! Turns one streamed chat reply into its measured speed: when each token
//! arrived, and the token counts the engine reports at the end.
//!
//! Reply speed is the tokens after the first divided by the time between the
//! first and the last, so the time spent reading the prompt never counts
//! against it; the prompt's own speed is the prompt tokens over the time to
//! the first token.

use std::time::Duration;

/// One measured reply.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplyMeasurement {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub seconds_to_first_token: f64,
    /// `None` when fewer than two tokens arrived, so no rate exists.
    pub reply_tokens_per_second: Option<f64>,
}

impl ReplyMeasurement {
    /// Prompt tokens over the time to the first token: the speed a person
    /// waits on before anything appears, request overhead included.
    #[must_use]
    pub fn prefill_tokens_per_second(&self) -> f64 {
        if self.seconds_to_first_token <= 0.0 {
            return 0.0;
        }
        self.prompt_tokens as f64 / self.seconds_to_first_token
    }
}

/// Collects the lines of one streamed reply, each with when it arrived
/// relative to the request.
#[derive(Debug, Default)]
pub struct StreamTiming {
    first_token_after: Option<Duration>,
    last_token_after: Option<Duration>,
    prompt_tokens: u64,
    completion_tokens: u64,
}

impl StreamTiming {
    /// Reads one server-sent line. Text or reasoning in a delta is a token
    /// arriving; a usage object carries the final counts.
    pub fn record_line(&mut self, line_text: &str, arrived_after: Duration) {
        let Some(chunk_text) = line_text.strip_prefix("data:").map(str::trim) else {
            return;
        };
        let Ok(chunk_document) = serde_json::from_str::<serde_json::Value>(chunk_text) else {
            return;
        };
        if let Some(usage_counts) = chunk_document
            .get("usage")
            .filter(|usage| usage.is_object())
        {
            self.prompt_tokens = usage_counts["prompt_tokens"].as_u64().unwrap_or(0);
            self.completion_tokens = usage_counts["completion_tokens"].as_u64().unwrap_or(0);
        }
        if chunk_carries_token(&chunk_document) {
            self.first_token_after.get_or_insert(arrived_after);
            self.last_token_after = Some(arrived_after);
        }
    }

    #[must_use]
    pub fn measurement(&self) -> ReplyMeasurement {
        let first_token_after = self.first_token_after.unwrap_or_default();
        let decode_seconds = self
            .last_token_after
            .unwrap_or_default()
            .saturating_sub(first_token_after)
            .as_secs_f64();
        let reply_tokens_per_second = (self.completion_tokens > 1 && decode_seconds > 0.0)
            .then(|| (self.completion_tokens - 1) as f64 / decode_seconds);
        ReplyMeasurement {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            seconds_to_first_token: first_token_after.as_secs_f64(),
            reply_tokens_per_second,
        }
    }
}

/// Whether a chunk's delta carries text the model wrote, its answer or its
/// thinking; empty deltas and role-only deltas are not tokens.
fn chunk_carries_token(chunk_document: &serde_json::Value) -> bool {
    let Some(choices) = chunk_document["choices"].as_array() else {
        return false;
    };
    choices.iter().any(|choice| {
        ["content", "reasoning_content", "reasoning"]
            .iter()
            .any(|field_name| {
                choice["delta"][field_name]
                    .as_str()
                    .is_some_and(|text| !text.is_empty())
            })
    })
}

#[cfg(test)]
#[path = "stream_timing_tests.rs"]
mod stream_timing_tests;
