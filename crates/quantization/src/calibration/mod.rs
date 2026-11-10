//! Calibration sequences for the exact sensitivity sweep.
//!
//! The bundled mix is OptiQ's `optiq.jsonl`, vendored verbatim under
//! `crates/quantization/calibration/` with its MIT license beside it: 40
//! hand-curated samples across six domains (prose, thinking traces, code
//! reasoning, agent loops, tool calling, constraint-bearing instructions).
//! Diversity of activation patterns is the point; a prose-only corpus leaves
//! the sweep blind to the layers tool calling depends on.
//!
//! Each JSONL line is one of:
//! - `{"domain": "prose", "text": "..."}`, raw text encoded directly;
//! - `{"domain": "thought", "messages": [...]}`, a chat;
//! - `{"domain": "tool", "messages": [...], "tools": [...]}`, a chat with
//!   tool definitions.
//!
//! OptiQ renders chat samples through the tokenizer's Jinja chat template,
//! and so does this port ([`chat_template`]), so calibration sees the
//! model's own role markers, tool preamble and reasoning tags. A sample the
//! template refuses, or any chat sample of a model without a template, is
//! rendered by role-prefixed concatenation (`[role]: content` joined by
//! blank lines), which is OptiQ's fallback too.
//!
//! Every sample is encoded, the token streams are concatenated, and the
//! stream is cut into fixed-length sequences whose starting offsets are
//! shuffled by a seeded generator and truncated to the requested count.

pub mod chat_template;
pub mod mix;
pub mod python_json;
pub mod sample;
pub mod seeded_shuffle;
pub mod tokenizer;

pub use mix::{
    BUNDLED_OPTIQ_MIX_JSONL, CalibrationMix, CalibrationMixSummary, DEFAULT_SEQUENCE_LENGTH_TOKENS,
    DEFAULT_SHUFFLE_SEED,
};
pub use sample::{CalibrationMessage, CalibrationSample};
pub use tokenizer::{CalibrationTokenizer, HuggingFaceCalibrationTokenizer};

#[cfg(test)]
#[path = "calibration_tests.rs"]
mod calibration_tests;
