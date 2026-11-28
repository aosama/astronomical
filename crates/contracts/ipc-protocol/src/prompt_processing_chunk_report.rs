//! The prompt chunk the loaded model reads prompts in, and why.

use serde::{Deserialize, Serialize};

/// Reported by the engine once a load settles, so a person can see which
/// chunk size is in use and what decided it, without reading the worker log.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PromptProcessingChunkReport {
    /// Tokens of a prompt read in one forward.
    pub chunk_tokens: u32,
    pub reason: PromptProcessingChunkReason,
}

/// What decided the chunk in use.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptProcessingChunkReason {
    /// The in-memory chunk the person set.
    Configured,
    /// Automatic: a model with experts on an M5 or newer GPU, where wider
    /// chunks give each expert's multiply more rows to fill its tiles.
    ExpertModelOnM5OrNewer,
    /// Automatic: a model without experts, which measured no gain from wider
    /// chunks.
    ModelWithoutExperts,
    /// Automatic: a model with experts on a GPU older than the M5, where the
    /// wider chunk has not been measured.
    GpuOlderThanM5,
    /// Automatic: a build without Metal, where MLX runs on the CPU and a
    /// small chunk keeps loading and progress reports short.
    CpuBuild,
    /// Part of the model is read from the SSD, so the SSD streaming chunk is
    /// in use and the in-memory chunk is not.
    SsdStreaming,
}

#[cfg(test)]
#[path = "prompt_processing_chunk_report_tests.rs"]
mod prompt_processing_chunk_report_tests;
