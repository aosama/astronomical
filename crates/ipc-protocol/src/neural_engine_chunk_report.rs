//! Where one prompt chunk ran, when a model has Neural Engine programs attached.

use serde::{Deserialize, Serialize};

/// Which engines ran a chunk, and why when the Neural Engine sat one out. A reader needs
/// the reason because each has a different answer: nothing to do, or a higher ceiling.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NeuralEngineChunkPlacement {
    BothEngines,
    /// The GPU's turn while the model's first long prompts compare the engines.
    GpuWhileComparing,
    /// The first chunk after an idle spell, which leads on the GPU.
    GpuWakingPrograms,
    /// The chunk would not fit under the memory ceiling beside the programs.
    GpuMemoryTight,
    /// Programs have a fixed shape, so a prompt's shorter chunks cannot use them.
    GpuChunkNotFull,
    GpuProgramFailed,
}

/// One finished chunk, sent whether or not the Neural Engine ran it, so a reader can say
/// what share of the work it really did rather than what share it was given.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NeuralEngineChunkReport {
    pub placement: NeuralEngineChunkPlacement,
    pub chunk_token_count: u32,
    pub neural_engine_token_count: u32,
    pub elapsed_millis: u64,
    /// Whether this chunk's prompt began with the programs already awake. Only such a
    /// chunk can be credited with time saved: a prompt that wakes cold programs can run
    /// slower overall than the GPU alone, whatever its later chunks gain.
    pub is_prompt_warm: bool,
}

#[cfg(test)]
#[path = "neural_engine_chunk_report_tests.rs"]
mod tests;
