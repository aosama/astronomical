//! The prompt chunk a text model reads prompts in: the person's own size, or
//! the size Automatic picks for the model and the Mac.
//!
//! Automatic is decided once, when the model loads, from facts the load
//! already has: whether the whole model is held in memory, whether it has
//! routed experts, and which GPU generation the Mac has. Every Automatic
//! size below was measured; a case that was not measured keeps the size
//! every Mac has used.

use ipc_protocol::{PromptProcessingChunkReason, PromptProcessingChunkReport};

/// The in-memory chunk every Mac used before Automatic, and still the one
/// for models without experts: wider chunks measured no gain on a dense
/// model (a 9B 4-bit read a 15,000-token prompt within run-to-run drift at
/// 2,048, 4,096 and 8,192 on an M5 Pro), and a wider chunk holds more
/// working buffers at once.
#[cfg(target_os = "macos")]
pub const AUTOMATIC_PROMPT_PROCESSING_CHUNK_TOKENS: u32 = 2_048;
/// Builds without Metal run MLX on the CPU, where a prompt chunk costs
/// orders of magnitude more than on a GPU: tens of seconds for 64 tokens of
/// the smallest Qwen3.5 model. The engine reads one chunk at load to measure
/// its working buffers, so a 2,048-token chunk would keep a model loading for
/// most of an hour; 64 keeps the load and every progress report short.
#[cfg(not(target_os = "macos"))]
pub const AUTOMATIC_PROMPT_PROCESSING_CHUNK_TOKENS: u32 = 64;

/// The in-memory chunk for a model with routed experts on an M5 or newer
/// GPU. Wider chunks give each expert's multiply more rows, so its tiles
/// fill: the routed experts of a 35B-A3B layer cost 3.0 µs a token at 2,048
/// tokens and 2.4 µs at 8,192. Served on an M5 Pro (Ornith 35B-A3B 4-bit, two
/// runs each, no prompt restored), 4,096 read a 15,000-token prompt 7.6%
/// faster than 2,048 and a 30,000-token prompt 2.3% faster, for 0.35 GB
/// more peak memory. 8,192 and 16,384 gained no more at 30,000 tokens and
/// cost 0.84 and 1.54 GB. The gain is not the M5 tiled kernels' alone: the
/// 6-bit 35B-A3B, whose experts run MLX's own kernels, read the same prompt
/// 6.4% faster at 4,096 (alternated runs), and under a 25 GB limit the 4-bit
/// model kept every expert held at 4,096 and still read 5.8% faster.
pub const EXPERT_MODEL_ON_M5_PROMPT_PROCESSING_CHUNK_TOKENS: u32 = 4_096;

/// The GPU a model loads on, as far as the prompt chunk cares.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PromptChunkGpu {
    /// An M5 or newer: the tensor units MLX's M5 kernels use.
    M5OrNewer,
    /// An older GPU, or one whose generation could not be read.
    OlderThanM5,
    /// A build without Metal, where MLX runs on the CPU.
    CpuOnly,
}

/// What the load knows when it picks the chunk.
#[derive(Clone, Copy, Debug)]
pub struct PromptProcessingChunkFacts {
    /// The in-memory chunk the person set; `None` is Automatic.
    pub configured_in_memory_chunk_tokens: Option<u32>,
    /// The chunk used while part of the model is read from the SSD.
    pub ssd_streaming_chunk_tokens: u32,
    pub holds_whole_model: bool,
    pub has_routed_experts: bool,
    pub gpu: PromptChunkGpu,
}

/// The chunk sizes the load settles on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PromptProcessingChunkChoice {
    /// The in-memory chunk, settled: the person's size or Automatic's.
    pub in_memory_chunk_tokens: u32,
    /// The chunk prompts are read in now, and what decided it.
    pub report: PromptProcessingChunkReport,
}

/// Settles the chunk for one load.
#[must_use]
pub const fn choose_prompt_processing_chunk(
    facts: PromptProcessingChunkFacts,
) -> PromptProcessingChunkChoice {
    let (in_memory_chunk_tokens, in_memory_reason) = match facts.configured_in_memory_chunk_tokens {
        Some(configured_chunk_tokens) => (
            configured_chunk_tokens,
            PromptProcessingChunkReason::Configured,
        ),
        None => automatic_in_memory_chunk(facts.has_routed_experts, facts.gpu),
    };
    let report = if facts.holds_whole_model {
        PromptProcessingChunkReport {
            chunk_tokens: in_memory_chunk_tokens,
            reason: in_memory_reason,
        }
    } else {
        PromptProcessingChunkReport {
            chunk_tokens: facts.ssd_streaming_chunk_tokens,
            reason: PromptProcessingChunkReason::SsdStreaming,
        }
    };
    PromptProcessingChunkChoice {
        in_memory_chunk_tokens,
        report,
    }
}

const fn automatic_in_memory_chunk(
    has_routed_experts: bool,
    gpu: PromptChunkGpu,
) -> (u32, PromptProcessingChunkReason) {
    match (gpu, has_routed_experts) {
        (PromptChunkGpu::CpuOnly, _) => (
            AUTOMATIC_PROMPT_PROCESSING_CHUNK_TOKENS,
            PromptProcessingChunkReason::CpuBuild,
        ),
        (_, false) => (
            AUTOMATIC_PROMPT_PROCESSING_CHUNK_TOKENS,
            PromptProcessingChunkReason::ModelWithoutExperts,
        ),
        (PromptChunkGpu::M5OrNewer, true) => (
            EXPERT_MODEL_ON_M5_PROMPT_PROCESSING_CHUNK_TOKENS,
            PromptProcessingChunkReason::ExpertModelOnM5OrNewer,
        ),
        (PromptChunkGpu::OlderThanM5, true) => (
            AUTOMATIC_PROMPT_PROCESSING_CHUNK_TOKENS,
            PromptProcessingChunkReason::GpuOlderThanM5,
        ),
    }
}

#[cfg(test)]
#[path = "prompt_processing_chunk_choice_tests.rs"]
mod prompt_processing_chunk_choice_tests;
