//! Settles the prompt chunk a Qwen3.5 text model reads prompts in, once,
//! when it loads.
//!
//! The rule itself is the text family's
//! (`text_generation::generation_loop::prompt_processing_chunk_choice`); this
//! step gathers the facts this family's load has (whether the whole model is
//! held, whether it has routed experts, which GPU generation the Mac has) and
//! writes the settled in-memory chunk back into the chunking every later
//! step reads, so no later step sees Automatic.

use ipc_protocol::{PromptProcessingChunkReport, WorkerChunkingConfiguration};
use text_generation::generation_loop::prompt_processing_chunk_choice::{
    PromptChunkGpu, PromptProcessingChunkFacts, choose_prompt_processing_chunk,
};

use super::qwen3_5_engine_selection::EngineStart;

/// The chunking with its in-memory chunk settled, the chunk prompts are read
/// in now, and what decided it.
#[derive(Clone, Debug)]
pub(super) struct SettledPromptChunk {
    pub(super) chunking: WorkerChunkingConfiguration,
    pub(super) report: PromptProcessingChunkReport,
}

/// Settles the chunk for a model that starts as `engine_start` says on a
/// GPU like `gpu`.
pub(super) fn settle_prompt_chunk(
    chunking: WorkerChunkingConfiguration,
    engine_start: EngineStart,
    gpu: PromptChunkGpu,
) -> SettledPromptChunk {
    let choice = choose_prompt_processing_chunk(PromptProcessingChunkFacts {
        configured_in_memory_chunk_tokens: chunking.prompt_processing_chunk_size_tokens,
        ssd_streaming_chunk_tokens: chunking
            .fixed_ssd_streaming_prompt_processing_chunk_size_tokens,
        holds_whole_model: engine_start.holds_whole_model,
        has_routed_experts: engine_start.has_routed_experts,
        gpu,
    });
    tracing::info!(
        chunk_tokens = choice.report.chunk_tokens,
        reason = ?choice.report.reason,
        in_memory_chunk_tokens = choice.in_memory_chunk_tokens,
        "prompt chunk settled for this load"
    );
    SettledPromptChunk {
        chunking: WorkerChunkingConfiguration {
            prompt_processing_chunk_size_tokens: Some(choice.in_memory_chunk_tokens),
            ..chunking
        },
        report: choice.report,
    }
}

/// The GPU this worker runs on, read from the architecture name MLX's Metal
/// device reports. A name that cannot be read counts as older than the M5,
/// so the load keeps the size every Mac has used.
#[cfg(target_os = "macos")]
pub(super) fn this_macs_prompt_chunk_gpu() -> PromptChunkGpu {
    match runtime_integration::gpu_architecture_name() {
        Ok(architecture_name) if runtime_integration::is_m5_or_newer_gpu(&architecture_name) => {
            PromptChunkGpu::M5OrNewer
        }
        Ok(_) => PromptChunkGpu::OlderThanM5,
        Err(read_error) => {
            tracing::warn!(
                error = %read_error,
                "the GPU architecture could not be read; the prompt chunk keeps the size for GPUs older than the M5"
            );
            PromptChunkGpu::OlderThanM5
        }
    }
}

/// A build without Metal runs MLX on the CPU.
#[cfg(not(target_os = "macos"))]
pub(super) const fn this_macs_prompt_chunk_gpu() -> PromptChunkGpu {
    PromptChunkGpu::CpuOnly
}

#[cfg(test)]
#[path = "qwen3_5_prompt_chunk_settlement_tests.rs"]
mod qwen3_5_prompt_chunk_settlement_tests;
