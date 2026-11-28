use ipc_protocol::worker_chunking_test_support::worker_chunking_with_prompt_chunks;
use ipc_protocol::{
    PromptProcessingChunkReason, PromptProcessingChunkReport, WorkerChunkingConfiguration,
};
use text_generation::generation_loop::prompt_processing_chunk_choice::PromptChunkGpu;

use super::super::qwen3_5_engine_selection::EngineStart;
use super::settle_prompt_chunk;

fn chunking_with(in_memory_chunk_tokens: Option<u32>) -> WorkerChunkingConfiguration {
    worker_chunking_with_prompt_chunks(in_memory_chunk_tokens, 4_096)
}

const HELD_EXPERT_MODEL: EngineStart = EngineStart {
    holds_whole_model: true,
    has_routed_experts: true,
};

#[test]
fn should_settle_automatic_to_4096_for_a_held_expert_model_on_an_m5() {
    let settled = settle_prompt_chunk(
        chunking_with(None),
        HELD_EXPERT_MODEL,
        PromptChunkGpu::M5OrNewer,
    );

    assert_eq!(
        settled.chunking.prompt_processing_chunk_size_tokens,
        Some(4_096)
    );
    assert_eq!(
        settled.report,
        PromptProcessingChunkReport {
            chunk_tokens: 4_096,
            reason: PromptProcessingChunkReason::ExpertModelOnM5OrNewer,
        }
    );
}

#[test]
fn should_settle_automatic_even_when_the_model_streams_from_the_ssd() {
    let streaming_expert_model = EngineStart {
        holds_whole_model: false,
        has_routed_experts: true,
    };

    let settled = settle_prompt_chunk(
        chunking_with(None),
        streaming_expert_model,
        PromptChunkGpu::OlderThanM5,
    );

    assert_eq!(
        settled.chunking.prompt_processing_chunk_size_tokens,
        Some(2_048)
    );
    assert_eq!(
        settled.report.reason,
        PromptProcessingChunkReason::SsdStreaming
    );
    assert_eq!(settled.report.chunk_tokens, 4_096);
}

#[test]
fn should_keep_the_persons_size_and_every_other_setting() {
    let configured = chunking_with(Some(1_024));

    let settled = settle_prompt_chunk(
        configured.clone(),
        HELD_EXPERT_MODEL,
        PromptChunkGpu::M5OrNewer,
    );

    assert_eq!(settled.chunking, configured);
    assert_eq!(
        settled.report.reason,
        PromptProcessingChunkReason::Configured
    );
}
