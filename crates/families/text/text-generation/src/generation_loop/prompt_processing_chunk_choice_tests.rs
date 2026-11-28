use ipc_protocol::{PromptProcessingChunkReason, PromptProcessingChunkReport};

use super::{
    AUTOMATIC_PROMPT_PROCESSING_CHUNK_TOKENS, EXPERT_MODEL_ON_M5_PROMPT_PROCESSING_CHUNK_TOKENS,
    PromptChunkGpu, PromptProcessingChunkFacts, choose_prompt_processing_chunk,
};

const SSD_STREAMING_CHUNK_TOKENS: u32 = 4_096;

fn held_whole(has_routed_experts: bool, gpu: PromptChunkGpu) -> PromptProcessingChunkFacts {
    PromptProcessingChunkFacts {
        configured_in_memory_chunk_tokens: None,
        ssd_streaming_chunk_tokens: SSD_STREAMING_CHUNK_TOKENS,
        holds_whole_model: true,
        has_routed_experts,
        gpu,
    }
}

fn report_of(facts: PromptProcessingChunkFacts) -> PromptProcessingChunkReport {
    choose_prompt_processing_chunk(facts).report
}

#[test]
fn should_pick_4096_for_an_expert_model_on_an_m5() {
    let choice = choose_prompt_processing_chunk(held_whole(true, PromptChunkGpu::M5OrNewer));

    assert_eq!(choice.in_memory_chunk_tokens, 4_096);
    assert_eq!(
        choice.report,
        PromptProcessingChunkReport {
            chunk_tokens: EXPERT_MODEL_ON_M5_PROMPT_PROCESSING_CHUNK_TOKENS,
            reason: PromptProcessingChunkReason::ExpertModelOnM5OrNewer,
        }
    );
}

#[test]
fn should_keep_the_automatic_size_for_a_model_without_experts_on_an_m5() {
    assert_eq!(
        report_of(held_whole(false, PromptChunkGpu::M5OrNewer)),
        PromptProcessingChunkReport {
            chunk_tokens: AUTOMATIC_PROMPT_PROCESSING_CHUNK_TOKENS,
            reason: PromptProcessingChunkReason::ModelWithoutExperts,
        }
    );
}

#[test]
fn should_keep_the_automatic_size_for_an_expert_model_on_an_older_gpu() {
    assert_eq!(
        report_of(held_whole(true, PromptChunkGpu::OlderThanM5)),
        PromptProcessingChunkReport {
            chunk_tokens: AUTOMATIC_PROMPT_PROCESSING_CHUNK_TOKENS,
            reason: PromptProcessingChunkReason::GpuOlderThanM5,
        }
    );
}

#[test]
fn should_keep_the_small_cpu_size_on_a_build_without_metal() {
    assert_eq!(
        report_of(held_whole(true, PromptChunkGpu::CpuOnly)).reason,
        PromptProcessingChunkReason::CpuBuild
    );
}

#[test]
fn should_use_the_size_the_person_set_over_automatic() {
    let facts = PromptProcessingChunkFacts {
        configured_in_memory_chunk_tokens: Some(1_024),
        ..held_whole(true, PromptChunkGpu::M5OrNewer)
    };

    let choice = choose_prompt_processing_chunk(facts);

    assert_eq!(choice.in_memory_chunk_tokens, 1_024);
    assert_eq!(
        choice.report,
        PromptProcessingChunkReport {
            chunk_tokens: 1_024,
            reason: PromptProcessingChunkReason::Configured,
        }
    );
}

#[test]
fn should_report_the_ssd_streaming_chunk_while_part_of_the_model_streams() {
    let facts = PromptProcessingChunkFacts {
        holds_whole_model: false,
        ..held_whole(true, PromptChunkGpu::M5OrNewer)
    };

    let choice = choose_prompt_processing_chunk(facts);

    assert_eq!(
        choice.report,
        PromptProcessingChunkReport {
            chunk_tokens: SSD_STREAMING_CHUNK_TOKENS,
            reason: PromptProcessingChunkReason::SsdStreaming,
        }
    );
    assert_eq!(choice.in_memory_chunk_tokens, 4_096);
}

#[test]
fn should_keep_a_one_token_chunk_the_person_set() {
    let facts = PromptProcessingChunkFacts {
        configured_in_memory_chunk_tokens: Some(1),
        ..held_whole(false, PromptChunkGpu::OlderThanM5)
    };

    assert_eq!(report_of(facts).chunk_tokens, 1);
}
