use astronomical_ipc_protocol::{
    NeuralEngineChunkPlacement, NeuralEngineChunkReport, NeuralEnginePrefillStatus,
};

use super::NeuralEngineSessionTotals;

fn both_engines_chunk(elapsed_millis: u64, is_prompt_warm: bool) -> NeuralEngineChunkReport {
    NeuralEngineChunkReport {
        placement: NeuralEngineChunkPlacement::BothEngines,
        chunk_token_count: 2_048,
        neural_engine_token_count: 960,
        elapsed_millis,
        is_prompt_warm,
    }
}

fn gpu_chunk(placement: NeuralEngineChunkPlacement) -> NeuralEngineChunkReport {
    NeuralEngineChunkReport {
        placement,
        chunk_token_count: 2_048,
        neural_engine_token_count: 0,
        elapsed_millis: 1_000,
        is_prompt_warm: true,
    }
}

fn active(measured_speed_percent: Option<u32>) -> NeuralEnginePrefillStatus {
    NeuralEnginePrefillStatus::Active {
        layer_program_count: 28,
        neural_engine_tokens_per_chunk: 960,
        chunk_token_count: 2_048,
        measured_memory_bytes: 1,
        measured_speed_percent,
    }
}

fn totals_at_speed(measured_speed_percent: u32) -> NeuralEngineSessionTotals {
    let mut totals = NeuralEngineSessionTotals::empty();
    totals.note_prefill_status(Some(&active(Some(measured_speed_percent))));
    totals
}

#[test]
fn should_credit_a_warm_chunk_with_what_the_gpu_alone_would_have_added() {
    let mut totals = totals_at_speed(128);

    totals.record_chunk(both_engines_chunk(1_000, true));

    assert_eq!(totals.estimated_saved_millis, 280);
    assert_eq!(totals.prompt_token_count, 2_048);
    assert_eq!(totals.neural_engine_token_count, 960);
}

#[test]
fn should_give_no_credit_to_a_prompt_that_woke_cold_programs() {
    let mut totals = totals_at_speed(128);

    totals.record_chunk(both_engines_chunk(1_000, false));

    assert_eq!(totals.estimated_saved_millis, 0);
    assert_eq!(totals.neural_engine_token_count, 960);
}

#[test]
fn should_count_gpu_chunks_as_prompt_work_with_no_credit() {
    let mut totals = totals_at_speed(128);

    totals.record_chunk(gpu_chunk(NeuralEngineChunkPlacement::GpuMemoryTight));

    assert_eq!(totals.prompt_token_count, 2_048);
    assert_eq!(totals.neural_engine_token_count, 0);
    assert_eq!(totals.estimated_saved_millis, 0);
    assert_eq!(
        totals.last_chunk_placement,
        Some(NeuralEngineChunkPlacement::GpuMemoryTight)
    );
}

#[test]
fn should_hold_credit_until_the_comparison_finds_the_programs_worthwhile() {
    let mut totals = NeuralEngineSessionTotals::empty();
    totals.note_prefill_status(Some(&active(None)));
    totals.record_chunk(both_engines_chunk(1_000, true));
    assert_eq!(totals.estimated_saved_millis, 0);

    totals.note_prefill_status(Some(&active(Some(110))));

    assert_eq!(totals.estimated_saved_millis, 100);
}

#[test]
fn should_drop_waiting_credit_when_the_programs_are_turned_off() {
    let mut totals = NeuralEngineSessionTotals::empty();
    totals.record_chunk(both_engines_chunk(1_000, true));

    totals.note_prefill_status(Some(&NeuralEnginePrefillStatus::Unavailable {
        reason: "slower".to_owned(),
    }));
    totals.note_prefill_status(Some(&active(Some(150))));

    assert_eq!(totals.estimated_saved_millis, 0);
}

#[test]
fn should_keep_earned_time_when_the_model_changes() {
    let mut totals = totals_at_speed(128);
    totals.record_chunk(both_engines_chunk(1_000, true));

    totals.note_prefill_status(None);
    totals.record_chunk(both_engines_chunk(1_000, true));

    assert_eq!(totals.estimated_saved_millis, 280);
    assert_eq!(totals.prompt_token_count, 4_096);
}

#[test]
fn should_not_credit_a_saved_speed_at_or_below_the_gpu() {
    let mut totals = totals_at_speed(100);
    totals.record_chunk(both_engines_chunk(1_000, true));
    assert_eq!(totals.estimated_saved_millis, 0);

    let mut slower_totals = totals_at_speed(90);
    slower_totals.record_chunk(both_engines_chunk(1_000, true));
    assert_eq!(slower_totals.estimated_saved_millis, 0);
}

#[test]
fn should_not_overflow_on_enormous_figures() {
    let mut totals = totals_at_speed(u32::MAX);
    totals.record_chunk(both_engines_chunk(u64::MAX, true));
    totals.record_chunk(both_engines_chunk(u64::MAX, true));
    assert!(totals.estimated_saved_millis > 0);
}
