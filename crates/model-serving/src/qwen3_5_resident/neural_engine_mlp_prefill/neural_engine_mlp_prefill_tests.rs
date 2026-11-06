use std::cell::{Cell, RefCell};
use std::time::{Duration, Instant};

use astronomical_ipc_protocol::NeuralEngineChunkPlacement;

use super::{ChunkFitDecision, NeuralEngineMlpPrefill, neural_engine_tokens_for_chunk};
use crate::qwen3_5_resident::neural_engine_chunk_comparison::{
    ChunkComparison, ChunkComparisonVerdict,
};

/// A prefill whose programs have proven worthwhile and already led with the GPU, so only
/// memory decides its next chunk.
fn prefill_for_chunk(chunk_token_count: u32) -> NeuralEngineMlpPrefill {
    let mut chunk_comparison = ChunkComparison::already_decided(
        chunk_token_count as usize,
        ChunkComparisonVerdict::Worthwhile { gain: 1.2 },
    );
    let now = Instant::now();
    chunk_comparison.choose_engines(now);
    chunk_comparison.note_finished_chunk(0, chunk_token_count as usize, Duration::ZERO, now);
    NeuralEngineMlpPrefill {
        layer_programs: RefCell::new(Vec::new()),
        chunk_token_count,
        neural_engine_token_count: 896,
        measured_program_bytes: 13,
        surface_bytes: 1,
        has_failed: Cell::new(false),
        is_paused: Cell::new(false),
        does_chunk_fit_memory: Cell::new(true),
        does_chunk_use_programs: Cell::new(true),
        chunk_comparison: RefCell::new(chunk_comparison),
        speed_record: None,
        has_reported_verdict: Cell::new(true),
    }
}

fn decide_full_chunk(
    prefill: &NeuralEngineMlpPrefill,
    projected_chunk_peak_bytes: u64,
    active_memory_ceiling_bytes: u64,
) -> ChunkFitDecision {
    prefill.decide_chunk(
        prefill.chunk_token_count as usize,
        projected_chunk_peak_bytes,
        active_memory_ceiling_bytes,
        Instant::now(),
    )
}

#[test]
fn should_take_the_measured_share_of_the_default_chunk() {
    assert_eq!(neural_engine_tokens_for_chunk(2048), Some(960));
}

#[test]
fn should_round_the_share_down_to_a_whole_multiple_of_the_granularity() {
    assert_eq!(neural_engine_tokens_for_chunk(1000), Some(448));
    assert_eq!(neural_engine_tokens_for_chunk(137), Some(64));
}

#[test]
fn should_refuse_a_chunk_too_small_for_one_multiple_of_the_granularity() {
    assert_eq!(neural_engine_tokens_for_chunk(136), None);
    assert_eq!(neural_engine_tokens_for_chunk(1), None);
    assert_eq!(neural_engine_tokens_for_chunk(0), None);
}

#[test]
fn should_not_overflow_on_the_largest_chunk() {
    assert!(neural_engine_tokens_for_chunk(u32::MAX).is_some());
}

#[test]
fn should_handle_only_exactly_the_configured_chunk() {
    let prefill = prefill_for_chunk(2048);

    assert!(prefill.handles(2048));
    assert!(!prefill.handles(2047));
    assert!(!prefill.handles(2049));
    assert!(!prefill.handles(1));
    assert!(!prefill.handles(-2048));
}

#[test]
fn should_hand_a_chunk_to_the_gpu_only_while_told_not_to_use_the_programs() {
    let prefill = prefill_for_chunk(2048);

    prefill.does_chunk_use_programs.set(false);
    assert!(!prefill.handles(2048));

    prefill.does_chunk_use_programs.set(true);
    assert!(prefill.handles(2048));
}

#[test]
fn should_keep_handling_chunks_that_fit_beside_the_programs() {
    let prefill = prefill_for_chunk(2048);

    assert_eq!(
        decide_full_chunk(&prefill, 19, 32),
        ChunkFitDecision {
            does_chunk_fit_memory: true,
            has_changed: false,
        }
    );
    assert!(prefill.handles(2048));
}

#[test]
fn should_hand_a_chunk_to_the_gpu_when_it_would_pass_the_ceiling() {
    let prefill = prefill_for_chunk(2048);

    assert_eq!(
        decide_full_chunk(&prefill, 20, 32),
        ChunkFitDecision {
            does_chunk_fit_memory: false,
            has_changed: true,
        }
    );
    assert!(!prefill.handles(2048));
}

#[test]
fn should_report_no_change_while_chunks_stay_on_the_gpu() {
    let prefill = prefill_for_chunk(2048);
    decide_full_chunk(&prefill, 20, 32);

    assert!(!decide_full_chunk(&prefill, 21, 32).has_changed);
}

#[test]
fn should_use_the_programs_again_when_the_next_prompt_fits() {
    let prefill = prefill_for_chunk(2048);
    decide_full_chunk(&prefill, 20, 32);

    assert!(decide_full_chunk(&prefill, 6, 32).has_changed);
    assert!(prefill.handles(2048));
}

#[test]
fn should_follow_a_raised_ceiling() {
    let prefill = prefill_for_chunk(2048);
    decide_full_chunk(&prefill, 20, 32);

    assert!(decide_full_chunk(&prefill, 20, 33).does_chunk_fit_memory);
}

#[test]
fn should_stop_handling_chunks_after_a_failure() {
    let prefill = prefill_for_chunk(2048);

    prefill.has_failed.set(true);

    assert!(!prefill.handles(2048));
}

#[test]
fn should_lead_with_the_gpu_on_a_full_chunk_while_the_programs_are_cold() {
    let prefill = prefill_for_chunk(2048);
    prefill
        .chunk_comparison
        .replace(ChunkComparison::already_decided(
            2048,
            ChunkComparisonVerdict::Worthwhile { gain: 1.2 },
        ));

    decide_full_chunk(&prefill, 6, 32);
    assert!(!prefill.handles(2048));
    prefill.note_finished_chunk(0, 2048, Duration::from_millis(1), Instant::now());

    decide_full_chunk(&prefill, 6, 32);
    assert!(prefill.handles(2048));
}

#[test]
fn should_keep_a_shorter_chunk_off_the_programs_and_out_of_the_comparison() {
    let prefill = prefill_for_chunk(2048);

    prefill.decide_chunk(512, 6, 32, Instant::now());

    assert!(!prefill.does_chunk_use_programs.get());
}

#[test]
fn should_compare_only_chunks_that_fit_memory() {
    let prefill = prefill_for_chunk(2048);
    prefill.chunk_comparison.replace(ChunkComparison::new(2048));
    let chunk_time = Duration::from_millis(100);
    for chunk_index in 0..40_usize {
        decide_full_chunk(&prefill, 20, 32);
        prefill.note_finished_chunk(chunk_index * 2048, 2048, chunk_time, Instant::now());
    }
    assert_eq!(prefill.chunk_comparison.borrow().verdict(), None);
}

#[test]
fn should_reach_a_verdict_on_chunks_that_fit_memory() {
    let prefill = prefill_for_chunk(2048);
    prefill.chunk_comparison.replace(ChunkComparison::new(2048));
    let chunk_time = Duration::from_millis(100);
    for chunk_index in 0..40_usize {
        // Every eighth chunk starts another prompt.
        decide_full_chunk(&prefill, 6, 32);
        prefill.note_finished_chunk(chunk_index % 8 * 2048, 2048, chunk_time, Instant::now());
    }
    assert_eq!(
        prefill.chunk_comparison.borrow().verdict(),
        Some(ChunkComparisonVerdict::NotWorthwhile { gain: 1.0 })
    );
}

#[test]
fn should_report_a_warm_full_chunk_on_both_engines_with_its_share() {
    let prefill = prefill_for_chunk(2048);
    decide_full_chunk(&prefill, 6, 32);

    let report = prefill.note_finished_chunk(0, 2048, Duration::from_millis(350), Instant::now());

    assert_eq!(report.placement, NeuralEngineChunkPlacement::BothEngines);
    assert_eq!(report.chunk_token_count, 2048);
    assert_eq!(report.neural_engine_token_count, 896);
    assert_eq!(report.elapsed_millis, 350);
}

#[test]
fn should_report_the_lead_of_a_cold_prompt_and_give_it_no_share() {
    let prefill = prefill_for_chunk(2048);
    prefill
        .chunk_comparison
        .replace(ChunkComparison::already_decided(
            2048,
            ChunkComparisonVerdict::Worthwhile { gain: 1.2 },
        ));
    decide_full_chunk(&prefill, 6, 32);

    let report = prefill.note_finished_chunk(0, 2048, Duration::from_millis(1), Instant::now());

    assert_eq!(
        report.placement,
        NeuralEngineChunkPlacement::GpuWakingPrograms
    );
    assert_eq!(report.neural_engine_token_count, 0);
    assert!(!report.is_prompt_warm);
}

#[test]
fn should_report_the_gpu_s_turn_while_comparing() {
    let prefill = prefill_for_chunk(2048);
    prefill.chunk_comparison.replace(ChunkComparison::new(2048));
    decide_full_chunk(&prefill, 6, 32);
    prefill.note_finished_chunk(0, 2048, Duration::from_millis(1), Instant::now());
    decide_full_chunk(&prefill, 6, 32);
    let both_report =
        prefill.note_finished_chunk(2048, 2048, Duration::from_millis(1), Instant::now());
    decide_full_chunk(&prefill, 6, 32);

    let gpu_report =
        prefill.note_finished_chunk(4096, 2048, Duration::from_millis(1), Instant::now());

    assert_eq!(
        both_report.placement,
        NeuralEngineChunkPlacement::BothEngines
    );
    assert_eq!(
        gpu_report.placement,
        NeuralEngineChunkPlacement::GpuWhileComparing
    );
}

#[test]
fn should_report_memory_ahead_of_the_comparison() {
    let prefill = prefill_for_chunk(2048);
    decide_full_chunk(&prefill, 20, 32);

    let report = prefill.note_finished_chunk(0, 2048, Duration::from_millis(1), Instant::now());

    assert_eq!(report.placement, NeuralEngineChunkPlacement::GpuMemoryTight);
    assert_eq!(report.neural_engine_token_count, 0);
}

#[test]
fn should_report_a_chunk_too_short_for_the_programs() {
    let prefill = prefill_for_chunk(2048);
    prefill.decide_chunk(512, 6, 32, Instant::now());

    let report = prefill.note_finished_chunk(0, 512, Duration::from_millis(1), Instant::now());

    assert_eq!(
        report.placement,
        NeuralEngineChunkPlacement::GpuChunkNotFull
    );
    assert_eq!(report.chunk_token_count, 512);
}

#[test]
fn should_report_a_failure_ahead_of_everything_else() {
    let prefill = prefill_for_chunk(2048);
    decide_full_chunk(&prefill, 6, 32);
    prefill.has_failed.set(true);

    let report = prefill.note_finished_chunk(0, 2048, Duration::from_millis(1), Instant::now());

    assert_eq!(
        report.placement,
        NeuralEngineChunkPlacement::GpuProgramFailed
    );
    assert_eq!(report.neural_engine_token_count, 0);
}
