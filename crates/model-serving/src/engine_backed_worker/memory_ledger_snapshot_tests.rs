use astronomical_memory_policy::ledger::{
    HeldMemory, LedgerPhase, LedgerReading, MeasuredStep, MemoryGovernor, ModelMemoryShape,
};

use super::worker_memory_ledger_snapshot;

const GIBIBYTE: u64 = 1 << 30;
const READING: LedgerPhase = LedgerPhase::ReadingPrompt {
    span_token_count: 4_096,
};
const MODEL_SHAPE: ModelMemoryShape = ModelMemoryShape {
    core_weight_bytes: 3 * GIBIBYTE,
    streamable_weight_bytes: 20 * GIBIBYTE,
    reading_prompt_stream_slot_bytes: 2 * GIBIBYTE,
    writing_reply_stream_slot_bytes: GIBIBYTE / 8,
};

/// A prompt step that kept `kept_weight_bytes`, read two gibibytes of
/// weights and needed one more of MLX working buffers.
fn measured_prompt_step(kept_weight_bytes: u64, unaccounted_bytes: u64) -> MeasuredStep {
    let held_bytes = 3 * GIBIBYTE + GIBIBYTE / 2 + kept_weight_bytes;
    MeasuredStep {
        phase: READING,
        held_memory: HeldMemory {
            context_bytes: GIBIBYTE / 2,
            kept_weight_bytes,
            stream_slot_bytes: 0,
        },
        in_flight_weight_bytes: 2 * GIBIBYTE,
        active_bytes_before_step: held_bytes,
        active_bytes_after_step: held_bytes + unaccounted_bytes,
        peak_bytes_during_step: held_bytes + unaccounted_bytes + 3 * GIBIBYTE,
    }
}

/// The reading of one prompt step under `limit_bytes`, in a request that
/// holds half a gibibyte of stored state.
fn reading_after_one_prompt_step(
    limit_bytes: u64,
    kept_weight_bytes: u64,
    unaccounted_bytes: u64,
) -> LedgerReading {
    let mut memory_governor =
        MemoryGovernor::new(limit_bytes, MODEL_SHAPE).expect("a nonzero limit opens a governor");
    let _ = memory_governor.start_request(READING, GIBIBYTE / 2);
    memory_governor
        .complete_step(measured_prompt_step(kept_weight_bytes, unaccounted_bytes))
        .reading
}

#[test]
fn should_publish_the_accounts_the_held_memory_and_the_comparison_unchanged() {
    let ledger_reading = reading_after_one_prompt_step(16 * GIBIBYTE, 8 * GIBIBYTE, GIBIBYTE / 4);
    let snapshot = worker_memory_ledger_snapshot(ledger_reading);
    assert_eq!(snapshot.limit_bytes, 16 * GIBIBYTE);
    assert_eq!(snapshot.core_weight_bytes, 3 * GIBIBYTE);
    assert_eq!(snapshot.mlx_working_buffer_bytes, GIBIBYTE);
    assert_eq!(
        snapshot.kept_weight_capacity_bytes,
        ledger_reading.accounts.kept_weight_capacity_bytes
    );
    assert_eq!(snapshot.held_kept_weight_bytes, 8 * GIBIBYTE);
    assert_eq!(snapshot.held_context_bytes, GIBIBYTE / 2);
    assert_eq!(snapshot.unaccounted_bytes, GIBIBYTE / 4);
    assert_eq!(snapshot.overcounted_bytes, 0);
    assert_eq!(snapshot.largest_unaccounted_bytes, GIBIBYTE / 4);
    assert_eq!(
        snapshot.largest_reading_step_peak_beyond_held_bytes,
        3 * GIBIBYTE
    );
    assert_eq!(snapshot.largest_writing_step_peak_beyond_held_bytes, 0);
    assert_eq!(snapshot.observed_step_count, 1);
    assert!(snapshot.mlx_working_buffers_are_measured);
}

#[test]
fn should_publish_no_reading_share_when_the_ledger_leaves_no_capacity() {
    let snapshot =
        worker_memory_ledger_snapshot(reading_after_one_prompt_step(4 * GIBIBYTE, 8 * GIBIBYTE, 0));
    assert_eq!(snapshot.kept_weight_capacity_bytes, 0);
    assert!(snapshot.shortfall_bytes > 0);
    assert_eq!(snapshot.lowest_kept_share_while_reading_permille, None);
    assert!(snapshot.kept_over_capacity_bytes > 0);
}

#[test]
fn should_publish_a_full_share_when_the_engine_keeps_all_the_ledger_allows() {
    let capacity_bytes = reading_after_one_prompt_step(16 * GIBIBYTE, 8 * GIBIBYTE, 0)
        .accounts
        .kept_weight_capacity_bytes;
    let mut memory_governor =
        MemoryGovernor::new(16 * GIBIBYTE, MODEL_SHAPE).expect("a nonzero limit opens a governor");
    let _ = memory_governor.record_probe(READING, GIBIBYTE);
    let _ = memory_governor.start_request(READING, GIBIBYTE / 2);
    let snapshot = worker_memory_ledger_snapshot(
        memory_governor
            .complete_step(measured_prompt_step(capacity_bytes, 0))
            .reading,
    );
    assert_eq!(
        snapshot.lowest_kept_share_while_reading_permille,
        Some(1_000)
    );
    assert_eq!(snapshot.kept_over_capacity_bytes, 0);
}
