use astronomical_memory_policy::ledger::{
    HeldMemory, LedgerObserver, LedgerPhase, ModelMemoryShape, ObservedStep,
};

use super::worker_memory_ledger_snapshot;

const GIBIBYTE: u64 = 1 << 30;
const MODEL_SHAPE: ModelMemoryShape = ModelMemoryShape {
    core_weight_bytes: 3 * GIBIBYTE,
    streamable_weight_bytes: 20 * GIBIBYTE,
    reading_prompt_stream_slot_bytes: 2 * GIBIBYTE,
    writing_reply_stream_slot_bytes: GIBIBYTE / 8,
};

fn observed_prompt_step(unaccounted_bytes: u64) -> ObservedStep {
    let held_bytes = 3 * GIBIBYTE + GIBIBYTE / 2 + 8 * GIBIBYTE;
    ObservedStep {
        phase: LedgerPhase::ReadingPrompt {
            span_token_count: 4_096,
        },
        limit_bytes: 16 * GIBIBYTE,
        request_context_bytes: GIBIBYTE / 2,
        held_memory: HeldMemory {
            context_bytes: GIBIBYTE / 2,
            kept_weight_bytes: 8 * GIBIBYTE,
            stream_slot_bytes: 0,
        },
        active_bytes_before_step: held_bytes,
        active_bytes_after_step: held_bytes + unaccounted_bytes,
        peak_bytes_during_step: held_bytes + unaccounted_bytes + 3 * GIBIBYTE,
    }
}

#[test]
fn should_publish_the_accounts_the_held_memory_and_the_comparison_unchanged() {
    let mut ledger_observer =
        LedgerObserver::new(16 * GIBIBYTE, MODEL_SHAPE).expect("a nonzero limit opens an observer");
    let ledger_reading = ledger_observer.observe_step(observed_prompt_step(GIBIBYTE / 4));
    let snapshot = worker_memory_ledger_snapshot(ledger_reading);
    assert_eq!(snapshot.limit_bytes, 16 * GIBIBYTE);
    assert_eq!(snapshot.core_weight_bytes, 3 * GIBIBYTE);
    assert_eq!(snapshot.working_memory_bytes, GIBIBYTE);
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
    assert!(snapshot.working_memory_is_measured);
}

#[test]
fn should_publish_no_reading_share_when_the_ledger_leaves_no_capacity() {
    let mut ledger_observer =
        LedgerObserver::new(4 * GIBIBYTE, MODEL_SHAPE).expect("a nonzero limit opens an observer");
    let mut observed_step = observed_prompt_step(0);
    observed_step.limit_bytes = 4 * GIBIBYTE;
    let snapshot = worker_memory_ledger_snapshot(ledger_observer.observe_step(observed_step));
    assert_eq!(snapshot.kept_weight_capacity_bytes, 0);
    assert!(snapshot.shortfall_bytes > 0);
    assert_eq!(snapshot.lowest_kept_share_while_reading_permille, None);
    assert!(snapshot.kept_over_capacity_bytes > 0);
}

#[test]
fn should_publish_a_full_share_when_the_engine_keeps_all_the_ledger_allows() {
    let mut ledger_observer =
        LedgerObserver::new(16 * GIBIBYTE, MODEL_SHAPE).expect("a nonzero limit opens an observer");
    let first_reading = ledger_observer.observe_step(observed_prompt_step(0));
    let mut full_step = observed_prompt_step(0);
    full_step.held_memory.kept_weight_bytes = first_reading.accounts.kept_weight_capacity_bytes;
    ledger_observer.finish_request(0);
    let snapshot = worker_memory_ledger_snapshot(ledger_observer.observe_step(full_step));
    assert_eq!(
        snapshot.lowest_kept_share_while_reading_permille,
        Some(1_000)
    );
    assert_eq!(snapshot.kept_over_capacity_bytes, 0);
}
