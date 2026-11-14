use super::{LedgerObserver, ObservedStep, kept_share_permille};
use crate::ledger::model_shape::ModelMemoryShape;
use crate::ledger::phase::LedgerPhase;
use crate::ledger::reconciliation::HeldMemory;

const GIBIBYTE: u64 = 1 << 30;
const MEBIBYTE: u64 = 1 << 20;
const LIMIT_BYTES: u64 = 16 * GIBIBYTE;
const READING: LedgerPhase = LedgerPhase::ReadingPrompt {
    span_token_count: 4_096,
};
const MODEL_SHAPE: ModelMemoryShape = ModelMemoryShape {
    core_weight_bytes: 3 * GIBIBYTE,
    streamable_weight_bytes: 20 * GIBIBYTE,
    reading_prompt_stream_slot_bytes: 2 * GIBIBYTE,
    writing_reply_stream_slot_bytes: GIBIBYTE / 8,
};

fn observer() -> LedgerObserver {
    LedgerObserver::new(LIMIT_BYTES, MODEL_SHAPE).expect("a nonzero limit opens an observer")
}

/// A prompt step that kept `kept_weight_bytes` and left `unaccounted_bytes`
/// in memory beyond what its owners explain.
fn reading_step(kept_weight_bytes: u64, unaccounted_bytes: u64) -> ObservedStep {
    let context_bytes = GIBIBYTE / 2;
    let held_bytes = 3 * GIBIBYTE + context_bytes + kept_weight_bytes;
    ObservedStep {
        phase: READING,
        limit_bytes: LIMIT_BYTES,
        request_context_bytes: context_bytes,
        held_memory: HeldMemory {
            context_bytes,
            kept_weight_bytes,
            stream_slot_bytes: 0,
        },
        active_bytes_before_step: held_bytes,
        active_bytes_after_step: held_bytes + unaccounted_bytes,
        // Two slots of weights and one gibibyte of working memory.
        peak_bytes_during_step: held_bytes + unaccounted_bytes + 3 * GIBIBYTE,
    }
}

#[test]
fn should_have_no_reading_before_any_step_is_observed() {
    assert_eq!(observer().latest_reading(), None);
}

#[test]
fn should_record_the_peak_beyond_the_slots_as_working_memory() {
    let mut observer = observer();
    let reading = observer.observe_step(reading_step(8 * GIBIBYTE, 0));
    assert_eq!(reading.step_peak_beyond_held_bytes, 3 * GIBIBYTE);
    assert_eq!(reading.accounts.working_memory_bytes, GIBIBYTE);
    assert!(reading.accounts.working_memory_is_measured);
    assert_eq!(observer.latest_reading(), Some(reading));
}

#[test]
fn should_agree_when_the_engines_owners_explain_everything() {
    let mut observer = observer();
    let reading = observer.observe_step(reading_step(8 * GIBIBYTE, 0));
    assert_eq!(reading.reconciliation.disagreement_bytes(), 0);
    assert_eq!(reading.largest_unaccounted_bytes, 0);
    assert_eq!(reading.largest_overcounted_bytes, 0);
}

#[test]
fn should_remember_the_largest_unaccounted_reading_of_the_request() {
    let mut observer = observer();
    observer.observe_step(reading_step(8 * GIBIBYTE, 100 * MEBIBYTE));
    observer.observe_step(reading_step(8 * GIBIBYTE, 860 * MEBIBYTE));
    let reading = observer.observe_step(reading_step(8 * GIBIBYTE, 0));
    assert_eq!(reading.reconciliation.unaccounted_bytes, 0);
    assert_eq!(reading.largest_unaccounted_bytes, 860 * MEBIBYTE);
    assert_eq!(reading.observed_step_count, 3);
}

#[test]
fn should_start_the_worst_readings_again_with_each_request() {
    let mut observer = observer();
    observer.observe_step(reading_step(8 * GIBIBYTE, 860 * MEBIBYTE));
    observer.finish_request(0);
    let reading = observer.observe_step(reading_step(8 * GIBIBYTE, 0));
    assert_eq!(reading.largest_unaccounted_bytes, 0);
    assert_eq!(reading.observed_step_count, 1);
}

#[test]
fn should_report_how_empty_the_kept_weights_were_while_reading() {
    let mut observer = observer();
    let nearly_empty = observer.observe_step(reading_step(GIBIBYTE, 0));
    let capacity_bytes = nearly_empty.accounts.kept_weight_capacity_bytes;
    let expected_permille =
        u32::try_from(GIBIBYTE * 1_000 / capacity_bytes).expect("a share in thousandths fits u32");
    assert_eq!(
        nearly_empty.lowest_kept_share_while_reading_permille,
        Some(expected_permille)
    );
    let fuller = observer.observe_step(reading_step(8 * GIBIBYTE, 0));
    assert_eq!(
        fuller.lowest_kept_share_while_reading_permille,
        Some(expected_permille),
        "the lowest share of the request is what is reported"
    );
}

#[test]
fn should_report_weights_kept_beyond_what_the_ledger_would_allow() {
    let mut observer = observer();
    let reading = observer.observe_step(reading_step(12 * GIBIBYTE, 0));
    assert!(reading.reconciliation.kept_over_capacity_bytes > 0);
    assert_eq!(
        reading.largest_kept_over_capacity_bytes,
        reading.reconciliation.kept_over_capacity_bytes
    );
    assert_eq!(
        reading.lowest_kept_share_while_reading_permille,
        Some(1_000)
    );
}

#[test]
fn should_follow_a_limit_that_moved_between_steps() {
    let mut observer = observer();
    let at_16 = observer.observe_step(reading_step(8 * GIBIBYTE, 0));
    let mut step_at_12 = reading_step(8 * GIBIBYTE, 0);
    step_at_12.limit_bytes = 12 * GIBIBYTE;
    let at_12 = observer.observe_step(step_at_12);
    assert_eq!(at_12.accounts.limit_bytes, 12 * GIBIBYTE);
    assert!(at_12.accounts.kept_weight_capacity_bytes < at_16.accounts.kept_weight_capacity_bytes);
}

#[test]
fn should_keep_its_limit_when_a_step_reports_a_zero_limit() {
    let mut observer = observer();
    let mut step_with_zero_limit = reading_step(8 * GIBIBYTE, 0);
    step_with_zero_limit.limit_bytes = 0;
    let reading = observer.observe_step(step_with_zero_limit);
    assert_eq!(reading.accounts.limit_bytes, LIMIT_BYTES);
}

#[test]
fn should_grow_the_context_account_as_the_request_grows() {
    let mut observer = observer();
    let first = observer.observe_step(reading_step(8 * GIBIBYTE, 0));
    let mut later_step = reading_step(8 * GIBIBYTE, 0);
    later_step.phase = LedgerPhase::WritingReply;
    later_step.request_context_bytes = GIBIBYTE;
    let later = observer.observe_step(later_step);
    assert_eq!(first.accounts.context_bytes, GIBIBYTE / 2);
    assert_eq!(later.accounts.context_bytes, GIBIBYTE);
    assert_eq!(
        later.lowest_kept_share_while_reading_permille,
        first.lowest_kept_share_while_reading_permille,
        "a reply step does not change the reading-phase share"
    );
}

#[test]
fn should_keep_the_largest_step_peak_of_each_phase_apart() {
    let mut observer = observer();
    observer.observe_step(reading_step(8 * GIBIBYTE, 0));
    let mut reply_step = reading_step(8 * GIBIBYTE, 0);
    reply_step.phase = LedgerPhase::WritingReply;
    reply_step.peak_bytes_during_step = reply_step.active_bytes_after_step + 200 * MEBIBYTE;
    let reading = observer.observe_step(reply_step);
    assert_eq!(
        reading.largest_reading_step_peak_beyond_held_bytes,
        3 * GIBIBYTE
    );
    assert_eq!(
        reading.largest_writing_step_peak_beyond_held_bytes,
        200 * MEBIBYTE
    );
}

#[test]
fn should_express_the_kept_share_in_thousandths() {
    assert_eq!(kept_share_permille(0, 1_000), Some(0));
    assert_eq!(kept_share_permille(899, 1_000), Some(899));
    assert_eq!(kept_share_permille(1_000, 1_000), Some(1_000));
    assert_eq!(kept_share_permille(5_000, 1_000), Some(1_000));
    assert_eq!(kept_share_permille(5, 0), None);
    assert_eq!(kept_share_permille(u64::MAX, u64::MAX), Some(1_000));
}
