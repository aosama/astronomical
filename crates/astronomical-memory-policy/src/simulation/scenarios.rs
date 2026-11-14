//! The incidents, replayed against the ledger.
//!
//! Each test is one real memory incident, or one promise of the redesign,
//! run on a model sized like the 35B: about 2.5 GiB of core weights and
//! forty half-gibibyte layers under a 12 GiB limit.

use super::simulated_engine::{EngineJournal, KeptLayerOrder, SimulatedEngine};
use super::simulated_model::SimulatedModel;
use crate::ledger::LeasePurpose;

const KIBIBYTE: u64 = 1 << 10;
const MEBIBYTE: u64 = 1 << 20;
const GIBIBYTE: u64 = 1 << 30;
const LAYER_WEIGHT_BYTES: u64 = 512 * MEBIBYTE;
const LAYER_COUNT: usize = 40;
const LIMIT_BYTES: u64 = 12 * GIBIBYTE;
const SPAN_TOKEN_COUNT: u32 = 4_096;
const SMALL_PROMPT_TOKEN_COUNT: u32 = 461;
const LARGE_PROMPT_TOKEN_COUNT: u32 = 10_240;
const REPLY_TOKEN_COUNT: u32 = 500;

/// One 4,096-token span through one layer truly needs 1 GiB.
fn model_like_the_35b() -> SimulatedModel {
    SimulatedModel {
        core_weight_bytes: 2_560 * MEBIBYTE,
        layer_weight_bytes: vec![LAYER_WEIGHT_BYTES; LAYER_COUNT],
        context_bytes_per_token: 20 * KIBIBYTE,
        reading_stream_slot_count: 2,
        writing_reply_stream_slot_bytes: 16 * MEBIBYTE,
        true_reading_working_memory_bytes_per_span_token: 256 * KIBIBYTE,
        true_writing_working_memory_bytes: 64 * MEBIBYTE,
    }
}

/// The same model with layers that need 2 GiB per span, so a measurement
/// misjudged by half is too large for the safety margin to absorb.
fn model_with_heavy_working_memory() -> SimulatedModel {
    SimulatedModel {
        true_reading_working_memory_bytes_per_span_token: 512 * KIBIBYTE,
        ..model_like_the_35b()
    }
}

fn probed_engine(model: SimulatedModel, limit_bytes: u64) -> SimulatedEngine {
    let mut engine = SimulatedEngine::load(model, limit_bytes);
    engine.probe_working_memory(SPAN_TOKEN_COUNT, 100);
    engine
}

fn allowed_active_bytes(limit_bytes: u64) -> u64 {
    limit_bytes + limit_bytes / 100
}

/// What every healthy run must show, whatever the scenario.
fn assert_healthy(journal: EngineJournal, limit_bytes: u64) {
    assert!(!journal.ran_out_of_memory, "the request must complete");
    assert_eq!(journal.overflow_count, 0, "no step may hit the limit");
    assert_eq!(
        journal.largest_unaccounted_bytes, 0,
        "the ledger must account for everything in memory"
    );
    assert!(
        journal.peak_active_bytes <= allowed_active_bytes(limit_bytes),
        "the peak must stay within what MLX allows"
    );
}

#[test]
fn should_use_the_capacity_while_reading_the_very_first_prompt() {
    let mut engine = probed_engine(model_like_the_35b(), LIMIT_BYTES);
    engine.start_request(LARGE_PROMPT_TOKEN_COUNT);
    engine.read_prompt(LARGE_PROMPT_TOKEN_COUNT, SPAN_TOKEN_COUNT);
    let capacity_bytes = engine.accounts().kept_weight_capacity_bytes;
    assert!(
        engine.kept_weight_bytes() * 10 >= capacity_bytes * 9,
        "the first prompt must leave at least 90% of the capacity kept: kept {} of {capacity_bytes}",
        engine.kept_weight_bytes()
    );
    engine.write_reply(REPLY_TOKEN_COUNT);
    engine.finish_request();
    assert_healthy(engine.journal(), LIMIT_BYTES);
}

/// Issue #2217: a small request, then a large one, used to overflow and
/// discard every kept expert.
#[test]
fn should_serve_a_first_large_prompt_after_a_small_one_without_overflow() {
    let mut engine = probed_engine(model_like_the_35b(), LIMIT_BYTES);
    engine.serve(
        SMALL_PROMPT_TOKEN_COUNT,
        REPLY_TOKEN_COUNT,
        SPAN_TOKEN_COUNT,
    );
    let kept_after_small_request_bytes = engine.kept_weight_bytes();
    engine.start_request(LARGE_PROMPT_TOKEN_COUNT);
    engine.read_prompt(LARGE_PROMPT_TOKEN_COUNT, SPAN_TOKEN_COUNT);
    let capacity_bytes = engine.accounts().kept_weight_capacity_bytes;
    assert!(engine.kept_weight_bytes() * 10 >= capacity_bytes * 9);
    assert!(
        engine.kept_weight_bytes() * 2 > kept_after_small_request_bytes,
        "most of what the small request left kept must survive the large prompt"
    );
    engine.write_reply(REPLY_TOKEN_COUNT);
    engine.finish_request();
    assert_healthy(engine.journal(), LIMIT_BYTES);
}

/// The forced misjudgment: the probe reports half the true working memory,
/// and the first step of a long prompt streams with the cache full.
#[test]
fn should_redo_one_layer_step_once_when_working_memory_was_misjudged_by_half() {
    let mut engine = SimulatedEngine::load_keeping(
        model_with_heavy_working_memory(),
        LIMIT_BYTES,
        KeptLayerOrder::HighestLayersFirst,
    );
    engine.probe_working_memory(SPAN_TOKEN_COUNT, 50);
    engine.serve(
        SMALL_PROMPT_TOKEN_COUNT,
        REPLY_TOKEN_COUNT,
        SPAN_TOKEN_COUNT,
    );
    engine.serve(
        LARGE_PROMPT_TOKEN_COUNT,
        REPLY_TOKEN_COUNT,
        SPAN_TOKEN_COUNT,
    );
    let journal = engine.journal();
    assert!(!journal.ran_out_of_memory, "the request must complete");
    assert_eq!(journal.overflow_count, 1, "exactly one step hits the limit");
    assert_eq!(journal.redone_layer_step_count, 1, "and it is redone once");
    assert!(journal.kept_weight_bytes_at_last_overflow > 0);
    assert!(
        (journal.kept_weight_bytes_at_last_overflow - journal.evicted_for_redo_bytes) * 10
            >= journal.kept_weight_bytes_at_last_overflow * 8,
        "at least 80% of the kept weights must survive the redo: {} of {} given back",
        journal.evicted_for_redo_bytes,
        journal.kept_weight_bytes_at_last_overflow
    );
    assert_eq!(journal.largest_unaccounted_bytes, 0);
}

/// Once the redone step has been measured, the same prompt again fits.
#[test]
fn should_not_overflow_again_once_the_true_working_memory_is_known() {
    let mut engine = SimulatedEngine::load_keeping(
        model_with_heavy_working_memory(),
        LIMIT_BYTES,
        KeptLayerOrder::HighestLayersFirst,
    );
    engine.probe_working_memory(SPAN_TOKEN_COUNT, 50);
    engine.serve(
        SMALL_PROMPT_TOKEN_COUNT,
        REPLY_TOKEN_COUNT,
        SPAN_TOKEN_COUNT,
    );
    engine.serve(
        LARGE_PROMPT_TOKEN_COUNT,
        REPLY_TOKEN_COUNT,
        SPAN_TOKEN_COUNT,
    );
    let overflow_count_after_first_large_prompt = engine.journal().overflow_count;
    engine.serve(
        LARGE_PROMPT_TOKEN_COUNT,
        REPLY_TOKEN_COUNT,
        SPAN_TOKEN_COUNT,
    );
    assert_eq!(
        engine.journal().overflow_count,
        overflow_count_after_first_large_prompt
    );
}

/// Issue #1108: lowering the limit mid-request must only give back weights.
#[test]
fn should_only_trim_when_the_limit_is_lowered_in_the_middle_of_a_request() {
    let lowered_limit_bytes = 10 * GIBIBYTE;
    let mut engine = probed_engine(model_like_the_35b(), LIMIT_BYTES);
    engine.start_request(LARGE_PROMPT_TOKEN_COUNT);
    engine.read_prompt(LARGE_PROMPT_TOKEN_COUNT, SPAN_TOKEN_COUNT);
    let journal_before_change = engine.journal();
    engine.change_limit(lowered_limit_bytes);
    let evicted_by_the_change_bytes =
        engine.journal().evicted_weight_bytes - journal_before_change.evicted_weight_bytes;
    assert!(
        evicted_by_the_change_bytes <= LIMIT_BYTES - lowered_limit_bytes + LAYER_WEIGHT_BYTES,
        "lowering the limit by 2 GiB must give back about 2 GiB: {evicted_by_the_change_bytes}"
    );
    engine.write_reply(REPLY_TOKEN_COUNT);
    engine.finish_request();
    assert_healthy(engine.journal(), LIMIT_BYTES);
}

#[test]
fn should_start_the_next_request_cleanly_after_a_cancelled_one() {
    let mut engine = probed_engine(model_like_the_35b(), LIMIT_BYTES);
    engine.start_request(LARGE_PROMPT_TOKEN_COUNT);
    engine.read_prompt(LARGE_PROMPT_TOKEN_COUNT, SPAN_TOKEN_COUNT);
    engine.write_reply(10);
    // Cancelled: the request ends with the cache as full as writing allows.
    engine.finish_request();
    engine.serve(
        LARGE_PROMPT_TOKEN_COUNT,
        REPLY_TOKEN_COUNT,
        SPAN_TOKEN_COUNT,
    );
    assert_healthy(engine.journal(), LIMIT_BYTES);
}

/// The October seat thrash: the same request twice must not give back and
/// re-read the same layers inside a phase.
#[test]
fn should_give_weights_back_once_between_two_identical_requests() {
    let mut engine = probed_engine(model_like_the_35b(), LIMIT_BYTES);
    engine.serve(
        LARGE_PROMPT_TOKEN_COUNT,
        REPLY_TOKEN_COUNT,
        SPAN_TOKEN_COUNT,
    );
    let journal_before_second_request = engine.journal();
    let kept_for_writing_bytes = engine.kept_weight_bytes();
    engine.start_request(LARGE_PROMPT_TOKEN_COUNT);
    engine.read_prompt(LARGE_PROMPT_TOKEN_COUNT, SPAN_TOKEN_COUNT);
    let journal_after_reading = engine.journal();
    let kept_for_reading_bytes = engine.kept_weight_bytes();
    assert_eq!(
        journal_after_reading.eviction_event_count
            - journal_before_second_request.eviction_event_count,
        1,
        "reading the same prompt again gives weights back exactly once"
    );
    assert_eq!(
        journal_after_reading.evicted_weight_bytes
            - journal_before_second_request.evicted_weight_bytes,
        kept_for_writing_bytes - kept_for_reading_bytes,
        "and only the difference between the two phases"
    );
    let unkept_layer_count = LAYER_COUNT as u64 - kept_for_reading_bytes / LAYER_WEIGHT_BYTES;
    assert_eq!(
        journal_after_reading.storage_read_bytes - journal_before_second_request.storage_read_bytes,
        unkept_layer_count * LAYER_WEIGHT_BYTES,
        "each layer that is not kept is read once for the whole prompt"
    );
    engine.write_reply(REPLY_TOKEN_COUNT);
    engine.finish_request();
    assert_healthy(engine.journal(), LIMIT_BYTES);
}

#[test]
fn should_keep_every_weight_and_read_nothing_once_the_model_fits() {
    let roomy_limit_bytes = 32 * GIBIBYTE;
    let mut engine = probed_engine(model_like_the_35b(), roomy_limit_bytes);
    engine.serve(
        LARGE_PROMPT_TOKEN_COUNT,
        REPLY_TOKEN_COUNT,
        SPAN_TOKEN_COUNT,
    );
    assert_eq!(
        engine.kept_weight_bytes(),
        LAYER_COUNT as u64 * LAYER_WEIGHT_BYTES
    );
    let storage_read_after_first_request_bytes = engine.journal().storage_read_bytes;
    engine.start_request(LARGE_PROMPT_TOKEN_COUNT);
    engine.read_prompt(LARGE_PROMPT_TOKEN_COUNT, SPAN_TOKEN_COUNT);
    assert_eq!(engine.accounts().stream_slot_bytes, 0);
    engine.write_reply(REPLY_TOKEN_COUNT);
    engine.finish_request();
    assert_eq!(
        engine.journal().storage_read_bytes,
        storage_read_after_first_request_bytes,
        "nothing is read once everything is kept"
    );
    assert_eq!(engine.journal().eviction_event_count, 0);
    assert_healthy(engine.journal(), roomy_limit_bytes);
}

/// Issue #1902: the ledger never refuses a model because the limit is
/// small. It reports the shortfall, keeps nothing, and MLX decides.
#[test]
fn should_report_a_shortfall_and_refuse_nothing_below_the_streaming_floor() {
    let tiny_limit_bytes = 3 * GIBIBYTE;
    let mut engine = probed_engine(model_like_the_35b(), tiny_limit_bytes);
    engine.start_request(LARGE_PROMPT_TOKEN_COUNT);
    engine.read_prompt(LARGE_PROMPT_TOKEN_COUNT, SPAN_TOKEN_COUNT);
    let accounts = engine.accounts();
    assert!(accounts.shortfall_bytes > 0);
    assert_eq!(accounts.kept_weight_capacity_bytes, 0);
    assert_eq!(engine.kept_weight_bytes(), 0);
    assert!(
        engine.journal().ran_out_of_memory,
        "a step this limit cannot hold fails in MLX, not in the ledger"
    );
}

#[test]
fn should_make_room_for_an_image_and_take_the_room_back() {
    let vision_tower_bytes = GIBIBYTE;
    let mut engine = probed_engine(model_like_the_35b(), LIMIT_BYTES);
    engine.serve(
        LARGE_PROMPT_TOKEN_COUNT,
        REPLY_TOKEN_COUNT,
        SPAN_TOKEN_COUNT,
    );
    engine.start_request(LARGE_PROMPT_TOKEN_COUNT);
    let vision_lease = engine.grant_lease(LeasePurpose::VisionTower, vision_tower_bytes);
    engine.read_prompt(LARGE_PROMPT_TOKEN_COUNT, SPAN_TOKEN_COUNT);
    let capacity_with_image_bytes = engine.accounts().kept_weight_capacity_bytes;
    engine.give_back_lease(vision_lease, vision_tower_bytes);
    assert_eq!(
        engine.accounts().kept_weight_capacity_bytes - capacity_with_image_bytes,
        vision_tower_bytes
    );
    engine.write_reply(REPLY_TOKEN_COUNT);
    engine.finish_request();
    assert_healthy(engine.journal(), LIMIT_BYTES);
}

#[test]
fn should_give_back_a_layer_at_a_time_as_a_long_reply_grows_the_context() {
    let long_reply_token_count = 65_536;
    let mut engine = probed_engine(model_like_the_35b(), LIMIT_BYTES);
    engine.serve(
        LARGE_PROMPT_TOKEN_COUNT,
        long_reply_token_count,
        SPAN_TOKEN_COUNT,
    );
    let journal = engine.journal();
    assert!(
        journal.eviction_event_count > 1,
        "a reply this long must give weights back more than once"
    );
    assert_eq!(
        journal.largest_single_eviction_bytes, LAYER_WEIGHT_BYTES,
        "each give-back is one layer, never the cache"
    );
    assert_healthy(journal, LIMIT_BYTES);
}

/// What reconciliation is for: memory nobody declared shows up as a number.
#[test]
fn should_expose_an_allocation_that_nobody_declared() {
    let undeclared_bytes = 300 * MEBIBYTE;
    let mut engine = probed_engine(model_like_the_35b(), LIMIT_BYTES);
    engine.hold_undeclared_bytes(undeclared_bytes);
    engine.serve(
        LARGE_PROMPT_TOKEN_COUNT,
        REPLY_TOKEN_COUNT,
        SPAN_TOKEN_COUNT,
    );
    let journal = engine.journal();
    assert_eq!(journal.largest_unaccounted_bytes, undeclared_bytes);
    assert!(
        !journal.ran_out_of_memory,
        "the margin absorbs it while the ledger reports it"
    );
}
