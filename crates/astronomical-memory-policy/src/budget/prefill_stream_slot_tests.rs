//! Unit contracts for dropping the prefill streaming slot at full fit.

use super::prefill_stream_slot_bytes_after_full_fit;

const CEILING_BYTES: u64 = 40_000;
const EXPERT_PAYLOAD_BYTES: u64 = 30_000;
const STREAM_SLOT_BYTES: u64 = 1_700;

#[test]
fn should_not_reserve_stream_slots_for_layers_that_are_resident() {
    // 9,000 fixed bytes leave 31,000 for 30,000 of experts: every layer fits.
    assert_eq!(
        prefill_stream_slot_bytes_after_full_fit(
            CEILING_BYTES,
            9_000,
            EXPERT_PAYLOAD_BYTES,
            STREAM_SLOT_BYTES
        ),
        0
    );
}

#[test]
fn should_keep_the_stream_slot_when_one_layer_stays_cold() {
    // 10,001 fixed bytes leave 29,999: one byte short, so a layer streams.
    assert_eq!(
        prefill_stream_slot_bytes_after_full_fit(
            CEILING_BYTES,
            10_001,
            EXPERT_PAYLOAD_BYTES,
            STREAM_SLOT_BYTES
        ),
        STREAM_SLOT_BYTES
    );
}

#[test]
fn should_drop_the_stream_slot_exactly_at_the_full_fit_boundary() {
    assert_eq!(
        prefill_stream_slot_bytes_after_full_fit(
            CEILING_BYTES,
            10_000,
            EXPERT_PAYLOAD_BYTES,
            STREAM_SLOT_BYTES
        ),
        0
    );
}

#[test]
fn should_keep_the_stream_slot_for_a_model_without_expert_payload() {
    assert_eq!(
        prefill_stream_slot_bytes_after_full_fit(CEILING_BYTES, 9_000, 0, STREAM_SLOT_BYTES),
        STREAM_SLOT_BYTES
    );
}

#[test]
fn should_keep_the_stream_slot_when_fixed_bytes_exceed_the_ceiling() {
    assert_eq!(
        prefill_stream_slot_bytes_after_full_fit(
            CEILING_BYTES,
            CEILING_BYTES + 1,
            EXPERT_PAYLOAD_BYTES,
            STREAM_SLOT_BYTES
        ),
        STREAM_SLOT_BYTES
    );
}
