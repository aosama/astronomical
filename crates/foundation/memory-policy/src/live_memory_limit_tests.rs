use crate::live_memory_limit::{
    InFlightRequestMemory, LiveMemoryLimitRefusal, live_ceiling_refusal, needs_at_least_message,
    request_aware_ceiling_floor_bytes,
};

#[test]
fn should_accept_a_ceiling_above_the_held_memory() {
    assert_eq!(live_ceiling_refusal(8_000, 4_000), None);
}

#[test]
fn should_accept_a_ceiling_equal_to_the_held_memory() {
    assert_eq!(live_ceiling_refusal(4_000, 4_000), None);
}

#[test]
fn should_refuse_a_ceiling_one_byte_below_the_held_memory_and_name_the_floor() {
    assert_eq!(
        live_ceiling_refusal(3_999, 4_000),
        Some(LiveMemoryLimitRefusal::BelowModelFloor {
            minimum_mlx_memory_ceiling_bytes: 4_000
        })
    );
}

#[test]
fn should_accept_any_ceiling_when_the_model_holds_nothing() {
    assert_eq!(live_ceiling_refusal(0, 0), None);
}

const GIBIBYTE: u64 = 1 << 30;

fn in_flight(non_evictable_gib: u64, working_buffer_gib: u64) -> Option<InFlightRequestMemory> {
    Some(InFlightRequestMemory {
        non_evictable_active_bytes: non_evictable_gib * GIBIBYTE,
        next_step_working_buffer_bytes: working_buffer_gib * GIBIBYTE,
    })
}

#[test]
fn should_keep_the_idle_floor_exactly_when_no_request_runs() {
    assert_eq!(
        request_aware_ceiling_floor_bytes(5 * GIBIBYTE, None),
        5 * GIBIBYTE
    );
}

#[test]
fn should_raise_the_floor_to_what_a_running_request_cannot_give_back_plus_its_next_step() {
    assert_eq!(
        request_aware_ceiling_floor_bytes(5 * GIBIBYTE, in_flight(7, 1)),
        8 * GIBIBYTE
    );
}

#[test]
fn should_keep_the_idle_floor_when_a_running_request_needs_less() {
    assert_eq!(
        request_aware_ceiling_floor_bytes(5 * GIBIBYTE, in_flight(3, 1)),
        5 * GIBIBYTE
    );
}

#[test]
fn should_grow_the_floor_with_the_context_a_request_holds() {
    let early_floor = request_aware_ceiling_floor_bytes(GIBIBYTE, in_flight(6, 1));
    let later_floor = request_aware_ceiling_floor_bytes(GIBIBYTE, in_flight(7, 1));
    assert!(later_floor > early_floor);
}

#[test]
fn should_saturate_rather_than_wrap_on_impossible_readings() {
    assert_eq!(
        request_aware_ceiling_floor_bytes(
            0,
            Some(InFlightRequestMemory {
                non_evictable_active_bytes: u64::MAX,
                next_step_working_buffer_bytes: 1,
            })
        ),
        u64::MAX
    );
}

#[test]
fn should_name_the_needed_ceiling_in_gigabytes_of_two_to_the_thirty_bytes() {
    assert_eq!(
        needs_at_least_message(22 * GIBIBYTE + GIBIBYTE / 2),
        "needs at least 22.5 GB"
    );
}

#[test]
fn should_name_a_zero_floor_plainly() {
    assert_eq!(needs_at_least_message(0), "needs at least 0.0 GB");
}
