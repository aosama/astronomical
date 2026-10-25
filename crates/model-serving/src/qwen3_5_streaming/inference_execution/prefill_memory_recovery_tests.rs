use super::{
    MAXIMUM_PREFILL_MEMORY_RECOVERY_ATTEMPTS_PER_REQUEST, should_attempt_prefill_memory_recovery,
};

#[test]
fn should_attempt_recovery_for_the_first_text_prefill_memory_failure() {
    assert!(should_attempt_prefill_memory_recovery(0, false));
}

#[test]
fn should_not_attempt_recovery_once_the_request_used_its_recovery() {
    assert!(!should_attempt_prefill_memory_recovery(
        MAXIMUM_PREFILL_MEMORY_RECOVERY_ATTEMPTS_PER_REQUEST,
        false
    ));
}

#[test]
fn should_not_attempt_recovery_for_image_requests() {
    assert!(!should_attempt_prefill_memory_recovery(0, true));
}

#[test]
fn should_not_attempt_recovery_when_the_attempt_counter_is_saturated() {
    assert!(!should_attempt_prefill_memory_recovery(u8::MAX, false));
}
