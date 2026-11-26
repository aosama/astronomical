//! Unit contracts for the attention-boundary decision of a decoder layer. The
//! layer's graph construction and the evaluation it can force need live GPU
//! work and are covered by the engine's forward tests.

use super::should_isolate_prefill_attention_wait;

#[test]
fn should_isolate_the_attention_wait_for_a_multi_token_prefill_step_under_attribution() {
    assert!(should_isolate_prefill_attention_wait(true, 512));
}

#[test]
fn should_isolate_the_attention_wait_at_the_smallest_multi_token_step() {
    assert!(should_isolate_prefill_attention_wait(true, 2));
}

#[test]
fn should_leave_a_one_token_decode_step_unserialized() {
    assert!(!should_isolate_prefill_attention_wait(true, 1));
}

#[test]
fn should_never_isolate_the_attention_wait_while_attribution_is_off() {
    assert!(!should_isolate_prefill_attention_wait(false, 512));
}

#[test]
fn should_not_isolate_the_attention_wait_for_an_empty_step() {
    assert!(!should_isolate_prefill_attention_wait(true, 0));
}
