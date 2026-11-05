use std::cell::{Cell, RefCell};

use super::{NeuralEngineMlpPrefill, neural_engine_tokens_for_chunk};

fn prefill_for_chunk(chunk_token_count: u32) -> NeuralEngineMlpPrefill {
    NeuralEngineMlpPrefill {
        layer_programs: RefCell::new(Vec::new()),
        chunk_token_count,
        neural_engine_token_count: 896,
        has_failed: Cell::new(false),
        is_paused: Cell::new(false),
    }
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
fn should_stop_handling_chunks_after_a_failure() {
    let prefill = prefill_for_chunk(2048);

    prefill.has_failed.set(true);

    assert!(!prefill.handles(2048));
}
