use std::cell::{Cell, RefCell};

use super::{NeuralEngineOutputGatePrefill, chunk_fits_surface_rows};
use crate::qwen3_5_core::model::output_gate_engine::LinearAttentionOutputGateEngine;

fn prefill_for_chunk(chunk_token_count: u32) -> NeuralEngineOutputGatePrefill {
    NeuralEngineOutputGatePrefill {
        layer_programs: RefCell::new(Vec::new()),
        chunk_token_count,
        has_failed: Cell::new(false),
        is_paused: Cell::new(false),
    }
}

#[test]
fn should_handle_only_exactly_the_configured_chunk() {
    let prefill = prefill_for_chunk(2048);

    assert!(prefill.handles(2048));
    assert!(!prefill.handles(2047));
    assert!(!prefill.handles(1));
    assert!(!prefill.handles(-2048));
}

#[test]
fn should_stop_handling_chunks_after_a_failure() {
    let prefill = prefill_for_chunk(2048);

    prefill.has_failed.set(true);

    assert!(!prefill.handles(2048));
}

#[test]
fn should_stop_handling_chunks_only_while_paused() {
    let prefill = prefill_for_chunk(2048);

    prefill.set_paused(true);
    assert!(!prefill.handles(2048));

    prefill.set_paused(false);
    assert!(prefill.handles(2048));
}

#[test]
fn should_accept_a_chunk_that_fills_its_surface_rows() {
    assert!(chunk_fits_surface_rows(2048));
    assert!(chunk_fits_surface_rows(64));
}

#[test]
fn should_refuse_a_chunk_that_would_pad_its_surface_rows() {
    assert!(!chunk_fits_surface_rows(2047));
    assert!(!chunk_fits_surface_rows(63));
    assert!(!chunk_fits_surface_rows(0));
}
