use ipc_protocol::RequestId;
use serial_test::serial;

use crate::PerformanceAttribution;
use crate::prism_hadamard_qwen35::engine::fixture_engine_test_support::loaded_fixture_engine;

// Positive and edge: with no store open, a request starts from position zero
// with one unallocated decoder-state owner per fixture decoder layer.
#[test]
#[serial]
fn should_start_a_request_from_position_zero_when_no_store_is_open() {
    let mut fixture_engine = loaded_fixture_engine(|_| {});

    let starting_decoder_state = fixture_engine
        .execution
        .starting_request_decoder_state(
            RequestId::new(51),
            &[1, 2, 3],
            8,
            &mut PerformanceAttribution::disabled(),
        )
        .expect("a request without a store should start");

    assert!(starting_decoder_state.restore_outcome.is_none());
    let request_decoder_state = starting_decoder_state.request_decoder_state;
    let decoder_layer_count = fixture_engine
        .execution
        .execution_core
        .loaded_model()
        .expect("the fixture model is loaded")
        .weights
        .decoder_layers
        .len();
    assert_eq!(request_decoder_state.layer_count(), decoder_layer_count);
    assert_eq!(request_decoder_state.payload_byte_count(), 0);
}
