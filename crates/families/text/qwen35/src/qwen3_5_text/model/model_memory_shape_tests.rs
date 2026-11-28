use super::{dense_model_memory_shape, sparse_model_memory_shape};
use crate::ModelMemoryGeometry;

fn sparse_geometry_for_test() -> ModelMemoryGeometry {
    ModelMemoryGeometry {
        model_core_payload_bytes: 2_600,
        complete_expert_payload_bytes: 34_000,
        largest_complete_expert_layer_bytes: 860,
        largest_routed_expert_page_bytes: 27,
        sequence_state_bytes_per_token: 20,
    }
}

#[test]
fn should_size_the_reply_slot_for_every_row_a_drafted_forward_routes() {
    let model_shape = sparse_model_memory_shape(sparse_geometry_for_test(), 4);
    assert_eq!(model_shape.writing_reply_stream_slot_bytes, 108);
    assert_eq!(model_shape.reading_prompt_stream_slot_bytes, 1_720);
}

#[test]
fn should_never_size_the_reply_slot_beyond_one_complete_layer() {
    let at_the_layer = sparse_model_memory_shape(sparse_geometry_for_test(), 32);
    let far_beyond_it = sparse_model_memory_shape(sparse_geometry_for_test(), u64::MAX);
    assert_eq!(at_the_layer.writing_reply_stream_slot_bytes, 860);
    assert_eq!(far_beyond_it.writing_reply_stream_slot_bytes, 860);
}

#[test]
fn should_describe_a_sparse_model_by_its_experts_and_two_layer_slots() {
    let model_shape = sparse_model_memory_shape(sparse_geometry_for_test(), 1);
    assert_eq!(model_shape.core_weight_bytes, 2_600);
    assert_eq!(model_shape.streamable_weight_bytes, 34_000);
    assert_eq!(model_shape.reading_prompt_stream_slot_bytes, 1_720);
    assert_eq!(model_shape.writing_reply_stream_slot_bytes, 27);
}

#[test]
fn should_describe_a_dense_model_by_its_decoder_layers_and_one_slot_size() {
    let model_shape = dense_model_memory_shape(900, &[150, 150, 160], 320);
    assert_eq!(model_shape.core_weight_bytes, 900);
    assert_eq!(model_shape.streamable_weight_bytes, 460);
    assert_eq!(model_shape.reading_prompt_stream_slot_bytes, 320);
    assert_eq!(model_shape.writing_reply_stream_slot_bytes, 320);
}

#[test]
fn should_describe_a_dense_model_with_no_streamed_layers_as_having_nothing_to_keep() {
    let model_shape = dense_model_memory_shape(900, &[], 0);
    assert_eq!(model_shape.streamable_weight_bytes, 0);
    assert_eq!(model_shape.reading_prompt_stream_slot_bytes, 0);
}

#[test]
fn should_saturate_instead_of_wrapping_on_absurd_layer_sizes() {
    let model_shape = dense_model_memory_shape(0, &[u64::MAX, 5], 0);
    assert_eq!(model_shape.streamable_weight_bytes, u64::MAX);
}
