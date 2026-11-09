use mlx_c_rust::MlxDtype;

use super::{
    GatedDeltaSequenceInputShapes, GatedDeltaSequenceShape, sequence_shape_from_input_shapes,
    validate_sequence_dtypes,
};

/// Qwen3.5 linear-attention geometry: 16 key heads, 32 value heads, key
/// dimension 128, value dimension 128, one batch row of 512 prompt tokens.
struct SequenceShapes {
    query_shape: Vec<i32>,
    key_shape: Vec<i32>,
    value_shape: Vec<i32>,
    decay_shape: Vec<i32>,
    update_rate_shape: Vec<i32>,
    recurrent_state_shape: Vec<i32>,
}

impl SequenceShapes {
    fn qwen3_5_prompt_chunk() -> Self {
        Self {
            query_shape: vec![1, 512, 16, 128],
            key_shape: vec![1, 512, 16, 128],
            value_shape: vec![1, 512, 32, 128],
            decay_shape: vec![1, 512, 32],
            update_rate_shape: vec![1, 512, 32],
            recurrent_state_shape: vec![1, 32, 128, 128],
        }
    }

    fn validate(&self) -> Result<GatedDeltaSequenceShape, String> {
        sequence_shape_from_input_shapes(&GatedDeltaSequenceInputShapes {
            query_shape: &self.query_shape,
            key_shape: &self.key_shape,
            value_shape: &self.value_shape,
            decay_shape: &self.decay_shape,
            update_rate_shape: &self.update_rate_shape,
            recurrent_state_shape: &self.recurrent_state_shape,
        })
        .map_err(|runtime_error| runtime_error.to_string())
    }
}

fn refusal_of(sequence_shapes: &SequenceShapes) -> String {
    sequence_shapes
        .validate()
        .expect_err("the shapes must be refused")
}

#[test]
fn should_derive_the_kernel_geometry_from_queries_and_values() {
    let sequence_shape = SequenceShapes::qwen3_5_prompt_chunk()
        .validate()
        .expect("the Qwen3.5 geometry is supported");
    assert_eq!(
        sequence_shape,
        GatedDeltaSequenceShape {
            batch_size: 1,
            token_count: 512,
            key_head_count: 16,
            value_head_count: 32,
            key_head_dimension: 128,
            value_head_dimension: 128,
        }
    );
}

#[test]
fn should_accept_a_single_token_and_the_smallest_value_block() {
    let mut single_token = SequenceShapes::qwen3_5_prompt_chunk();
    single_token.query_shape = vec![1, 1, 16, 128];
    single_token.key_shape = vec![1, 1, 16, 128];
    single_token.value_shape = vec![1, 1, 32, 16];
    single_token.decay_shape = vec![1, 1, 32];
    single_token.update_rate_shape = vec![1, 1, 32];
    single_token.recurrent_state_shape = vec![1, 32, 16, 128];
    assert_eq!(single_token.validate().expect("one token").token_count, 1);
}

#[test]
fn should_refuse_an_input_of_the_wrong_rank() {
    let mut flattened_decays = SequenceShapes::qwen3_5_prompt_chunk();
    flattened_decays.decay_shape = vec![512, 32];
    assert!(refusal_of(&flattened_decays).contains("invalid ranks"));
}

#[test]
fn should_refuse_keys_that_differ_from_queries() {
    let mut mismatched_keys = SequenceShapes::qwen3_5_prompt_chunk();
    mismatched_keys.key_shape = vec![1, 511, 16, 128];
    assert!(refusal_of(&mismatched_keys).contains("queries and keys"));
}

#[test]
fn should_name_each_broken_kernel_limit_on_its_own() {
    let mut empty_prompt = SequenceShapes::qwen3_5_prompt_chunk();
    empty_prompt.query_shape[1] = 0;
    empty_prompt.key_shape[1] = 0;
    assert!(refusal_of(&empty_prompt).contains("must be positive"));

    let mut uneven_heads = SequenceShapes::qwen3_5_prompt_chunk();
    uneven_heads.value_shape[2] = 24;
    assert!(refusal_of(&uneven_heads).contains("multiple of key heads"));

    let mut narrow_keys = SequenceShapes::qwen3_5_prompt_chunk();
    narrow_keys.query_shape[3] = 64;
    narrow_keys.key_shape[3] = 64;
    assert!(refusal_of(&narrow_keys).contains("key dimension must be 128"));

    let mut unblocked_values = SequenceShapes::qwen3_5_prompt_chunk();
    unblocked_values.value_shape[3] = 120;
    assert!(refusal_of(&unblocked_values).contains("divide by 16"));
}

#[test]
fn should_refuse_companion_shapes_that_disagree_with_the_geometry() {
    let mut short_values = SequenceShapes::qwen3_5_prompt_chunk();
    short_values.value_shape[1] = 511;
    let mut transposed_state = SequenceShapes::qwen3_5_prompt_chunk();
    transposed_state.recurrent_state_shape = vec![1, 32, 128, 64];
    let mut per_key_head_rates = SequenceShapes::qwen3_5_prompt_chunk();
    per_key_head_rates.update_rate_shape = vec![1, 512, 16];
    for incompatible_shapes in [short_values, transposed_state, per_key_head_rates] {
        assert!(refusal_of(&incompatible_shapes).contains("incompatible"));
    }
}

#[test]
fn should_accept_any_supported_activation_dtype_with_a_float32_state() {
    for activation_dtype in [MlxDtype::Float16, MlxDtype::BFloat16, MlxDtype::Float32] {
        assert!(validate_sequence_dtypes(MlxDtype::Float32, [activation_dtype; 5]).is_ok());
    }
}

#[test]
fn should_refuse_a_narrow_state_or_an_unsupported_activation_dtype() {
    let narrow_state = validate_sequence_dtypes(MlxDtype::BFloat16, [MlxDtype::BFloat16; 5])
        .expect_err("a bfloat16 state must be refused");
    assert!(
        narrow_state
            .to_string()
            .contains("recurrent state must use float32")
    );
    let mut activation_dtypes = [MlxDtype::BFloat16; 5];
    activation_dtypes[4] = MlxDtype::Int32;
    let integer_rates = validate_sequence_dtypes(MlxDtype::Float32, activation_dtypes)
        .expect_err("integer update rates must be refused");
    assert!(
        integer_rates
            .to_string()
            .contains("float16, bfloat16, or float32")
    );
}
