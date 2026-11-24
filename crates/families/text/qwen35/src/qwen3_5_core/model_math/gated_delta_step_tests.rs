use mlx_bridge::MlxDtype;

use super::{
    GatedDeltaStepInputShapes, repeat_factor_from_step_input_shapes, validate_step_dtypes,
};

/// The six step input shapes, owned so a test can break one before validating.
struct StepShapes {
    query_shape: Vec<i32>,
    key_shape: Vec<i32>,
    value_shape: Vec<i32>,
    decay_shape: Vec<i32>,
    update_rate_shape: Vec<i32>,
    recurrent_state_shape: Vec<i32>,
}

impl StepShapes {
    /// One Qwen3.5 decode token: 16 key heads, 32 value heads, both of
    /// dimension 128, one batch row.
    fn qwen3_5_decode_token() -> Self {
        Self {
            query_shape: vec![1, 16, 128],
            key_shape: vec![1, 16, 128],
            value_shape: vec![1, 32, 128],
            decay_shape: vec![1, 32],
            update_rate_shape: vec![1, 32],
            recurrent_state_shape: vec![1, 32, 128, 128],
        }
    }

    fn repeat_factor(&self) -> Result<i32, String> {
        repeat_factor_from_step_input_shapes(&GatedDeltaStepInputShapes {
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

fn refusal_of(step_shapes: &StepShapes) -> String {
    step_shapes
        .repeat_factor()
        .expect_err("the shapes must be refused")
}

#[test]
fn should_repeat_each_key_head_across_its_value_heads() {
    assert_eq!(StepShapes::qwen3_5_decode_token().repeat_factor(), Ok(2));
}

#[test]
fn should_repeat_once_when_key_and_value_heads_match() {
    let mut equal_heads = StepShapes::qwen3_5_decode_token();
    equal_heads.value_shape = vec![1, 16, 128];
    equal_heads.decay_shape = vec![1, 16];
    equal_heads.update_rate_shape = vec![1, 16];
    equal_heads.recurrent_state_shape = vec![1, 16, 128, 128];
    assert_eq!(equal_heads.repeat_factor(), Ok(1));
}

#[test]
fn should_refuse_an_input_of_the_wrong_rank() {
    let mut token_axis_queries = StepShapes::qwen3_5_decode_token();
    token_axis_queries.query_shape = vec![1, 1, 16, 128];
    assert!(refusal_of(&token_axis_queries).contains("must have ranks"));
}

#[test]
fn should_refuse_keys_that_differ_from_queries() {
    let mut mismatched_keys = StepShapes::qwen3_5_decode_token();
    mismatched_keys.key_shape = vec![1, 16, 64];
    assert!(refusal_of(&mismatched_keys).contains("queries and keys must have identical shapes"));
}

#[test]
fn should_refuse_update_rates_that_differ_from_decays() {
    let mut per_key_head_rates = StepShapes::qwen3_5_decode_token();
    per_key_head_rates.update_rate_shape = vec![1, 16];
    assert!(
        refusal_of(&per_key_head_rates)
            .contains("update rates and decays must have identical shapes")
    );
}

#[test]
fn should_refuse_broken_head_geometry_with_one_combined_message() {
    let mut no_key_heads = StepShapes::qwen3_5_decode_token();
    no_key_heads.query_shape[1] = 0;
    no_key_heads.key_shape[1] = 0;
    let mut uneven_heads = StepShapes::qwen3_5_decode_token();
    uneven_heads.value_shape[1] = 24;
    let mut fewer_value_heads = StepShapes::qwen3_5_decode_token();
    fewer_value_heads.value_shape[1] = 8;
    for broken_geometry in [no_key_heads, uneven_heads, fewer_value_heads] {
        assert!(refusal_of(&broken_geometry).contains(
            "dimensions must be positive and value heads must divide evenly by key heads"
        ));
    }
}

#[test]
fn should_refuse_companion_shapes_that_disagree_with_the_geometry() {
    let mut transposed_state = StepShapes::qwen3_5_decode_token();
    transposed_state.recurrent_state_shape = vec![1, 32, 128, 64];
    let mut second_batch_values = StepShapes::qwen3_5_decode_token();
    second_batch_values.value_shape[0] = 2;
    let mut per_key_head_decays = StepShapes::qwen3_5_decode_token();
    per_key_head_decays.decay_shape = vec![1, 16];
    per_key_head_decays.update_rate_shape = vec![1, 16];
    for incompatible_shapes in [transposed_state, second_batch_values, per_key_head_decays] {
        assert!(refusal_of(&incompatible_shapes).contains("dimensions are incompatible"));
    }
}

#[test]
fn should_accept_supported_activation_dtypes_with_a_float32_state() {
    for activation_dtype in [MlxDtype::Float16, MlxDtype::BFloat16, MlxDtype::Float32] {
        assert!(validate_step_dtypes(MlxDtype::Float32, [activation_dtype; 5]).is_ok());
    }
}

#[test]
fn should_refuse_a_non_float32_state_before_an_integer_activation() {
    let narrow_state = validate_step_dtypes(MlxDtype::BFloat16, [MlxDtype::Int32; 5])
        .expect_err("a bfloat16 state must be refused");
    assert!(
        narrow_state
            .to_string()
            .contains("recurrent state must use float32")
    );
    let mut activation_dtypes = [MlxDtype::BFloat16; 5];
    activation_dtypes[2] = MlxDtype::Int32;
    let integer_values = validate_step_dtypes(MlxDtype::Float32, activation_dtypes)
        .expect_err("integer values must be refused");
    assert!(
        integer_values
            .to_string()
            .contains("float16, bfloat16, or float32")
    );
}
