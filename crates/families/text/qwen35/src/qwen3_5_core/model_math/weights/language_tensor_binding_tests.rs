//! Unit contracts for the pure decisions inside language tensor binding:
//! payload accounting, one-time binding of each name, and finding the
//! convolution weight that identifies a raw Hugging Face checkpoint. Reading
//! the shards and handing tensors to typed owners need a live MLX runtime and
//! real checkpoints, which the engine's loading tests cover.

use std::collections::HashMap;

use super::{add_tensor_payload_bytes, first_convolution_profile_name, insert_tensor_once};
use crate::{TensorDtype, TensorProfile};
use text_generation::execution_error::TextExecutionError;

fn float_profile(tensor_name: &str) -> TensorProfile {
    TensorProfile {
        name: tensor_name.to_owned(),
        dtype: TensorDtype::ModelFloat,
        shape: vec![1],
        equivalent_published_shapes: Vec::new(),
    }
}

#[test]
fn should_add_a_tensor_payload_to_the_running_total() {
    let total_payload_bytes = add_tensor_payload_bytes(1_073_741_824, 512, "norm.weight")
        .expect("a total far below the u64 range should add");

    assert_eq!(total_payload_bytes, 1_073_742_336);
}

#[test]
fn should_accept_a_total_that_lands_exactly_on_the_u64_limit() {
    let total_payload_bytes = add_tensor_payload_bytes(u64::MAX - 8, 8, "norm.weight")
        .expect("a total equal to u64::MAX should not count as an overflow");

    assert_eq!(total_payload_bytes, u64::MAX);
}

#[test]
fn should_name_the_tensor_that_overflows_the_payload_total() {
    let Err(overflow_error) =
        add_tensor_payload_bytes(u64::MAX, 1, "language_model.lm_head.weight")
    else {
        panic!("a total past u64::MAX must be refused");
    };

    assert!(matches!(
        overflow_error,
        TextExecutionError::InvalidTensor {
            tensor_name,
            description: "total tensor payload byte count overflowed",
        } if tensor_name == "language_model.lm_head.weight"
    ));
}

#[test]
fn should_bind_each_tensor_name_once() {
    let mut bound_tensors = HashMap::new();

    insert_tensor_once(&mut bound_tensors, "language_model.model.norm.weight", 7)
        .expect("the first binding of a name should succeed");
    insert_tensor_once(&mut bound_tensors, "language_model.lm_head.weight", 9)
        .expect("a different name should bind separately");

    assert_eq!(bound_tensors.len(), 2);
    assert_eq!(bound_tensors["language_model.model.norm.weight"], 7);
}

#[test]
fn should_refuse_a_second_binding_of_the_same_tensor_name() {
    let mut bound_tensors = HashMap::new();
    insert_tensor_once(&mut bound_tensors, "language_model.model.norm.weight", 7)
        .expect("the first binding of a name should succeed");

    let Err(duplicate_error) =
        insert_tensor_once(&mut bound_tensors, "language_model.model.norm.weight", 8)
    else {
        panic!("a second binding of one name must be refused");
    };

    assert!(matches!(
        duplicate_error,
        TextExecutionError::InvalidTensor {
            tensor_name,
            description: "tensor was bound more than once",
        } if tensor_name == "language_model.model.norm.weight"
    ));
}

#[test]
fn should_pick_the_first_linear_attention_convolution_weight() {
    let language_tensor_profiles = [
        float_profile("language_model.model.layers.0.input_layernorm.weight"),
        float_profile("language_model.model.layers.0.linear_attn.conv1d.weight"),
        float_profile("language_model.model.layers.1.linear_attn.conv1d.weight"),
    ];

    assert_eq!(
        first_convolution_profile_name(&language_tensor_profiles),
        Some("language_model.model.layers.0.linear_attn.conv1d.weight")
    );
}

#[test]
fn should_find_no_convolution_weight_without_a_linear_attention_layer() {
    let language_tensor_profiles = [
        float_profile("language_model.model.layers.3.self_attn.q_norm.weight"),
        float_profile("language_model.model.layers.0.linear_attn.dt_bias"),
    ];

    assert_eq!(
        first_convolution_profile_name(&language_tensor_profiles),
        None
    );
    assert_eq!(first_convolution_profile_name(&[]), None);
}
