//! Unit tests for the structural bit policy and the shape adjustment.

use super::*;

fn bits(value: u32) -> BitWidth {
    BitWidth::new(value).expect("valid")
}

fn group(value: u32) -> GroupSize {
    GroupSize::new(value).expect("valid")
}

#[test]
#[ignore = "pending implementation of BitPolicy::decide"]
fn should_never_quantize_the_router_under_any_policy() {
    let structural = BitPolicy::Structural(StructuralMixtureOfExpertsPolicy::optiq_defaults());
    let uniform = BitPolicy::Uniform {
        bit_width: bits(4),
        group_size: group(64),
    };

    assert_eq!(
        structural.decide("model.layers.3.mlp.gate.weight"),
        TensorEncodingDecision::PassThrough
    );
    assert_eq!(
        uniform.decide("model.layers.3.mlp.gate.weight"),
        TensorEncodingDecision::PassThrough
    );
}

#[test]
#[ignore = "pending implementation of BitPolicy::decide"]
fn should_put_routed_experts_at_two_bits_and_attention_at_six_under_the_structural_policy() {
    let structural = BitPolicy::Structural(StructuralMixtureOfExpertsPolicy::optiq_defaults());

    assert_eq!(
        structural.decide("model.layers.3.mlp.switch_mlp.up_proj.weight"),
        TensorEncodingDecision::Encode {
            bit_width: bits(2),
            group_size: group(128),
        }
    );
    assert_eq!(
        structural.decide("model.layers.3.self_attn.q_proj.weight"),
        TensorEncodingDecision::Encode {
            bit_width: bits(6),
            group_size: group(128),
        }
    );
    assert_eq!(
        structural.decide("model.layers.3.mlp.shared_experts.up_proj.weight"),
        TensorEncodingDecision::Encode {
            bit_width: bits(8),
            group_size: group(64),
        }
    );
}

#[test]
#[ignore = "pending implementation of BitPolicy::adjust_for_shape"]
fn should_fall_back_to_group_size_32_when_the_planned_group_does_not_divide() {
    let planned = TensorEncodingDecision::Encode {
        bit_width: bits(4),
        group_size: group(64),
    };

    assert_eq!(
        BitPolicy::adjust_for_shape(planned, &[512, 160]),
        TensorEncodingDecision::Encode {
            bit_width: bits(4),
            group_size: group(32),
        }
    );
    assert_eq!(
        BitPolicy::adjust_for_shape(planned, &[512, 100]),
        TensorEncodingDecision::PassThrough
    );
    assert_eq!(
        BitPolicy::adjust_for_shape(planned, &[512]),
        TensorEncodingDecision::PassThrough
    );
}

#[test]
#[ignore = "pending implementation of is_router_tensor_key"]
fn should_recognize_every_router_spelling() {
    assert!(is_router_tensor_key("model.layers.1.mlp.gate.weight"));
    assert!(is_router_tensor_key("model.layers.1.ffn.gate.weight"));
    assert!(is_router_tensor_key("model.layers.1.mlp.gate.expert_bias"));
    assert!(is_router_tensor_key(
        "model.layers.1.mlp.router.proj.weight"
    ));
    assert!(!is_router_tensor_key("model.layers.1.mlp.gate_proj.weight"));
}
