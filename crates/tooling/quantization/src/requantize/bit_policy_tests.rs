//! Unit tests for the structural bit policy and the shape adjustment.

use super::*;

fn bits(value: u32) -> BitWidth {
    BitWidth::new(value).expect("valid")
}

fn group(value: u32) -> GroupSize {
    GroupSize::new(value).expect("valid")
}

#[test]
fn should_never_quantize_the_router_under_the_structural_policy() {
    let structural = BitPolicy::Structural(StructuralMixtureOfExpertsPolicy::optiq_defaults());

    assert_eq!(
        structural.decide("model.layers.3.mlp.gate.weight"),
        TensorEncodingDecision::PassThrough
    );
}

#[test]
fn should_pin_the_router_and_the_shared_expert_gate_at_eight_bits_under_the_uniform_policy() {
    let uniform = BitPolicy::Uniform {
        bit_width: bits(4),
        group_size: group(128),
    };

    assert_eq!(
        uniform.decide("model.layers.3.mlp.gate.weight"),
        TensorEncodingDecision::Encode {
            bit_width: bits(8),
            group_size: group(64),
        }
    );
    assert_eq!(
        uniform.decide("model.layers.3.mlp.shared_expert_gate.weight"),
        TensorEncodingDecision::Encode {
            bit_width: bits(8),
            group_size: group(64),
        }
    );
    assert_eq!(
        uniform.decide("model.layers.3.mlp.gate_proj.weight"),
        TensorEncodingDecision::Encode {
            bit_width: bits(4),
            group_size: group(128),
        }
    );
}

#[test]
fn should_fall_back_to_the_largest_group_that_divides_the_last_dimension() {
    let planned = TensorEncodingDecision::Encode {
        bit_width: bits(4),
        group_size: group(128),
    };

    assert_eq!(
        BitPolicy::adjust_for_shape(planned, &[8, 192]),
        TensorEncodingDecision::Encode {
            bit_width: bits(4),
            group_size: group(64),
        }
    );
    assert_eq!(
        BitPolicy::adjust_for_shape(planned, &[8, 160]),
        TensorEncodingDecision::Encode {
            bit_width: bits(4),
            group_size: group(32),
        }
    );
    assert_eq!(
        BitPolicy::adjust_for_shape(planned, &[8, 100]),
        TensorEncodingDecision::PassThrough
    );
}

#[test]
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
fn should_recognize_every_router_spelling() {
    assert!(is_router_tensor_key("model.layers.1.mlp.gate.weight"));
    assert!(is_router_tensor_key("model.layers.1.ffn.gate.weight"));
    assert!(is_router_tensor_key("model.layers.1.mlp.gate.expert_bias"));
    assert!(is_router_tensor_key(
        "model.layers.1.mlp.router.proj.weight"
    ));
    assert!(!is_router_tensor_key("model.layers.1.mlp.gate_proj.weight"));
}

#[test]
fn should_apply_the_one_width_under_the_uniform_policy() {
    let uniform = BitPolicy::Uniform {
        bit_width: bits(4),
        group_size: group(64),
    };

    assert_eq!(
        uniform.decide("model.layers.3.mlp.up_proj.weight"),
        TensorEncodingDecision::Encode {
            bit_width: bits(4),
            group_size: group(64),
        }
    );
}

#[test]
fn should_pass_a_keep_bfloat16_module_through_under_the_predicate_policy() {
    let decision = decision_from_predicate(QuantizationDecisionForModule::KeepBfloat16);

    assert_eq!(decision, TensorEncodingDecision::PassThrough);
}

#[test]
fn should_leave_a_pass_through_decision_alone_and_accept_three_dimensional_experts() {
    let planned = TensorEncodingDecision::Encode {
        bit_width: bits(2),
        group_size: group(128),
    };

    assert_eq!(
        BitPolicy::adjust_for_shape(TensorEncodingDecision::PassThrough, &[64, 128]),
        TensorEncodingDecision::PassThrough
    );
    assert_eq!(
        BitPolicy::adjust_for_shape(planned, &[8, 256, 512]),
        planned
    );
    assert_eq!(
        BitPolicy::adjust_for_shape(planned, &[2, 8, 256, 512]),
        TensorEncodingDecision::PassThrough
    );
}

#[test]
fn should_exclude_the_shared_expert_from_routed_experts() {
    assert!(is_routed_expert_tensor_key(
        "model.layers.1.mlp.experts.gate_up_proj"
    ));
    assert!(is_routed_expert_tensor_key(
        "model.layers.1.mlp.switch_mlp.up_proj.weight"
    ));
    assert!(!is_routed_expert_tensor_key(
        "model.layers.1.mlp.shared_experts.up_proj.weight"
    ));
}

#[test]
fn should_encode_nothing_under_the_keep_everything_policy() {
    let keep_everything = BitPolicy::KeepEverythingBfloat16;

    assert_eq!(
        keep_everything.decide("model.layers.3.mlp.up_proj.weight"),
        TensorEncodingDecision::PassThrough
    );
    assert_eq!(
        keep_everything.decide("model.layers.3.mlp.gate.weight"),
        TensorEncodingDecision::PassThrough
    );
}
