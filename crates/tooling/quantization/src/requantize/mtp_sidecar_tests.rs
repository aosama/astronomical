use super::{
    FusedMtpExpertTensor, fused_mtp_expert_tensor, is_mtp_head_encodable, is_mtp_quantize_target,
    mtp_layer_count, unfused_mtp_expert_weight_name,
};
use crate::module_inventory::PassThroughTensor;

fn mtp_tensor(tensor_name: &str, shape: &[usize]) -> PassThroughTensor {
    PassThroughTensor::unchanged(
        "model.safetensors".to_owned(),
        tensor_name.to_owned(),
        shape.to_vec(),
    )
}

#[test]
fn should_quantize_projections_and_leave_norms_and_the_fusion_layer_alone() {
    assert!(is_mtp_quantize_target(
        "mtp.layers.0.self_attn.q_proj.weight"
    ));
    assert!(is_mtp_quantize_target("mtp.layers.0.mlp.down_proj.weight"));
    assert!(!is_mtp_quantize_target("mtp.fc.weight"));
    assert!(!is_mtp_quantize_target("mtp.norm.weight"));
    assert!(!is_mtp_quantize_target(
        "mtp.layers.0.self_attn.q_norm.weight"
    ));
    assert!(!is_mtp_quantize_target("mtp.pre_fc_norm_hidden.weight"));
    assert!(!is_mtp_quantize_target("mtp.layers.0.linear_attn.A_log"));
}

#[test]
fn should_recognize_only_three_dimensional_fused_expert_tensors() {
    assert_eq!(
        fused_mtp_expert_tensor("mtp.layers.0.mlp.experts.gate_up_proj", &[8, 128, 64]),
        Some(FusedMtpExpertTensor::GateUp)
    );
    assert_eq!(
        fused_mtp_expert_tensor("mtp.layers.0.mlp.experts.down_proj", &[8, 64, 64]),
        Some(FusedMtpExpertTensor::Down)
    );
    assert_eq!(
        fused_mtp_expert_tensor("mtp.layers.0.mlp.experts.down_proj", &[64, 64]),
        None
    );
    assert_eq!(
        fused_mtp_expert_tensor("mtp.layers.0.mlp.down_proj.weight", &[8, 64, 64]),
        None
    );
}

#[test]
fn should_name_a_split_expert_slice_by_its_expert_and_projection() {
    assert_eq!(
        unfused_mtp_expert_weight_name(
            "mtp.layers.0.mlp.experts.gate_up_proj",
            "gate_up_proj",
            3,
            "up_proj"
        ),
        "mtp.layers.0.mlp.experts.3.up_proj.weight"
    );
}

#[test]
fn should_count_distinct_head_layers_and_never_report_zero() {
    let two_layers = [
        mtp_tensor("mtp.layers.0.mlp.up_proj.weight", &[64, 64]),
        mtp_tensor("mtp.layers.0.mlp.down_proj.weight", &[64, 64]),
        mtp_tensor("mtp.layers.1.mlp.up_proj.weight", &[64, 64]),
        mtp_tensor("mtp.fc.weight", &[64, 128]),
    ];

    assert_eq!(mtp_layer_count(&two_layers), 2);
    assert_eq!(
        mtp_layer_count(&[mtp_tensor("mtp.fc.weight", &[64, 128])]),
        1
    );
    assert_eq!(mtp_layer_count(&[]), 1);
}

#[test]
fn should_refuse_a_head_with_a_projection_the_group_size_does_not_divide() {
    let encodable = [
        mtp_tensor("mtp.layers.0.mlp.up_proj.weight", &[128, 64]),
        mtp_tensor("mtp.norm.weight", &[100]),
        mtp_tensor("mtp.fc.weight", &[64, 100]),
    ];
    let with_an_odd_projection = [
        mtp_tensor("mtp.layers.0.mlp.up_proj.weight", &[128, 64]),
        mtp_tensor("mtp.layers.0.self_attn.q_proj.weight", &[128, 100]),
    ];
    let with_a_convolution = [mtp_tensor(
        "mtp.layers.0.linear_attn.conv1d.weight",
        &[64, 1, 4],
    )];

    assert!(is_mtp_head_encodable(&encodable, 64));
    assert!(!is_mtp_head_encodable(&with_an_odd_projection, 64));
    assert!(!is_mtp_head_encodable(&with_a_convolution, 64));
}
