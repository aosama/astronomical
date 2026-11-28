use std::collections::BTreeSet;

use super::refuse_unexpected_head_tensors;
use crate::qwen3_5_core::artifacts::dense_mtp_head_sidecar::{
    DenseMtpHeadSidecar, dense_mtp_head_tensor_profiles,
};
use crate::qwen3_5_core::frozen_config_test_support::frozen_dense_qwen3_6_config;

fn head_tensor_profiles() -> Vec<crate::TensorProfile> {
    let head_config = DenseMtpHeadSidecar {
        quantization_bits: 4,
        quantization_group_size: 64,
    }
    .head_config(&frozen_dense_qwen3_6_config());
    dense_mtp_head_tensor_profiles(&head_config).expect("the head's tensors should list")
}

fn profile_names(profiles: &[crate::TensorProfile]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for profile in profiles {
        names.insert(profile.name.clone());
    }
    names
}

#[test]
fn should_accept_a_file_holding_exactly_the_heads_tensors() {
    let profiles = head_tensor_profiles();
    refuse_unexpected_head_tensors(&profile_names(&profiles), &profiles)
        .expect("the head's own tensors are accepted");
}

/// A second decoder layer is a head of another shape, which this engine
/// would misread by binding only its first layer.
#[test]
fn should_refuse_a_file_with_a_second_decoder_layer() {
    let profiles = head_tensor_profiles();
    let mut shipped_names = profile_names(&profiles);
    shipped_names.insert("layers.1.input_layernorm.weight".to_owned());
    let refusal = refuse_unexpected_head_tensors(&shipped_names, &profiles)
        .expect_err("a second layer must be refused");
    assert!(
        refusal
            .to_string()
            .contains("layers.1.input_layernorm.weight")
    );
}

#[test]
fn should_accept_an_empty_file_here_and_leave_missing_tensors_to_binding() {
    let profiles = head_tensor_profiles();
    refuse_unexpected_head_tensors(&BTreeSet::new(), &profiles)
        .expect("missing tensors are refused when each is bound, not here");
}
