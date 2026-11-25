use std::collections::BTreeSet;

use crate::qwen3_5_core::qwen4_exp_config_test_support::frozen_flash_next_config;
use crate::{TensorDtype, TensorProfile};

use super::qwen4_exp_mtp_head_tensor_profiles;

/// The `mtp.*` tensor names of the published checkpoint's index.
const PUBLISHED_MTP_TENSOR_NAMES: [&str; 54] = [
    "mtp.fc_embedding.weight",
    "mtp.fc_hidden.weight",
    "mtp.hyper_connection_mixer.hc_norm.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_down.biases",
    "mtp.hyper_connection_mixer.input_mix_weight_down.scales",
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_up.biases",
    "mtp.hyper_connection_mixer.input_mix_weight_up.scales",
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.biases",
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.scales",
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.biases",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.scales",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.biases",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.scales",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight",
    "mtp.layers.0.mlp.switch_mlp.down_proj.biases",
    "mtp.layers.0.mlp.switch_mlp.down_proj.scales",
    "mtp.layers.0.mlp.switch_mlp.down_proj.weight",
    "mtp.layers.0.mlp.switch_mlp.gate_proj.biases",
    "mtp.layers.0.mlp.switch_mlp.gate_proj.scales",
    "mtp.layers.0.mlp.switch_mlp.gate_proj.weight",
    "mtp.layers.0.mlp.switch_mlp.up_proj.biases",
    "mtp.layers.0.mlp.switch_mlp.up_proj.scales",
    "mtp.layers.0.mlp.switch_mlp.up_proj.weight",
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.biases",
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.scales",
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight",
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.biases",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.scales",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.biases",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.scales",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight",
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
];

fn published_mtp_profiles() -> Vec<TensorProfile> {
    let mut config = frozen_flash_next_config();
    let published_names: BTreeSet<String> = PUBLISHED_MTP_TENSOR_NAMES
        .iter()
        .map(|name| (*name).to_owned())
        .collect();
    config.resolve_unquantized_modules_from_shard_index(&published_names);
    let architecture = config
        .qwen4_exp_architecture()
        .expect("the frozen configuration is Flash-Next");
    qwen4_exp_mtp_head_tensor_profiles(&config, &architecture).expect("the head's inventory builds")
}

fn profile_named<'profiles>(
    profiles: &'profiles [TensorProfile],
    name: &str,
) -> &'profiles TensorProfile {
    profiles
        .iter()
        .find(|profile| profile.name == name)
        .unwrap_or_else(|| panic!("{name} is missing from the head's inventory"))
}

#[test]
fn should_list_exactly_the_published_mtp_tensors() {
    let profile_names: BTreeSet<String> = published_mtp_profiles()
        .into_iter()
        .map(|profile| profile.name)
        .collect();
    let published_names: BTreeSet<String> = PUBLISHED_MTP_TENSOR_NAMES
        .iter()
        .map(|name| (*name).to_owned())
        .collect();
    assert_eq!(profile_names, published_names);
}

#[test]
fn should_expect_unquantized_attention_and_four_bit_experts() {
    let profiles = published_mtp_profiles();
    let query_projection = profile_named(&profiles, "mtp.layers.0.self_attn.q_proj.weight");
    assert_eq!(query_projection.dtype, TensorDtype::ModelFloat);
    assert_eq!(query_projection.shape, vec![12_288, 2_560]);
    let expert_gate = profile_named(&profiles, "mtp.layers.0.mlp.switch_mlp.gate_proj.weight");
    assert_eq!(expert_gate.dtype, TensorDtype::UInt32);
    assert_eq!(expert_gate.shape, vec![512, 640, 320]);
}

#[test]
fn should_size_the_fusion_norms_for_one_hidden_state_and_all_lanes() {
    let profiles = published_mtp_profiles();
    assert_eq!(
        profile_named(&profiles, "mtp.pre_fc_norm_embedding.weight").shape,
        vec![2_560]
    );
    assert_eq!(
        profile_named(&profiles, "mtp.pre_fc_norm_hidden.weight").shape,
        vec![10_240]
    );
}

#[test]
fn should_list_no_scales_for_a_head_whose_every_module_is_unquantized() {
    let mut config = frozen_flash_next_config();
    let weight_only_names: BTreeSet<String> = PUBLISHED_MTP_TENSOR_NAMES
        .iter()
        .filter(|name| name.ends_with(".weight"))
        .map(|name| (*name).to_owned())
        .collect();
    config.resolve_unquantized_modules_from_shard_index(&weight_only_names);
    let architecture = config.qwen4_exp_architecture().expect("Flash-Next");
    let profiles = qwen4_exp_mtp_head_tensor_profiles(&config, &architecture).expect("builds");
    assert!(
        profiles
            .iter()
            .all(|profile| profile.name.ends_with(".weight"))
    );
}
