//! Unit contracts for the generated language tensor profiles.
//!
//! The exact profile inventories pin the mixed-precision OptiQ layout of the
//! frozen Ornith 1.0 contract, including the resident profile that must never
//! bind sparse selected experts.

use std::collections::BTreeMap;

use crate::qwen3_5_core::frozen_config_test_support::frozen_ornith_1_0_config;
use test_harness::fixtures::frozen_ornith_1_0_fixture::expected_quantized_module_names;

use super::*;

#[test]
fn should_generate_the_complete_mixed_precision_optiq_language_tensor_profile() {
    let ornith_config = frozen_ornith_1_0_config();

    let tensor_profiles = qwen3_5_language_tensor_profiles(&ornith_config)
        .expect("the frozen ornith language tensor profiles should build");
    let tensor_profile_by_name = tensor_profile_by_name(&tensor_profiles);

    assert_eq!(tensor_profiles.len(), 1_757);
    assert_eq!(tensor_profile_by_name.len(), 1_757);
    assert_embedding_and_linear_attention_tensor_profiles(&tensor_profile_by_name);
    assert_first_layer_feed_forward_tensor_profiles(&tensor_profile_by_name);
    assert_attention_norm_and_output_head_tensor_profiles(&tensor_profile_by_name);
    assert_eq!(count_tensors_in_layer(&tensor_profiles, 0), 45);
    assert_eq!(count_tensors_in_layer(&tensor_profiles, 3), 40);
    assert!(
        !tensor_profile_by_name
            .keys()
            .any(|tensor_name| tensor_name.contains("in_proj_qkvz")
                || tensor_name.contains("in_proj_ba"))
    );
    assert!(
        !tensor_profile_by_name
            .keys()
            .any(|tensor_name| tensor_name.starts_with("vision_tower."))
    );
}

fn assert_embedding_and_linear_attention_tensor_profiles(
    tensor_profile_by_name: &BTreeMap<&str, &TensorProfile>,
) {
    assert_tensor(
        tensor_profile_by_name,
        "language_model.model.embed_tokens.weight",
        TensorDtype::UInt32,
        &[248_320, 512],
    );
    assert_tensor(
        tensor_profile_by_name,
        "language_model.model.embed_tokens.scales",
        TensorDtype::AffineQuantizationFloat,
        &[248_320, 32],
    );
    assert_tensor(
        tensor_profile_by_name,
        "language_model.model.layers.0.linear_attn.conv1d.weight",
        TensorDtype::ModelFloat,
        &[8_192, 4, 1],
    );
    assert_eq!(
        tensor_profile_by_name["language_model.model.layers.0.linear_attn.conv1d.weight"]
            .equivalent_published_shapes,
        [vec![8_192, 1, 4]]
    );
    assert_tensor(
        tensor_profile_by_name,
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
        TensorDtype::UInt32,
        &[8_192, 256],
    );
    assert_tensor(
        tensor_profile_by_name,
        "language_model.model.layers.0.linear_attn.in_proj_z.scales",
        TensorDtype::AffineQuantizationFloat,
        &[4_096, 32],
    );
    assert_tensor(
        tensor_profile_by_name,
        "language_model.model.layers.0.linear_attn.in_proj_b.biases",
        TensorDtype::AffineQuantizationFloat,
        &[32, 32],
    );
    assert_tensor(
        tensor_profile_by_name,
        "language_model.model.layers.0.linear_attn.A_log",
        TensorDtype::ModelFloat,
        &[32],
    );
    assert_tensor(
        tensor_profile_by_name,
        "language_model.model.layers.0.linear_attn.dt_bias",
        TensorDtype::ModelFloat,
        &[32],
    );
    assert_tensor(
        tensor_profile_by_name,
        "language_model.model.layers.0.linear_attn.norm.weight",
        TensorDtype::ModelFloat,
        &[128],
    );
}

fn assert_first_layer_feed_forward_tensor_profiles(
    tensor_profile_by_name: &BTreeMap<&str, &TensorProfile>,
) {
    assert_tensor(
        tensor_profile_by_name,
        "language_model.model.layers.0.mlp.gate.weight",
        TensorDtype::UInt32,
        &[256, 256],
    );
    assert_tensor(
        tensor_profile_by_name,
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
        TensorDtype::UInt32,
        &[256, 512, 256],
    );
    assert_tensor(
        tensor_profile_by_name,
        "language_model.model.layers.0.mlp.switch_mlp.down_proj.scales",
        TensorDtype::AffineQuantizationFloat,
        &[256, 2_048, 8],
    );
    assert_tensor(
        tensor_profile_by_name,
        "language_model.model.layers.0.mlp.shared_expert.down_proj.weight",
        TensorDtype::UInt32,
        &[2_048, 64],
    );
    assert_tensor(
        tensor_profile_by_name,
        "language_model.model.layers.0.mlp.shared_expert_gate.weight",
        TensorDtype::UInt32,
        &[1, 256],
    );
}

fn assert_attention_norm_and_output_head_tensor_profiles(
    tensor_profile_by_name: &BTreeMap<&str, &TensorProfile>,
) {
    assert_tensor(
        tensor_profile_by_name,
        "language_model.model.layers.3.self_attn.q_proj.weight",
        TensorDtype::UInt32,
        &[8_192, 256],
    );
    assert_tensor(
        tensor_profile_by_name,
        "language_model.model.layers.39.self_attn.q_proj.weight",
        TensorDtype::UInt32,
        &[8_192, 512],
    );
    assert_tensor(
        tensor_profile_by_name,
        "language_model.model.layers.3.self_attn.o_proj.scales",
        TensorDtype::AffineQuantizationFloat,
        &[2_048, 64],
    );
    assert_tensor(
        tensor_profile_by_name,
        "language_model.model.layers.3.self_attn.q_norm.weight",
        TensorDtype::ModelFloat,
        &[256],
    );
    assert_tensor(
        tensor_profile_by_name,
        "language_model.model.layers.39.self_attn.k_proj.biases",
        TensorDtype::AffineQuantizationFloat,
        &[512, 32],
    );
    assert_tensor(
        tensor_profile_by_name,
        "language_model.model.norm.weight",
        TensorDtype::ModelFloat,
        &[2_048],
    );
    assert_tensor(
        tensor_profile_by_name,
        "language_model.lm_head.biases",
        TensorDtype::AffineQuantizationFloat,
        &[248_320, 32],
    );
}

#[cfg(feature = "direct-mlx")]
#[test]
fn should_exclude_sparse_expert_tensors_from_every_resident_profile() {
    let ornith_config = frozen_ornith_1_0_config();

    let resident_tensor_profiles = qwen3_5_resident_language_tensor_profiles(&ornith_config)
        .expect("the frozen ornith resident language tensor profiles should build");
    let resident_tensor_profile_by_name = tensor_profile_by_name(&resident_tensor_profiles);

    assert_eq!(resident_tensor_profiles.len(), 1_397);
    assert_eq!(count_tensors_in_layer(&resident_tensor_profiles, 0), 36);
    assert_eq!(count_tensors_in_layer(&resident_tensor_profiles, 3), 31);
    assert!(
        !resident_tensor_profile_by_name
            .keys()
            .any(|tensor_name| tensor_name.contains(".mlp.switch_mlp.")),
        "resident profile must not bind sparse selected experts"
    );
    assert_tensor(
        &resident_tensor_profile_by_name,
        "language_model.model.layers.0.mlp.gate.weight",
        TensorDtype::UInt32,
        &[256, 256],
    );
    assert_tensor(
        &resident_tensor_profile_by_name,
        "language_model.model.layers.0.mlp.shared_expert.down_proj.weight",
        TensorDtype::UInt32,
        &[2_048, 64],
    );
    assert_tensor(
        &resident_tensor_profile_by_name,
        "language_model.model.layers.0.mlp.shared_expert_gate.weight",
        TensorDtype::UInt32,
        &[1, 256],
    );
}

#[test]
fn should_widen_gate_and_up_projections_and_narrow_the_down_projection() {
    let projection_dimensions = expert_projection_dimensions(2_048, 512);

    assert_eq!(
        projection_dimensions,
        [
            ("gate_proj", 512, 2_048),
            ("up_proj", 512, 2_048),
            ("down_proj", 2_048, 512),
        ]
    );
}

#[test]
fn should_keep_every_projection_square_when_hidden_and_intermediate_sizes_match() {
    let projection_dimensions = expert_projection_dimensions(1, 1);

    for (projection_name, output_dimension, input_dimension) in projection_dimensions {
        assert_eq!(
            (output_dimension, input_dimension),
            (1, 1),
            "widths of {projection_name}"
        );
    }
}

#[test]
fn should_append_router_routed_experts_shared_expert_and_its_gate_in_that_order() {
    let ornith_config = frozen_ornith_1_0_config();
    let decoder_layer_prefix = "language_model.model.layers.0";
    let mut tensor_profiles = Vec::new();

    append_qwen3_5_moe_feed_forward_tensor_profiles(
        &mut tensor_profiles,
        decoder_layer_prefix,
        2_048,
        &ornith_config,
    )
    .expect("the frozen ornith feed-forward tensor profiles should build");

    let mut module_names_in_order: Vec<&str> = Vec::new();
    for tensor_profile in &tensor_profiles {
        let (module_name, _tensor_kind) = tensor_profile
            .name
            .rsplit_once('.')
            .expect("every tensor name ends in its tensor kind");
        if module_names_in_order.last() != Some(&module_name) {
            module_names_in_order.push(module_name);
        }
    }
    // The shared fixture lists the layer's modules in the order a checkpoint
    // declares them, so the expected order is read from it rather than retyped.
    let expected_module_names = expected_quantized_module_names()
        .into_iter()
        .filter(|module_name| module_name.starts_with("language_model.model.layers.0.mlp."))
        .collect::<Vec<_>>();
    assert_eq!(module_names_in_order, expected_module_names);
    assert_eq!(tensor_profiles.len(), 24);
}

#[test]
fn should_stack_routed_experts_over_the_expert_count_and_leave_shared_experts_unstacked() {
    let ornith_config = frozen_ornith_1_0_config();
    let mut tensor_profiles = Vec::new();

    append_qwen3_5_moe_feed_forward_tensor_profiles(
        &mut tensor_profiles,
        "language_model.model.layers.0",
        2_048,
        &ornith_config,
    )
    .expect("the frozen ornith feed-forward tensor profiles should build");
    let tensor_profile_by_name = tensor_profile_by_name(&tensor_profiles);

    assert_tensor(
        &tensor_profile_by_name,
        "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight",
        TensorDtype::UInt32,
        &[256, 512, 256],
    );
    assert_tensor(
        &tensor_profile_by_name,
        "language_model.model.layers.0.mlp.shared_expert.up_proj.weight",
        TensorDtype::UInt32,
        &[512, 256],
    );
}

fn tensor_profile_by_name(tensor_profiles: &[TensorProfile]) -> BTreeMap<&str, &TensorProfile> {
    tensor_profiles
        .iter()
        .map(|tensor_profile| (tensor_profile.name.as_str(), tensor_profile))
        .collect()
}

fn assert_tensor(
    tensor_profile_by_name: &BTreeMap<&str, &TensorProfile>,
    tensor_name: &str,
    expected_dtype: TensorDtype,
    expected_shape: &[usize],
) {
    let tensor_profile = tensor_profile_by_name
        .get(tensor_name)
        .unwrap_or_else(|| panic!("expected tensor profile {tensor_name}"));
    assert_eq!(
        tensor_profile.dtype, expected_dtype,
        "dtype for {tensor_name}"
    );
    assert_eq!(
        tensor_profile.shape, expected_shape,
        "shape for {tensor_name}"
    );
}

fn count_tensors_in_layer(tensor_profiles: &[TensorProfile], decoder_layer_index: usize) -> usize {
    let layer_prefix = format!("language_model.model.layers.{decoder_layer_index}.");
    tensor_profiles
        .iter()
        .filter(|tensor_profile| tensor_profile.name.starts_with(&layer_prefix))
        .count()
}
