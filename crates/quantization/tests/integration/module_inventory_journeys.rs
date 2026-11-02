//! Inventory against real safetensors files written to a temporary
//! directory with the `safetensors` crate. No MLX: only headers are read.

use std::path::Path;

use astronomical_quantization::bit_width::GroupSize;
use astronomical_quantization::module_inventory::{
    ModuleInventory, ModuleWeightSource, SourceTensorDtype,
};
use astronomical_quantization::module_taxonomy::ModuleRole;

/// Writes one shard holding bf16 tensors of the given shapes (all zeros; the
/// inventory never reads the data) and a minimal Qwen3.5-style config.
pub fn write_bf16_source(directory: &Path, tensors: &[(&str, &[usize])], model_type: &str) {
    let mut views = Vec::new();
    let mut buffers = Vec::new();
    for (_, shape) in tensors {
        let element_count: usize = shape.iter().product();
        buffers.push(vec![0_u8; element_count * 2]);
    }
    for ((name, shape), buffer) in tensors.iter().zip(&buffers) {
        views.push((
            (*name).to_owned(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, shape.to_vec(), buffer)
                .expect("the fixture tensor view should be valid"),
        ));
    }
    // The workspace pins `safetensors` without its `std` feature, so the
    // in-memory serializer plus a plain file write stands in for
    // `serialize_to_file`.
    let shard_bytes =
        safetensors::tensor::serialize(views, None).expect("the fixture shard should serialize");
    std::fs::write(directory.join("model.safetensors"), shard_bytes)
        .expect("the fixture shard should be writable");
    std::fs::write(
        directory.join("config.json"),
        format!(
            r#"{{"model_type": "{model_type}", "architectures": ["Example"], "hidden_size": 128}}"#
        ),
    )
    .expect("the fixture config should be writable");
}

#[test]
fn should_inventory_two_dimensional_linears_and_skip_norms() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    write_bf16_source(
        directory.path(),
        &[
            ("model.embed_tokens.weight", &[512, 128]),
            ("model.layers.0.self_attn.q_proj.weight", &[128, 128]),
            ("model.layers.0.input_layernorm.weight", &[128]),
            ("lm_head.weight", &[512, 128]),
        ],
        "qwen3_5",
    );

    let inventory = ModuleInventory::from_source_directory(
        directory.path(),
        GroupSize::new(64).expect("valid"),
    )
    .expect("the inventory should build");

    assert_eq!(inventory.module_count(), 3);
    assert_eq!(
        inventory.pass_through_tensor_names().as_slice(),
        &["model.layers.0.input_layernorm.weight".to_owned()]
    );
    let head = inventory
        .resolve("lm_head")
        .expect("the head is quantizable");
    assert_eq!(head.role, ModuleRole::EmbeddingOrOutputHead);
    assert_eq!(head.logical_parameter_count, 512 * 128);
}

#[test]
fn should_split_a_fused_gate_up_expert_tensor_into_two_qwen_switch_mlp_modules() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    write_bf16_source(
        directory.path(),
        &[
            ("model.layers.0.mlp.experts.gate_up_proj", &[4, 256, 128]),
            ("model.layers.0.mlp.experts.down_proj", &[4, 128, 128]),
        ],
        "qwen3_5_moe",
    );

    let inventory = ModuleInventory::from_source_directory(
        directory.path(),
        GroupSize::new(64).expect("valid"),
    )
    .expect("the inventory should build");

    let gate = inventory
        .resolve("model.layers.0.mlp.switch_mlp.gate_proj")
        .expect("the gate half is a module");
    assert_eq!(gate.logical_shape, vec![4, 128, 128]);
    assert!(matches!(
        gate.weight_source,
        ModuleWeightSource::FusedGateUpHalf { half_index: 0, .. }
    ));
    assert!(
        inventory
            .resolve("model.layers.0.mlp.switch_mlp.up_proj")
            .is_some()
    );
    assert!(
        inventory
            .resolve("model.layers.0.mlp.switch_mlp.down_proj")
            .is_some()
    );
}

#[test]
fn should_stack_unfused_per_expert_tensors_in_expert_order() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    write_bf16_source(
        directory.path(),
        &[
            ("model.layers.0.mlp.experts.1.up_proj.weight", &[256, 128]),
            ("model.layers.0.mlp.experts.0.up_proj.weight", &[256, 128]),
        ],
        "laguna",
    );

    let inventory = ModuleInventory::from_source_directory(
        directory.path(),
        GroupSize::new(64).expect("valid"),
    )
    .expect("the inventory should build");

    let stacked = inventory
        .resolve("model.layers.0.mlp.switch_mlp.up_proj")
        .expect("the stacked module exists");
    assert_eq!(stacked.logical_shape, vec![2, 256, 128]);
    match &stacked.weight_source {
        ModuleWeightSource::StackedPerExpert { per_expert_sources } => {
            assert_eq!(
                per_expert_sources[0].1,
                "model.layers.0.mlp.experts.0.up_proj.weight"
            );
            assert_eq!(
                per_expert_sources[1].1,
                "model.layers.0.mlp.experts.1.up_proj.weight"
            );
        }
        other => panic!("expected a stacked source, got {other:?}"),
    }
}

/// Writes one shard of tensors with explicit dtypes and the given config.
pub fn write_typed_source(
    directory: &Path,
    tensors: &[(&str, safetensors::Dtype, &[usize])],
    config_json: &str,
) {
    let mut buffers = Vec::new();
    for (_, dtype, shape) in tensors {
        let element_count: usize = shape.iter().product();
        buffers.push(vec![0_u8; element_count * dtype.bitsize() / 8]);
    }
    let mut views = Vec::new();
    for ((name, dtype, shape), buffer) in tensors.iter().zip(&buffers) {
        views.push((
            (*name).to_owned(),
            safetensors::tensor::TensorView::new(*dtype, shape.to_vec(), buffer)
                .expect("the fixture tensor view should be valid"),
        ));
    }
    let shard_bytes =
        safetensors::tensor::serialize(views, None).expect("the fixture shard should serialize");
    std::fs::write(directory.join("model.safetensors"), shard_bytes)
        .expect("the fixture shard should be writable");
    std::fs::write(directory.join("config.json"), config_json)
        .expect("the fixture config should be writable");
}

#[test]
fn should_rebuild_the_logical_width_of_an_mlx_packed_source_from_its_scales() {
    // A 4-bit packed [256, 128] linear is stored as [256, 16] words with
    // [256, 2] scales at group 64; the logical width is 2 * 64 = 128.
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    write_typed_source(
        directory.path(),
        &[
            (
                "model.layers.0.mlp.up_proj.weight",
                safetensors::Dtype::U32,
                &[256, 16],
            ),
            (
                "model.layers.0.mlp.up_proj.scales",
                safetensors::Dtype::BF16,
                &[256, 2],
            ),
            (
                "model.layers.0.mlp.up_proj.biases",
                safetensors::Dtype::BF16,
                &[256, 2],
            ),
        ],
        r#"{"model_type": "qwen3_5", "quantization": {"group_size": 64, "bits": 4}}"#,
    );

    let inventory = ModuleInventory::from_source_directory(
        directory.path(),
        GroupSize::new(64).expect("valid"),
    )
    .expect("the inventory should build");

    let up_projection = inventory
        .resolve("model.layers.0.mlp.up_proj")
        .expect("the packed linear is a module");
    assert_eq!(up_projection.logical_shape, vec![256, 128]);
    assert_eq!(up_projection.logical_parameter_count, 256 * 128);
    assert_eq!(
        up_projection.source_dtype,
        SourceTensorDtype::MlxAffinePacked
    );
    assert!(inventory.pass_through_tensor_names().is_empty());
}

#[test]
fn should_pass_a_packed_source_through_when_its_group_size_is_undeclared() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    write_typed_source(
        directory.path(),
        &[
            (
                "model.layers.0.mlp.up_proj.weight",
                safetensors::Dtype::U32,
                &[256, 16],
            ),
            (
                "model.layers.0.mlp.up_proj.scales",
                safetensors::Dtype::BF16,
                &[256, 2],
            ),
        ],
        r#"{"model_type": "qwen3_5"}"#,
    );

    let inventory = ModuleInventory::from_source_directory(
        directory.path(),
        GroupSize::new(64).expect("valid"),
    )
    .expect("the inventory should build");

    // The scales travel with the weight: without them the copied words are
    // undecodable.
    assert_eq!(inventory.module_count(), 0);
    assert_eq!(
        inventory.pass_through_tensor_names().as_slice(),
        &[
            "model.layers.0.mlp.up_proj.weight".to_owned(),
            "model.layers.0.mlp.up_proj.scales".to_owned()
        ]
    );
}

#[test]
fn should_accept_a_block_scaled_fp8_weight_and_consume_its_scale() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    write_typed_source(
        directory.path(),
        &[
            (
                "model.layers.0.self_attn.q_proj.weight",
                safetensors::Dtype::F8_E4M3,
                &[128, 128],
            ),
            (
                "model.layers.0.self_attn.q_proj.weight_scale_inv",
                safetensors::Dtype::F32,
                &[1, 1],
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                safetensors::Dtype::F8_E4M3,
                &[128, 128],
            ),
        ],
        r#"{"model_type": "qwen3_8"}"#,
    );

    let inventory = ModuleInventory::from_source_directory(
        directory.path(),
        GroupSize::new(64).expect("valid"),
    )
    .expect("the inventory should build");

    let query = inventory
        .resolve("model.layers.0.self_attn.q_proj")
        .expect("the scaled FP8 weight is a module");
    assert_eq!(
        query.source_dtype,
        SourceTensorDtype::Float8E4M3WithBlockScale
    );
    // An FP8 tensor with no block scale cannot be dequantized, so it passes.
    assert_eq!(
        inventory.pass_through_tensor_names().as_slice(),
        &["model.layers.0.self_attn.k_proj.weight".to_owned()]
    );
}

#[test]
fn should_pass_the_router_through_and_rename_nemotron_expert_projections() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    write_typed_source(
        directory.path(),
        &[
            (
                "backbone.layers.1.mixer.gate.weight",
                safetensors::Dtype::BF16,
                &[4, 128],
            ),
            (
                "backbone.layers.1.mixer.experts.0.up_proj.weight",
                safetensors::Dtype::BF16,
                &[256, 128],
            ),
            (
                "backbone.layers.1.mixer.experts.1.up_proj.weight",
                safetensors::Dtype::BF16,
                &[256, 128],
            ),
        ],
        r#"{"model_type": "nemotron_h", "architectures": ["NemotronHForCausalLM"]}"#,
    );

    let inventory = ModuleInventory::from_source_directory(
        directory.path(),
        GroupSize::new(64).expect("valid"),
    )
    .expect("the inventory should build");

    let first_feed_forward = inventory
        .resolve("backbone.layers.1.mixer.switch_mlp.fc1")
        .expect("up_proj stacks under fc1");
    assert_eq!(first_feed_forward.logical_shape, vec![2, 256, 128]);
    assert_eq!(first_feed_forward.role, ModuleRole::RoutedExpert);
    assert_eq!(
        inventory.pass_through_tensor_names().as_slice(),
        &["backbone.layers.1.mixer.gate.weight".to_owned()]
    );
}

#[test]
fn should_pass_a_single_unfused_expert_through_rather_than_stack_it() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    write_typed_source(
        directory.path(),
        &[(
            "model.layers.0.mlp.experts.0.down_proj.weight",
            safetensors::Dtype::BF16,
            &[128, 256],
        )],
        r#"{"model_type": "laguna"}"#,
    );

    let inventory = ModuleInventory::from_source_directory(
        directory.path(),
        GroupSize::new(64).expect("valid"),
    )
    .expect("the inventory should build");

    assert_eq!(inventory.module_count(), 0);
    assert_eq!(inventory.pass_through_tensor_names().len(), 1);
}

#[test]
fn should_refuse_a_config_that_is_not_json() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    write_typed_source(
        directory.path(),
        &[("lm_head.weight", safetensors::Dtype::BF16, &[512, 128])],
        "not json",
    );

    let inventory_outcome = ModuleInventory::from_source_directory(
        directory.path(),
        GroupSize::new(64).expect("valid"),
    );

    assert!(matches!(
        inventory_outcome,
        Err(astronomical_quantization::QuantizationError::ModelConfigUnreadable { .. })
    ));
}

#[test]
fn should_alias_the_flattened_and_reordered_language_tower_spellings() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    write_typed_source(
        directory.path(),
        &[(
            "model.language_model.layers.0.mlp.up_proj.weight",
            safetensors::Dtype::BF16,
            &[256, 128],
        )],
        r#"{"model_type": "qwen3_5"}"#,
    );

    let inventory = ModuleInventory::from_source_directory(
        directory.path(),
        GroupSize::new(64).expect("valid"),
    )
    .expect("the inventory should build");

    assert!(
        inventory
            .resolve("language_model.model.layers.0.mlp.up_proj")
            .is_some()
    );
    assert!(inventory.resolve("model.layers.0.mlp.up_proj").is_some());
    assert!(
        inventory
            .resolve("language_model.layers.0.mlp.up_proj")
            .is_some()
    );
}

#[test]
fn should_pass_unfused_experts_through_when_their_shapes_disagree() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    write_typed_source(
        directory.path(),
        &[
            (
                "model.layers.0.mlp.experts.0.up_proj.weight",
                safetensors::Dtype::BF16,
                &[256, 128],
            ),
            (
                "model.layers.0.mlp.experts.1.up_proj.weight",
                safetensors::Dtype::BF16,
                &[128, 128],
            ),
        ],
        r#"{"model_type": "laguna"}"#,
    );

    let inventory = ModuleInventory::from_source_directory(
        directory.path(),
        GroupSize::new(64).expect("valid"),
    )
    .expect("the inventory should build");

    assert_eq!(inventory.module_count(), 0);
    assert_eq!(inventory.pass_through_tensor_names().len(), 2);
}

#[test]
fn should_pass_a_packed_weight_through_with_its_scales_when_it_becomes_no_module() {
    // A 4-bit packed [8, 12] word array with [8, 3] scales at group 32 has a
    // logical width of 96, which does not divide group 64: the weight passes
    // through as stored, and its scales and biases must ride with it.
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    write_typed_source(
        directory.path(),
        &[
            (
                "model.layers.0.mlp.up_proj.weight",
                safetensors::Dtype::U32,
                &[8, 12],
            ),
            (
                "model.layers.0.mlp.up_proj.scales",
                safetensors::Dtype::BF16,
                &[8, 3],
            ),
            (
                "model.layers.0.mlp.up_proj.biases",
                safetensors::Dtype::BF16,
                &[8, 3],
            ),
        ],
        r#"{"model_type": "qwen3", "quantization": {"bits": 4, "group_size": 32}}"#,
    );

    let inventory = ModuleInventory::from_source_directory(
        directory.path(),
        GroupSize::new(64).expect("valid"),
    )
    .expect("the inventory should build");

    assert_eq!(inventory.module_count(), 0);
    let mut pass_through_names = inventory.pass_through_tensor_names();
    pass_through_names.sort();
    assert_eq!(
        pass_through_names,
        vec![
            "model.layers.0.mlp.up_proj.biases".to_owned(),
            "model.layers.0.mlp.up_proj.scales".to_owned(),
            "model.layers.0.mlp.up_proj.weight".to_owned(),
        ]
    );
}
