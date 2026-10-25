//! Inventory against real safetensors files written to a temporary
//! directory with the `safetensors` crate. No MLX: only headers are read.

use std::path::Path;

use astronomical_quantization::bit_width::GroupSize;
use astronomical_quantization::module_inventory::{ModuleInventory, ModuleWeightSource};
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
#[ignore = "pending implementation of ModuleInventory::from_source_directory"]
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
        inventory.pass_through_tensor_names(),
        &["model.layers.0.input_layernorm.weight".to_owned()]
    );
    let head = inventory
        .resolve("lm_head")
        .expect("the head is quantizable");
    assert_eq!(head.role, ModuleRole::EmbeddingOrOutputHead);
    assert_eq!(head.logical_parameter_count, 512 * 128);
}

#[test]
#[ignore = "pending implementation of ModuleInventory::from_source_directory"]
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
#[ignore = "pending implementation of ModuleInventory::from_source_directory"]
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
