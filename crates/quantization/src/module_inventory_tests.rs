//! Unit tests for the header-only module inventory. Integration coverage with
//! real safetensors files written to a temporary directory lives under
//! `tests/integration/module_inventory_journeys.rs`.

use super::*;

fn descriptor(
    module_name: &str,
    logical_shape: &[usize],
    role: ModuleRole,
) -> QuantizableModuleDescriptor {
    QuantizableModuleDescriptor {
        module_name: module_name.to_owned(),
        logical_shape: logical_shape.to_vec(),
        logical_parameter_count: logical_shape.iter().product::<usize>() as u64,
        source_dtype: SourceTensorDtype::BFloat16,
        weight_source: ModuleWeightSource::Whole {
            shard_file_name: "model.safetensors".to_owned(),
            tensor_name: format!("{module_name}.weight"),
        },
        role,
        transformer_block_index: crate::module_taxonomy::transformer_block_index(module_name),
    }
}

#[test]
#[ignore = "pending implementation of total_logical_parameter_count and transformer_block_count"]
fn should_sum_logical_parameters_and_count_blocks_from_descriptors() {
    let inventory = ModuleInventory::from_descriptors(
        PathBuf::from("unused"),
        GroupSize::new(64).expect("64 is a valid group size"),
        vec![
            descriptor(
                "model.layers.0.mlp.up_proj",
                &[256, 128],
                ModuleRole::DenseFeedForward,
            ),
            descriptor(
                "model.layers.4.mlp.up_proj",
                &[256, 128],
                ModuleRole::DenseFeedForward,
            ),
            descriptor("lm_head", &[1024, 128], ModuleRole::EmbeddingOrOutputHead),
        ],
    );

    assert_eq!(
        inventory.total_logical_parameter_count(),
        256 * 128 * 2 + 1024 * 128
    );
    assert_eq!(inventory.transformer_block_count(), 5);
}

#[test]
#[ignore = "pending implementation of resolve"]
fn should_resolve_the_model_prefix_stripped_alias() {
    // mlx-lm sometimes drops the leading `model.`; the probe model may name
    // the module either way and both must land on the same descriptor.
    let mut inventory = ModuleInventory::from_descriptors(
        PathBuf::from("unused"),
        GroupSize::new(64).expect("64 is a valid group size"),
        vec![descriptor(
            "model.layers.0.mlp.up_proj",
            &[256, 128],
            ModuleRole::DenseFeedForward,
        )],
    );
    inventory.aliases.insert(
        "layers.0.mlp.up_proj".to_owned(),
        "model.layers.0.mlp.up_proj".to_owned(),
    );

    assert!(inventory.resolve("layers.0.mlp.up_proj").is_some());
    assert!(inventory.resolve("model.layers.0.mlp.up_proj").is_some());
    assert!(inventory.resolve("layers.9.mlp.up_proj").is_none());
}

#[test]
#[ignore = "pending implementation of from_source_directory"]
fn should_report_a_missing_shard_directory_as_no_safetensors_shards() {
    let empty_directory = tempfile::tempdir().expect("a temporary directory should be creatable");

    let inventory_outcome = ModuleInventory::from_source_directory(
        empty_directory.path(),
        GroupSize::new(64).expect("64 is a valid group size"),
    );

    assert!(matches!(
        inventory_outcome,
        Err(QuantizationError::NoSafetensorsShards { .. })
    ));
}
