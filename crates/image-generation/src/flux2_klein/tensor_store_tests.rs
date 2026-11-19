//! Contracts for the layout-aware tensor store: BF16 tensors pass through,
//! 4-bit tensors are found by their Diffusers names, packed projections bind
//! or dequantize, and VAE kernels come back in Diffusers order.

use std::fs::File;
use std::io::Write;

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_bridge::MlxDtype;
use serde_json::{Map, json};
use serial_test::serial;

use super::{Flux2KleinTensorStore, Flux2KleinTensorStoreError, Flux2KleinWeight};
use crate::flux2_klein::weight_layout::{Flux2KleinComponent, Flux2KleinWeightLayout};

const BF16_ONE: [u8; 2] = [0x80, 0x3F];
const BF16_TWO: [u8; 2] = [0x00, 0x40];
const PACKED_PROJECTION_NAME: &str = "transformer_blocks.0.attn.to_out.0.weight";

struct StoredFixtureTensor {
    name: &'static str,
    dtype: &'static str,
    shape: Vec<usize>,
    payload: Vec<u8>,
}

fn initialize_test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

fn store_of(
    runtime: &MlxRuntime,
    layout: Flux2KleinWeightLayout,
    component: Flux2KleinComponent,
    tensors: &[StoredFixtureTensor],
) -> (tempfile::TempDir, Flux2KleinTensorStore) {
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let shard_path = shard_directory.path().join("0.safetensors");
    let mut header = Map::new();
    let mut payload = Vec::new();
    for tensor in tensors {
        let start = payload.len();
        payload.extend_from_slice(&tensor.payload);
        header.insert(
            tensor.name.to_owned(),
            json!({"dtype": tensor.dtype, "shape": tensor.shape, "data_offsets": [start, payload.len()]}),
        );
    }
    let header_bytes = serde_json::to_vec(&header).expect("the header should serialize");
    let mut shard_file = File::create(&shard_path).expect("the shard should be created");
    shard_file
        .write_all(&(header_bytes.len() as u64).to_le_bytes())
        .and_then(|()| shard_file.write_all(&header_bytes))
        .and_then(|()| shard_file.write_all(&payload))
        .expect("the shard should be written");
    let shard = runtime
        .load_safetensors(
            File::open(&shard_path).expect("the shard should open"),
            None,
        )
        .expect("the shard should map");
    let store = Flux2KleinTensorStore::new(layout, component, vec![shard]);
    (shard_directory, store)
}

fn bf16_tensor(name: &'static str, shape: Vec<usize>, value: [u8; 2]) -> StoredFixtureTensor {
    let value_count = shape.iter().product::<usize>();
    StoredFixtureTensor {
        name,
        dtype: "BF16",
        shape,
        payload: value.repeat(value_count),
    }
}

/// The transformer's `transformer_blocks.0.attn.to_out` projection under its
/// mflux name, `[64, 64]` packed as zeros with scale 1 and offset 2, so every
/// dequantized value is exactly 2.
fn packed_transformer_projection() -> Vec<StoredFixtureTensor> {
    vec![
        StoredFixtureTensor {
            name: "transformer_blocks.0.attn.to_out.weight",
            dtype: "U32",
            shape: vec![64, 8],
            payload: vec![0; 64 * 8 * 4],
        },
        bf16_tensor(
            "transformer_blocks.0.attn.to_out.scales",
            vec![64, 1],
            BF16_ONE,
        ),
        bf16_tensor(
            "transformer_blocks.0.attn.to_out.biases",
            vec![64, 1],
            BF16_TWO,
        ),
    ]
}

#[test]
#[serial]
fn should_return_bf16_kernels_untouched_in_the_bf16_layout() {
    let runtime = initialize_test_runtime();
    let (_shard_directory, store) = store_of(
        &runtime,
        Flux2KleinWeightLayout::DiffusersBf16,
        Flux2KleinComponent::Vae,
        &[bf16_tensor(
            "decoder.conv_in.weight",
            vec![2, 1, 1, 3],
            BF16_ONE,
        )],
    );

    let kernel = store
        .dense(&runtime, "decoder.conv_in.weight")
        .expect("the kernel should load");

    assert_eq!(kernel.shape(), vec![2, 1, 1, 3]);
}

#[test]
#[serial]
fn should_return_4bit_vae_kernels_in_the_diffusers_order() {
    let runtime = initialize_test_runtime();
    let (_shard_directory, store) = store_of(
        &runtime,
        Flux2KleinWeightLayout::MlxAffine4Bit,
        Flux2KleinComponent::Vae,
        &[bf16_tensor(
            "decoder.conv_in.weight",
            vec![2, 1, 1, 3],
            BF16_ONE,
        )],
    );

    let kernel = store
        .dense(&runtime, "decoder.conv_in.weight")
        .expect("the kernel should load");

    assert_eq!(kernel.shape(), vec![2, 3, 1, 1]);
}

#[test]
#[serial]
fn should_dequantize_a_packed_projection_found_by_its_diffusers_name() {
    let runtime = initialize_test_runtime();
    let (_shard_directory, store) = store_of(
        &runtime,
        Flux2KleinWeightLayout::MlxAffine4Bit,
        Flux2KleinComponent::Transformer,
        &packed_transformer_projection(),
    );

    let dense_projection = store
        .dense(&runtime, PACKED_PROJECTION_NAME)
        .expect("the packed projection should dequantize");

    assert_eq!(dense_projection.shape(), vec![64, 64]);
    let dense_values = runtime
        .astype(&dense_projection, MlxDtype::Float32)
        .and_then(|float32_projection| float32_projection.to_vec_f32())
        .expect("the projection should evaluate");
    assert!(dense_values.iter().all(|dense_value| *dense_value == 2.0));
}

#[test]
#[serial]
fn should_keep_a_packed_projection_packed_when_bound_as_a_weight() {
    let runtime = initialize_test_runtime();
    let (_shard_directory, store) = store_of(
        &runtime,
        Flux2KleinWeightLayout::MlxAffine4Bit,
        Flux2KleinComponent::Transformer,
        &packed_transformer_projection(),
    );

    let projection = store
        .weight(PACKED_PROJECTION_NAME, &[64, 64])
        .expect("the packed projection should bind");

    assert!(matches!(projection, Flux2KleinWeight::Affine4Bit(_)));
    assert_eq!(projection.arrays().len(), 3);
}

#[test]
#[serial]
fn should_refuse_a_packed_projection_bound_at_another_logical_shape() {
    let runtime = initialize_test_runtime();
    let (_shard_directory, store) = store_of(
        &runtime,
        Flux2KleinWeightLayout::MlxAffine4Bit,
        Flux2KleinComponent::Transformer,
        &packed_transformer_projection(),
    );

    let refusal = store.weight(PACKED_PROJECTION_NAME, &[64, 128]);

    assert!(matches!(
        refusal,
        Err(Flux2KleinTensorStoreError::QuantizedLinear(_))
    ));
}

#[test]
#[serial]
fn should_report_a_tensor_no_shard_holds_as_missing() {
    let runtime = initialize_test_runtime();
    let (_shard_directory, store) = store_of(
        &runtime,
        Flux2KleinWeightLayout::MlxAffine4Bit,
        Flux2KleinComponent::Transformer,
        &packed_transformer_projection(),
    );

    let refusal = store.dense(&runtime, "proj_out.weight");

    assert!(matches!(
        refusal,
        Err(Flux2KleinTensorStoreError::MissingTensor { tensor_name, .. }) if tensor_name == "proj_out.weight"
    ));
}
