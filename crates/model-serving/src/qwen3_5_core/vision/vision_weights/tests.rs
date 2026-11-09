use std::collections::HashMap;

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_c_rust::{MlxArray, MlxDtype};
use serial_test::serial;

use crate::qwen3_5_core::quantizations::mlx_quantization::ModuleQuantizationProfile;

use super::Qwen3_5VisionWeights;
use crate::qwen3_5_core::vision::vision_tensor_source::Qwen3_5VisionTensorSource;

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the vision linear test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

#[test]
#[serial]
fn should_apply_quantized_vision_linear_and_preserve_the_native_module_bias() {
    let runtime = test_runtime();
    let linear_module_name = "vision_tower.blocks.0.attn.proj";
    let mut bound_tensors = HashMap::new();
    bound_tensors.insert(
        format!("{linear_module_name}.weight"),
        MlxArray::from_u32(&[0; 32], &[2, 16])
            .expect("two affine packed output rows should be valid"),
    );
    let bfloat16_scales = runtime
        .array_from_f32(&[1.0, 1.0], &[2, 1])
        .and_then(|float32_scales| runtime.astype(&float32_scales, MlxDtype::BFloat16))
        .expect("two float32 scales should convert to bfloat16");
    let bfloat16_quantization_biases = runtime
        .array_from_f32(&[0.0, 0.0], &[2, 1])
        .and_then(|float32_biases| runtime.astype(&float32_biases, MlxDtype::BFloat16))
        .expect("two affine biases should convert to bfloat16");
    let bfloat16_linear_bias = runtime
        .array_from_f32(&[2.0, 3.0], &[2])
        .and_then(|float32_bias| runtime.astype(&float32_bias, MlxDtype::BFloat16))
        .expect("the learned linear bias should convert to bfloat16");
    bound_tensors.insert(format!("{linear_module_name}.scales"), bfloat16_scales);
    bound_tensors.insert(
        format!("{linear_module_name}.biases"),
        bfloat16_quantization_biases,
    );
    bound_tensors.insert(format!("{linear_module_name}.bias"), bfloat16_linear_bias);
    let vision_weights = Qwen3_5VisionWeights {
        bound_tensors,
        quantization_profiles: HashMap::from([(
            linear_module_name.to_owned(),
            ModuleQuantizationProfile::affine(8, 64),
        )]),
        tensor_profiles: Vec::new(),
        tensor_source: Qwen3_5VisionTensorSource::without_files(),
        source_safetensors: Vec::new(),
        total_payload_bytes: 0,
    };
    let input_states = runtime
        .array_from_f32(&vec![1.0; 64], &[1, 64])
        .expect("one vision input row should have 64 channels");

    let linear_output = vision_weights
        .linear(&runtime, &input_states, linear_module_name)
        .expect("the vision affine projection should execute");
    let linear_output_float32 = runtime
        .astype(&linear_output, MlxDtype::Float32)
        .expect("the output should convert to float32 for comparison");
    let linear_output_values = linear_output_float32
        .to_vec_f32()
        .expect("the projected values should be readable");

    assert_eq!(linear_output.shape(), [1, 2]);
    assert_eq!(linear_output_values, [2.0, 3.0]);
}

const RELEASED_TENSOR_NAME: &str = "vision_tower.blocks.0.attn.proj.bias";
/// Eight mebibytes of BF16: large enough that the active-memory drop on
/// release cannot be confused with allocator bookkeeping noise.
const RELEASED_TENSOR_ELEMENT_COUNT: usize = 4 * 1024 * 1024;
const RELEASED_TENSOR_PAYLOAD_BYTES: usize = RELEASED_TENSOR_ELEMENT_COUNT * 2;
const BF16_ONE: [u8; 2] = [0x80, 0x3F];

/// Writes a one-tensor SafeTensors file holding `element_count` BF16 ones.
fn write_single_bf16_shard(shard_path: &std::path::Path, element_count: usize) {
    use std::io::Write;
    let header = serde_json::json!({
        RELEASED_TENSOR_NAME: {
            "dtype": "BF16",
            "shape": [element_count],
            "data_offsets": [0, element_count * 2],
        }
    });
    let header_bytes = serde_json::to_vec(&header).expect("the header should serialize");
    let mut shard_file = std::fs::File::create(shard_path).expect("the shard should be created");
    shard_file
        .write_all(&(header_bytes.len() as u64).to_le_bytes())
        .and_then(|()| shard_file.write_all(&header_bytes))
        .and_then(|()| shard_file.write_all(&BF16_ONE.repeat(element_count)))
        .expect("the shard should be written");
}

/// Binds lazy vision weights over a one-tensor shard through the production
/// source and binder.
fn single_tensor_vision_weights(
    runtime: &MlxRuntime,
    shard_path: &std::path::Path,
) -> Qwen3_5VisionWeights {
    let tensor_profiles = vec![crate::TensorProfile {
        name: RELEASED_TENSOR_NAME.to_owned(),
        dtype: crate::TensorDtype::BFloat16,
        shape: vec![RELEASED_TENSOR_ELEMENT_COUNT],
        equivalent_published_shapes: Vec::new(),
    }];
    let tensor_source = Qwen3_5VisionTensorSource::new(
        vec![std::fs::File::open(shard_path).expect("the shard should open")],
        HashMap::from([(
            RELEASED_TENSOR_NAME.to_owned(),
            super::EmbeddedVisionTensorLocation {
                shard_position: 0,
                stored_tensor_name: RELEASED_TENSOR_NAME.to_owned(),
            },
        )]),
    );
    let source_safetensors = tensor_source
        .load_lazy_safetensors(runtime)
        .expect("the shard should load lazily");
    let (bound_tensors, total_payload_bytes) = super::bind_vision_tensors(
        runtime,
        &tensor_profiles,
        &tensor_source,
        &source_safetensors,
    )
    .expect("the vision tensor should bind");
    Qwen3_5VisionWeights {
        bound_tensors,
        quantization_profiles: HashMap::new(),
        tensor_profiles,
        tensor_source,
        source_safetensors,
        total_payload_bytes,
    }
}

fn active_memory_bytes(runtime: &MlxRuntime) -> usize {
    runtime
        .memory_snapshot()
        .expect("the MLX memory snapshot should be readable")
        .active_memory_bytes()
}

fn bound_tensor_sum(runtime: &MlxRuntime, vision_weights: &Qwen3_5VisionWeights) -> f32 {
    let bound_tensor = vision_weights
        .tensor(RELEASED_TENSOR_NAME)
        .expect("the vision tensor should be bound");
    runtime
        .astype(bound_tensor, MlxDtype::Float32)
        .and_then(|float32_tensor| runtime.sum(&float32_tensor, false))
        .and_then(|summed_tensor| summed_tensor.to_vec_f32())
        .expect("the bound tensor should sum")[0]
}

#[test]
#[serial]
fn should_release_evaluated_vision_bytes_and_read_them_again_after_rebinding() {
    let runtime = test_runtime();
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let shard_path = shard_directory.path().join("model-vision.safetensors");
    write_single_bf16_shard(&shard_path, RELEASED_TENSOR_ELEMENT_COUNT);
    let mut vision_weights = single_tensor_vision_weights(&runtime, &shard_path);
    runtime
        .evaluate_arrays(&[vision_weights
            .tensor(RELEASED_TENSOR_NAME)
            .expect("the vision tensor should be bound")])
        .expect("the vision tensor should evaluate");
    let active_bytes_while_resident = active_memory_bytes(&runtime);

    vision_weights
        .rebind_lazily(&runtime)
        .expect("the vision weights should rebind");
    let active_bytes_after_release = active_memory_bytes(&runtime);

    assert!(
        active_bytes_after_release + RELEASED_TENSOR_PAYLOAD_BYTES <= active_bytes_while_resident,
        "releasing must free the evaluated tower bytes: {active_bytes_while_resident} before, {active_bytes_after_release} after"
    );
    assert_eq!(
        bound_tensor_sum(&runtime, &vision_weights),
        RELEASED_TENSOR_ELEMENT_COUNT as f32,
        "the rebound tensor must read the same values from the file"
    );
}

#[test]
#[serial]
fn should_rebind_never_evaluated_vision_weights_without_reading_them() {
    let runtime = test_runtime();
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let shard_path = shard_directory.path().join("model-vision.safetensors");
    write_single_bf16_shard(&shard_path, RELEASED_TENSOR_ELEMENT_COUNT);
    let mut vision_weights = single_tensor_vision_weights(&runtime, &shard_path);
    let active_bytes_before_rebinding = active_memory_bytes(&runtime);

    vision_weights
        .rebind_lazily(&runtime)
        .expect("lazy vision weights should rebind");

    assert!(
        active_memory_bytes(&runtime)
            < active_bytes_before_rebinding + RELEASED_TENSOR_PAYLOAD_BYTES,
        "rebinding lazy weights must not load the tower"
    );
}

#[test]
#[serial]
fn should_keep_the_bound_tensors_when_the_source_file_changed_size() {
    let runtime = test_runtime();
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let shard_path = shard_directory.path().join("model-vision.safetensors");
    write_single_bf16_shard(&shard_path, RELEASED_TENSOR_ELEMENT_COUNT);
    let mut vision_weights = single_tensor_vision_weights(&runtime, &shard_path);
    // Rewriting in place keeps the inode the retained descriptor points at,
    // so the reload sees a tensor of a different shape.
    write_single_bf16_shard(&shard_path, RELEASED_TENSOR_ELEMENT_COUNT / 2);

    let rebind_error = vision_weights
        .rebind_lazily(&runtime)
        .expect_err("a changed tower must not rebind");

    assert!(
        matches!(
            rebind_error,
            super::Qwen3_5ExecutionError::InvalidTensor { .. }
        ),
        "the shape check must reject the changed tensor, got {rebind_error}"
    );
    assert_eq!(
        vision_weights
            .tensor(RELEASED_TENSOR_NAME)
            .expect("the original tensor should stay bound")
            .shape(),
        [RELEASED_TENSOR_ELEMENT_COUNT as i32]
    );
}
