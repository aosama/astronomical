use std::collections::HashMap;

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_c_rust::{MlxArray, MlxDtype};
use serial_test::serial;

use crate::qwen3_5_core::quantizations::mlx_quantization::ModuleQuantizationProfile;

use super::Qwen3_5VisionWeights;
use crate::qwen3_5_core::vision::vision_tensor_source::Qwen3_5VisionTensorSource;

pub(crate) fn test_runtime() -> MlxRuntime {
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
/// The stored spelling of the same tensor in a unified-naming shard, where the
/// tower is embedded beside the language weights.
const EMBEDDED_STORED_TENSOR_NAME: &str = "model.visual.blocks.0.attn.proj.bias";
const LANGUAGE_TENSOR_NAME: &str = "language_model.model.norm.weight";
/// Eight mebibytes of BF16: large enough that the memory drop on release
/// cannot be confused with allocator bookkeeping noise.
const RELEASED_TENSOR_ELEMENT_COUNT: usize = 4 * 1024 * 1024;
const RELEASED_TENSOR_PAYLOAD_BYTES: usize = RELEASED_TENSOR_ELEMENT_COUNT * 2;
const LANGUAGE_TENSOR_ELEMENT_COUNT: usize = 4096;
const BF16_ONE: [u8; 2] = [0x80, 0x3F];

/// Writes a SafeTensors file of BF16 ones, one tensor per `(name, element
/// count)` pair, laid out in the given order.
fn write_bf16_shard(shard_path: &std::path::Path, stored_tensors: &[(&str, usize)]) {
    use std::io::Write;
    let mut header = serde_json::Map::new();
    let mut payload_byte_count = 0_usize;
    for (stored_tensor_name, element_count) in stored_tensors {
        let tensor_start = payload_byte_count;
        payload_byte_count += element_count * 2;
        header.insert(
            (*stored_tensor_name).to_owned(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": [element_count],
                "data_offsets": [tensor_start, payload_byte_count],
            }),
        );
    }
    let header_bytes = serde_json::to_vec(&header).expect("the header should serialize");
    let mut shard_file = std::fs::File::create(shard_path).expect("the shard should be created");
    shard_file
        .write_all(&(header_bytes.len() as u64).to_le_bytes())
        .and_then(|()| shard_file.write_all(&header_bytes))
        .and_then(|()| shard_file.write_all(&BF16_ONE.repeat(payload_byte_count / 2)))
        .expect("the shard should be written");
}

fn released_tensor_profiles() -> Vec<crate::TensorProfile> {
    vec![crate::TensorProfile {
        name: RELEASED_TENSOR_NAME.to_owned(),
        dtype: crate::TensorDtype::BFloat16,
        shape: vec![RELEASED_TENSOR_ELEMENT_COUNT],
        equivalent_published_shapes: Vec::new(),
    }]
}

/// Binds lazy vision weights through the production source and binder.
pub(crate) fn vision_weights_over(
    runtime: &MlxRuntime,
    tensor_source: Qwen3_5VisionTensorSource,
) -> Result<Qwen3_5VisionWeights, super::Qwen3_5ExecutionError> {
    let tensor_profiles = released_tensor_profiles();
    let (bound_tensors, total_payload_bytes) =
        super::bind_vision_tensors(runtime, &tensor_profiles, &tensor_source)?;
    Ok(Qwen3_5VisionWeights {
        bound_tensors,
        quantization_profiles: HashMap::new(),
        tensor_profiles,
        tensor_source,
        total_payload_bytes,
    })
}

/// A sidecar-style source: one file holding the tower under its canonical name.
pub(crate) fn sidecar_source(shard_path: &std::path::Path) -> Qwen3_5VisionTensorSource {
    Qwen3_5VisionTensorSource::new(
        vec![std::fs::File::open(shard_path).expect("the shard should open")],
        HashMap::from([(
            RELEASED_TENSOR_NAME.to_owned(),
            super::VisionTensorLocation {
                source_file_position: 0,
                stored_tensor_name: RELEASED_TENSOR_NAME.to_owned(),
            },
        )]),
    )
}

/// Writes the one-tensor sidecar file the release tests read.
pub(crate) fn write_released_tensor_sidecar(shard_path: &std::path::Path, element_count: usize) {
    write_bf16_shard(shard_path, &[(RELEASED_TENSOR_NAME, element_count)]);
}

/// MLX memory the process holds after returning reclaimable buffers, so a
/// release that only moved bytes into the allocator cache does not count.
fn held_memory_bytes(runtime: &MlxRuntime) -> usize {
    runtime
        .synchronize_gpu_stream_and_clear_allocator_cache()
        .expect("the allocator cache should clear");
    let memory_snapshot = runtime
        .memory_snapshot()
        .expect("the MLX memory snapshot should be readable");
    memory_snapshot.active_memory_bytes() + memory_snapshot.allocator_cache_memory_bytes()
}

fn evaluate_released_tensor(runtime: &MlxRuntime, vision_weights: &Qwen3_5VisionWeights) {
    runtime
        .evaluate_arrays(&[vision_weights
            .tensor(RELEASED_TENSOR_NAME)
            .expect("the vision tensor should be bound")])
        .expect("the vision tensor should evaluate");
}

fn bound_tensor_sum(runtime: &MlxRuntime, vision_weights: &Qwen3_5VisionWeights) -> f32 {
    let bound_tensor = vision_weights
        .tensor(RELEASED_TENSOR_NAME)
        .expect("the vision tensor should be bound");
    let float32_tensor = runtime
        .astype(bound_tensor, MlxDtype::Float32)
        .expect("the bound tensor should widen");
    runtime
        .sum(&float32_tensor, false)
        .expect("the bound tensor should sum")
        .to_vec_f32()
        .expect("the sum should be readable")[0]
}

/// Evaluates, releases, and returns the held bytes before and after.
fn held_bytes_around_one_release(
    runtime: &MlxRuntime,
    vision_weights: &mut Qwen3_5VisionWeights,
) -> (usize, usize) {
    evaluate_released_tensor(runtime, vision_weights);
    let held_bytes_while_resident = held_memory_bytes(runtime);
    vision_weights
        .rebind_lazily(runtime)
        .expect("the vision weights should rebind");
    (held_bytes_while_resident, held_memory_bytes(runtime))
}

#[test]
#[serial]
fn should_release_evaluated_vision_bytes_and_read_them_again_after_rebinding() {
    let runtime = test_runtime();
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let shard_path = shard_directory.path().join("model-vision.safetensors");
    write_released_tensor_sidecar(&shard_path, RELEASED_TENSOR_ELEMENT_COUNT);
    let mut vision_weights = vision_weights_over(&runtime, sidecar_source(&shard_path))
        .expect("the vision tensor should bind");

    let (held_bytes_while_resident, held_bytes_after_release) =
        held_bytes_around_one_release(&runtime, &mut vision_weights);

    assert!(
        held_bytes_after_release + RELEASED_TENSOR_PAYLOAD_BYTES <= held_bytes_while_resident,
        "releasing must return the tower bytes, not park them in the allocator cache: \
         {held_bytes_while_resident} before, {held_bytes_after_release} after"
    );
    assert_eq!(
        bound_tensor_sum(&runtime, &vision_weights),
        RELEASED_TENSOR_ELEMENT_COUNT as f32,
        "the rebound tensor must read the same values from the file"
    );
}

#[test]
#[serial]
fn should_reuse_the_retained_descriptors_across_repeated_releases() {
    let runtime = test_runtime();
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let shard_path = shard_directory.path().join("model-vision.safetensors");
    write_released_tensor_sidecar(&shard_path, RELEASED_TENSOR_ELEMENT_COUNT);
    let mut vision_weights = vision_weights_over(&runtime, sidecar_source(&shard_path))
        .expect("the vision tensor should bind");

    held_bytes_around_one_release(&runtime, &mut vision_weights);
    let (held_bytes_while_resident, held_bytes_after_release) =
        held_bytes_around_one_release(&runtime, &mut vision_weights);

    assert!(
        held_bytes_after_release + RELEASED_TENSOR_PAYLOAD_BYTES <= held_bytes_while_resident,
        "a second image request must release the tower just like the first"
    );
    assert_eq!(
        bound_tensor_sum(&runtime, &vision_weights),
        RELEASED_TENSOR_ELEMENT_COUNT as f32
    );
}

#[test]
#[serial]
fn should_release_an_embedded_tower_while_the_language_map_of_its_shard_stays_loaded() {
    let runtime = test_runtime();
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let language_only_shard_path = shard_directory
        .path()
        .join("model-00001-of-00002.safetensors");
    let shared_shard_path = shard_directory
        .path()
        .join("model-00002-of-00002.safetensors");
    write_bf16_shard(
        &language_only_shard_path,
        &[(LANGUAGE_TENSOR_NAME, LANGUAGE_TENSOR_ELEMENT_COUNT)],
    );
    write_bf16_shard(
        &shared_shard_path,
        &[
            (LANGUAGE_TENSOR_NAME, LANGUAGE_TENSOR_ELEMENT_COUNT),
            (EMBEDDED_STORED_TENSOR_NAME, RELEASED_TENSOR_ELEMENT_COUNT),
        ],
    );
    // The language loader's own map of the shared shard, kept alive with an
    // evaluated tensor, as a loaded model keeps it.
    let language_shard_map = runtime
        .load_safetensors(
            std::fs::File::open(&shared_shard_path).expect("the shard should open"),
            None,
        )
        .expect("the language map should load");
    let language_tensor = language_shard_map
        .tensor(LANGUAGE_TENSOR_NAME)
        .expect("the language tensor should be present");
    runtime
        .evaluate_arrays(&[&language_tensor])
        .expect("the language tensor should evaluate");
    let tensor_source = Qwen3_5VisionTensorSource::keeping_referenced_files(
        vec![
            std::fs::File::open(&language_only_shard_path).expect("the shard should open"),
            std::fs::File::open(&shared_shard_path).expect("the shard should open"),
        ],
        HashMap::from([(
            RELEASED_TENSOR_NAME.to_owned(),
            super::VisionTensorLocation {
                source_file_position: 1,
                stored_tensor_name: EMBEDDED_STORED_TENSOR_NAME.to_owned(),
            },
        )]),
    );
    let mut vision_weights =
        vision_weights_over(&runtime, tensor_source).expect("the embedded tower should bind");

    let (held_bytes_while_resident, held_bytes_after_release) =
        held_bytes_around_one_release(&runtime, &mut vision_weights);

    assert_eq!(
        vision_weights.tensor_source.source_file_count(),
        1,
        "only the shard holding the tower is reparsed on release"
    );
    assert!(
        held_bytes_after_release + RELEASED_TENSOR_PAYLOAD_BYTES <= held_bytes_while_resident,
        "the language map of the same shard must not pin the tower's bytes"
    );
    assert_eq!(
        bound_tensor_sum(&runtime, &vision_weights),
        RELEASED_TENSOR_ELEMENT_COUNT as f32
    );
    drop(language_shard_map);
}

#[test]
#[serial]
fn should_rebind_never_evaluated_vision_weights_without_reading_them() {
    let runtime = test_runtime();
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let shard_path = shard_directory.path().join("model-vision.safetensors");
    write_released_tensor_sidecar(&shard_path, RELEASED_TENSOR_ELEMENT_COUNT);
    let mut vision_weights = vision_weights_over(&runtime, sidecar_source(&shard_path))
        .expect("the vision tensor should bind");
    let held_bytes_before_rebinding = held_memory_bytes(&runtime);

    vision_weights
        .rebind_lazily(&runtime)
        .expect("lazy vision weights should rebind");

    assert!(
        held_memory_bytes(&runtime) < held_bytes_before_rebinding + RELEASED_TENSOR_PAYLOAD_BYTES,
        "rebinding lazy weights must not load the tower"
    );
}

#[test]
#[serial]
fn should_keep_the_bound_tensors_when_the_source_file_changed_size() {
    let runtime = test_runtime();
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let shard_path = shard_directory.path().join("model-vision.safetensors");
    write_released_tensor_sidecar(&shard_path, RELEASED_TENSOR_ELEMENT_COUNT);
    let mut vision_weights = vision_weights_over(&runtime, sidecar_source(&shard_path))
        .expect("the vision tensor should bind");
    // Rewriting in place keeps the inode the retained descriptor points at,
    // so the reload sees a tensor of a different shape.
    write_released_tensor_sidecar(&shard_path, RELEASED_TENSOR_ELEMENT_COUNT / 2);

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

#[test]
#[serial]
fn should_reject_a_location_outside_the_source_files() {
    let runtime = test_runtime();
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let shard_path = shard_directory.path().join("model-vision.safetensors");
    write_released_tensor_sidecar(&shard_path, RELEASED_TENSOR_ELEMENT_COUNT);
    let tensor_source = Qwen3_5VisionTensorSource::keeping_referenced_files(
        vec![std::fs::File::open(&shard_path).expect("the shard should open")],
        HashMap::from([(
            RELEASED_TENSOR_NAME.to_owned(),
            super::VisionTensorLocation {
                source_file_position: 3,
                stored_tensor_name: RELEASED_TENSOR_NAME.to_owned(),
            },
        )]),
    );

    let binding_error = vision_weights_over(&runtime, tensor_source)
        .expect_err("a tensor in no source file must not bind");

    assert!(matches!(
        binding_error,
        super::Qwen3_5ExecutionError::InvalidTensor { tensor_name, .. }
            if tensor_name == RELEASED_TENSOR_NAME
    ));
}
