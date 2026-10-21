use std::collections::HashMap;

use astronomical_mlx_c_rust::{MlxArray, MlxDtype};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

use crate::qwen3_5_core::quantizations::optiq::OptiQQuantizationProfile;

use super::Qwen3_5VisionWeights;

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
            OptiQQuantizationProfile::affine(8, 64),
        )]),
        vision_sidecars: Vec::new(),
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
