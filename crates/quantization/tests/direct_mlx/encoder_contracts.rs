//! The Metal encoder against the pure reference, plus the invariants the
//! encoder must hold regardless of implementation.

use astronomical_quantization::bit_width::{BitWidth, GroupSize};
use astronomical_quantization::encoder::RangeSearchPolicy;
use astronomical_quantization::encoder::mlx::quantize_best;
use astronomical_quantization::encoder::reference::{dequantize_slab, encode_slab};

use super::support::{contract_runtime, outlier_heavy_weights};

const ROWS: usize = 16;
const COLUMNS: usize = 256;

#[test]
#[ignore = "pending implementation of quantize_best and the reference encoder"]
fn should_match_the_reference_encoder_bit_for_bit_on_packed_codes_at_two_bits() {
    let runtime = contract_runtime();
    let weights = outlier_heavy_weights(ROWS, COLUMNS);
    let weights_array = runtime
        .array_from_f32(&weights, &[ROWS as i32, COLUMNS as i32])
        .expect("the weights array should be created");
    let bit_width = BitWidth::new(2).expect("valid");
    let group_size = GroupSize::new(64).expect("valid");

    let gpu = quantize_best(
        &runtime,
        &weights_array,
        bit_width,
        group_size,
        RangeSearchPolicy::default_policy(),
    )
    .expect("the GPU encoder should succeed");
    let reference = encode_slab(&weights, ROWS, COLUMNS, bit_width, group_size, true)
        .expect("the reference encoder should succeed");

    assert_eq!(
        gpu.packed_weight
            .to_vec_u32()
            .expect("packed words read back"),
        reference.packed_words
    );
    let gpu_scales = runtime
        .astype(&gpu.scales, mlx_c_rust::MlxDtype::Float32)
        .expect("scales cast")
        .to_vec_f32()
        .expect("scales read back");
    for (gpu_scale, reference_scale) in gpu_scales.iter().zip(&reference.scales) {
        // Scales were rounded through float32 on both sides.
        assert!((gpu_scale - reference_scale).abs() <= 1e-6);
    }
}

#[test]
#[ignore = "pending implementation of quantize_best"]
fn should_produce_the_stock_mlx_layout_above_four_bits() {
    // At 8 bits the search is bypassed and the output must equal
    // runtime.quantize_affine exactly, so artifacts stay byte-reproducible.
    let runtime = contract_runtime();
    let weights = outlier_heavy_weights(ROWS, COLUMNS);
    let weights_array = runtime
        .array_from_f32(&weights, &[ROWS as i32, COLUMNS as i32])
        .expect("the weights array should be created");

    let searched = quantize_best(
        &runtime,
        &weights_array,
        BitWidth::new(8).expect("valid"),
        GroupSize::new(64).expect("valid"),
        RangeSearchPolicy::default_policy(),
    )
    .expect("the GPU encoder should succeed");
    let (stock_packed, _, _) = runtime
        .quantize_affine(&weights_array, 64, 8)
        .expect("stock quantization should succeed");

    assert_eq!(
        searched.packed_weight.to_vec_u32().expect("read back"),
        stock_packed.to_vec_u32().expect("read back")
    );
}

#[test]
#[ignore = "pending implementation of quantize_best"]
fn should_never_have_higher_squared_error_than_stock_min_max() {
    let runtime = contract_runtime();
    let weights = outlier_heavy_weights(ROWS, COLUMNS);
    let weights_array = runtime
        .array_from_f32(&weights, &[ROWS as i32, COLUMNS as i32])
        .expect("the weights array should be created");
    let bit_width = BitWidth::new(2).expect("valid");
    let group_size = GroupSize::new(64).expect("valid");

    let searched = quantize_best(
        &runtime,
        &weights_array,
        bit_width,
        group_size,
        RangeSearchPolicy::default_policy(),
    )
    .expect("the GPU encoder should succeed");
    let (stock_packed, stock_scales, stock_biases) = runtime
        .quantize_affine(&weights_array, 64, 2)
        .expect("stock quantization should succeed");
    let searched_reconstruction = runtime
        .dequantize_affine(
            &searched.packed_weight,
            &searched.scales,
            &searched.biases,
            64,
            2,
        )
        .expect("dequantize");
    let stock_reconstruction = runtime
        .dequantize_affine(&stock_packed, &stock_scales, &stock_biases, 64, 2)
        .expect("dequantize");

    let searched_error =
        squared_error_against(&weights, &to_f32(&runtime, &searched_reconstruction));
    let stock_error = squared_error_against(&weights, &to_f32(&runtime, &stock_reconstruction));
    assert!(searched_error <= stock_error);
}

#[test]
#[ignore = "pending implementation of quantize_best"]
fn should_dequantize_through_mlx_to_the_same_values_as_the_reference() {
    // The GPU-encoded artifact must read back through MLX's own dequantize
    // (what every loader uses) to the values the reference predicts.
    let runtime = contract_runtime();
    let weights = outlier_heavy_weights(ROWS, COLUMNS);
    let weights_array = runtime
        .array_from_f32(&weights, &[ROWS as i32, COLUMNS as i32])
        .expect("the weights array should be created");
    let bit_width = BitWidth::new(3).expect("valid");
    let group_size = GroupSize::new(64).expect("valid");

    let gpu = quantize_best(
        &runtime,
        &weights_array,
        bit_width,
        group_size,
        RangeSearchPolicy::default_policy(),
    )
    .expect("the GPU encoder should succeed");
    let mlx_reconstruction = to_f32(
        &runtime,
        &runtime
            .dequantize_affine(&gpu.packed_weight, &gpu.scales, &gpu.biases, 64, 3)
            .expect("dequantize"),
    );
    let reference = encode_slab(&weights, ROWS, COLUMNS, bit_width, group_size, true)
        .expect("the reference encoder should succeed");
    let reference_reconstruction = dequantize_slab(&reference);

    for (mlx_value, reference_value) in mlx_reconstruction.iter().zip(&reference_reconstruction) {
        assert!((mlx_value - reference_value).abs() <= 1e-5);
    }
}

fn to_f32(
    runtime: &astronomical_runtime_integration::MlxRuntime,
    array: &mlx_c_rust::MlxArray,
) -> Vec<f32> {
    runtime
        .astype(array, mlx_c_rust::MlxDtype::Float32)
        .expect("cast")
        .to_vec_f32()
        .expect("read back")
}

fn squared_error_against(original: &[f32], reconstruction: &[f32]) -> f64 {
    original
        .iter()
        .zip(reconstruction)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum()
}
