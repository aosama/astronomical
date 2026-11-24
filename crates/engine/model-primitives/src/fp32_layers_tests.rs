//! Contracts for the shared FP32 layers: each matches an f64 CPU oracle,
//! keeps the input dtype, and refuses a shape it cannot broadcast. Values
//! come from the Romeo and Juliet fixture.

use mlx_bridge::MlxDtype;
use serial_test::file_serial;

use crate::synthetic_weights_test_support::{
    float32_values, initialize_test_runtime, largest_difference, upload_values,
    wide_shakespeare_values,
};

use super::{fp32_layer_norm, fp32_rms_norm, rotate_adjacent_pairs};

const ROW_COUNT: usize = 3;
const ROW_WIDTH: usize = 64;
const NORMALIZATION_EPSILON: f32 = 0.000_001;
/// FP32 normalization differs from the f64 oracle only by float rounding.
const FLOAT32_TOLERANCE: f32 = 0.000_1;
const BATCH_SIZE: usize = 1;
const TOKEN_COUNT: usize = 3;
const HEAD_COUNT: usize = 2;
const FEATURE_WIDTH: usize = 8;
/// The rotation is a few FP32 multiply-adds, so it holds a tighter bound.
const ROTATION_TOLERANCE: f32 = 0.000_01;

/// The textbook layer norm of one row, in f64: zero mean, unit variance.
fn layer_norm_oracle(row_values: &[f32]) -> Vec<f32> {
    let row_length = row_values.len() as f64;
    let mean = row_values
        .iter()
        .map(|value| f64::from(*value))
        .sum::<f64>()
        / row_length;
    let variance = row_values
        .iter()
        .map(|value| (f64::from(*value) - mean).powi(2))
        .sum::<f64>()
        / row_length;
    let inverse_deviation = 1.0 / (variance + f64::from(NORMALIZATION_EPSILON)).sqrt();
    let mut normalized_row = Vec::with_capacity(row_values.len());
    for row_value in row_values {
        normalized_row.push(((f64::from(*row_value) - mean) * inverse_deviation) as f32);
    }
    normalized_row
}

/// The textbook RMS norm of one row times its scale, in f64.
fn rms_norm_oracle(row_values: &[f32], scale_values: &[f32]) -> Vec<f32> {
    let row_length = row_values.len() as f64;
    let mean_square = row_values
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        / row_length;
    let inverse_root = 1.0 / (mean_square + f64::from(NORMALIZATION_EPSILON)).sqrt();
    let mut scaled_row = Vec::with_capacity(row_values.len());
    for (row_value, scale_value) in row_values.iter().zip(scale_values) {
        scaled_row.push((f64::from(*row_value) * inverse_root * f64::from(*scale_value)) as f32);
    }
    scaled_row
}

#[test]
#[file_serial]
fn should_layer_normalize_each_row_like_the_f64_oracle() {
    let runtime = initialize_test_runtime();
    let input_values = wide_shakespeare_values(ROW_COUNT * ROW_WIDTH, 0);
    let input = upload_values(
        &runtime,
        &input_values,
        &[ROW_COUNT as i32, ROW_WIDTH as i32],
        MlxDtype::Float32,
    );
    let normalized = fp32_layer_norm(&runtime, &input, NORMALIZATION_EPSILON)
        .expect("the layer norm should run");
    let mut expected_values = Vec::with_capacity(input_values.len());
    for input_row in input_values.chunks(ROW_WIDTH) {
        expected_values.extend(layer_norm_oracle(input_row));
    }
    let measured_values = float32_values(&runtime, &normalized);
    assert!(largest_difference(&measured_values, &expected_values) < FLOAT32_TOLERANCE);
}

#[test]
#[file_serial]
fn should_rms_normalize_and_scale_each_row_like_the_f64_oracle() {
    let runtime = initialize_test_runtime();
    let input_values = wide_shakespeare_values(ROW_COUNT * ROW_WIDTH, 512);
    let scale_values = wide_shakespeare_values(ROW_WIDTH, 4096);
    let input = upload_values(
        &runtime,
        &input_values,
        &[ROW_COUNT as i32, ROW_WIDTH as i32],
        MlxDtype::Float32,
    );
    let scale = upload_values(
        &runtime,
        &scale_values,
        &[ROW_WIDTH as i32],
        MlxDtype::Float32,
    );
    let normalized = fp32_rms_norm(&runtime, &input, &scale, NORMALIZATION_EPSILON)
        .expect("the RMS norm should run");
    let mut expected_values = Vec::with_capacity(input_values.len());
    for input_row in input_values.chunks(ROW_WIDTH) {
        expected_values.extend(rms_norm_oracle(input_row, &scale_values));
    }
    let measured_values = float32_values(&runtime, &normalized);
    assert!(largest_difference(&measured_values, &expected_values) < FLOAT32_TOLERANCE);
}

#[test]
#[file_serial]
fn should_return_the_input_dtype_after_normalizing_in_fp32() {
    let runtime = initialize_test_runtime();
    let input_values = wide_shakespeare_values(ROW_WIDTH, 1024);
    let bfloat16_input = upload_values(
        &runtime,
        &input_values,
        &[1, ROW_WIDTH as i32],
        MlxDtype::BFloat16,
    );
    let bfloat16_scale = upload_values(
        &runtime,
        &wide_shakespeare_values(ROW_WIDTH, 2048),
        &[ROW_WIDTH as i32],
        MlxDtype::BFloat16,
    );
    let layer_normalized = fp32_layer_norm(&runtime, &bfloat16_input, NORMALIZATION_EPSILON)
        .expect("the layer norm should run");
    let rms_normalized = fp32_rms_norm(
        &runtime,
        &bfloat16_input,
        &bfloat16_scale,
        NORMALIZATION_EPSILON,
    )
    .expect("the RMS norm should run");
    assert_eq!(layer_normalized.dtype(), MlxDtype::BFloat16);
    assert_eq!(rms_normalized.dtype(), MlxDtype::BFloat16);
}

#[test]
#[file_serial]
fn should_refuse_a_scale_that_cannot_broadcast_over_the_row() {
    let runtime = initialize_test_runtime();
    let input = upload_values(
        &runtime,
        &wide_shakespeare_values(ROW_WIDTH, 0),
        &[1, ROW_WIDTH as i32],
        MlxDtype::Float32,
    );
    let mismatched_scale = upload_values(
        &runtime,
        &wide_shakespeare_values(ROW_WIDTH - 1, 0),
        &[ROW_WIDTH as i32 - 1],
        MlxDtype::Float32,
    );
    let scaled_outcome = fp32_rms_norm(&runtime, &input, &mismatched_scale, NORMALIZATION_EPSILON)
        .and_then(|scaled| scaled.to_vec_f32());
    assert!(scaled_outcome.is_err());
}

fn head_shape() -> [i32; 4] {
    [
        BATCH_SIZE as i32,
        TOKEN_COUNT as i32,
        HEAD_COUNT as i32,
        FEATURE_WIDTH as i32,
    ]
}

/// One angle per token and pair, repeated over the pair's two features, as
/// the families lay their tables out.
fn angle_tables(angles_per_pair: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut cosine_values = Vec::with_capacity(TOKEN_COUNT * FEATURE_WIDTH);
    let mut sine_values = Vec::with_capacity(TOKEN_COUNT * FEATURE_WIDTH);
    for pair_angle in angles_per_pair {
        for _pair_feature in 0..2 {
            cosine_values.push(pair_angle.cos());
            sine_values.push(pair_angle.sin());
        }
    }
    (cosine_values, sine_values)
}

/// `(real, imaginary)` rotated by each token's angle, in f64.
fn rotation_oracle(head_values: &[f32], cosine_values: &[f32], sine_values: &[f32]) -> Vec<f32> {
    let mut rotated_values = vec![0.0_f32; head_values.len()];
    for token_index in 0..TOKEN_COUNT {
        for head_index in 0..HEAD_COUNT {
            let row_start = (token_index * HEAD_COUNT + head_index) * FEATURE_WIDTH;
            for pair_start in (0..FEATURE_WIDTH).step_by(2) {
                let table_index = token_index * FEATURE_WIDTH + pair_start;
                let cosine = f64::from(cosine_values[table_index]);
                let sine = f64::from(sine_values[table_index]);
                let real = f64::from(head_values[row_start + pair_start]);
                let imaginary = f64::from(head_values[row_start + pair_start + 1]);
                rotated_values[row_start + pair_start] = (real * cosine - imaginary * sine) as f32;
                rotated_values[row_start + pair_start + 1] =
                    (imaginary * cosine + real * sine) as f32;
            }
        }
    }
    rotated_values
}

#[test]
#[file_serial]
fn should_rotate_each_adjacent_pair_like_the_f64_oracle() {
    let runtime = initialize_test_runtime();
    let head_values = wide_shakespeare_values(TOKEN_COUNT * HEAD_COUNT * FEATURE_WIDTH, 0);
    let pair_angles = wide_shakespeare_values(TOKEN_COUNT * FEATURE_WIDTH / 2, 777);
    let (cosine_values, sine_values) = angle_tables(&pair_angles);
    let table_shape = [TOKEN_COUNT as i32, FEATURE_WIDTH as i32];
    let rotated = rotate_adjacent_pairs(
        &runtime,
        &upload_values(&runtime, &head_values, &head_shape(), MlxDtype::Float32),
        &upload_values(&runtime, &cosine_values, &table_shape, MlxDtype::Float32),
        &upload_values(&runtime, &sine_values, &table_shape, MlxDtype::Float32),
    )
    .expect("the rotation should run");
    let expected_values = rotation_oracle(&head_values, &cosine_values, &sine_values);
    let measured_values = float32_values(&runtime, &rotated);
    assert!(largest_difference(&measured_values, &expected_values) < ROTATION_TOLERANCE);
}

#[test]
#[file_serial]
fn should_leave_heads_unchanged_at_a_zero_angle_and_keep_their_dtype() {
    let runtime = initialize_test_runtime();
    let head_values = wide_shakespeare_values(TOKEN_COUNT * HEAD_COUNT * FEATURE_WIDTH, 2048);
    let (cosine_values, sine_values) = angle_tables(&[0.0; TOKEN_COUNT * FEATURE_WIDTH / 2]);
    let table_shape = [TOKEN_COUNT as i32, FEATURE_WIDTH as i32];
    let bfloat16_heads = upload_values(&runtime, &head_values, &head_shape(), MlxDtype::BFloat16);
    let rotated = rotate_adjacent_pairs(
        &runtime,
        &bfloat16_heads,
        &upload_values(&runtime, &cosine_values, &table_shape, MlxDtype::Float32),
        &upload_values(&runtime, &sine_values, &table_shape, MlxDtype::Float32),
    )
    .expect("the rotation should run");
    assert_eq!(rotated.dtype(), MlxDtype::BFloat16);
    assert_eq!(
        float32_values(&runtime, &rotated),
        float32_values(&runtime, &bfloat16_heads)
    );
}

#[test]
#[file_serial]
fn should_refuse_tables_that_cannot_broadcast_over_the_heads() {
    let runtime = initialize_test_runtime();
    let head_values = wide_shakespeare_values(TOKEN_COUNT * HEAD_COUNT * FEATURE_WIDTH, 0);
    let short_table_values = wide_shakespeare_values((TOKEN_COUNT - 1) * FEATURE_WIDTH, 0);
    let short_table_shape = [TOKEN_COUNT as i32 - 1, FEATURE_WIDTH as i32];
    let short_table = upload_values(
        &runtime,
        &short_table_values,
        &short_table_shape,
        MlxDtype::Float32,
    );
    let rotated_outcome = rotate_adjacent_pairs(
        &runtime,
        &upload_values(&runtime, &head_values, &head_shape(), MlxDtype::Float32),
        &short_table,
        &short_table,
    )
    .and_then(|rotated| rotated.to_vec_f32());
    assert!(rotated_outcome.is_err());
}
