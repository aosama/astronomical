//! The precise matrix product against a float64 product on the host.

use serial_test::serial;

use super::super::gpu_test_support::sequence_values;
use super::*;
use astronomical_model_primitives::synthetic_weights_test_support::initialize_test_runtime;

/// `[rows, inner] · [inner, columns]` in float64, the reference.
fn host_product(left: &[f32], right: &[f32], shape: (usize, usize, usize)) -> Vec<f64> {
    let (rows, inner, columns) = shape;
    let mut product = vec![0.0_f64; rows * columns];
    for row_index in 0..rows {
        for column_index in 0..columns {
            let mut dot_product = 0.0_f64;
            for inner_index in 0..inner {
                dot_product += f64::from(left[row_index * inner + inner_index])
                    * f64::from(right[inner_index * columns + column_index]);
            }
            product[row_index * columns + column_index] = dot_product;
        }
    }
    product
}

fn largest_relative_gap(gpu_values: &[f32], reference_values: &[f64]) -> f64 {
    let mut reference_scale = 0.0_f64;
    for reference_value in reference_values {
        reference_scale = reference_scale.max(reference_value.abs());
    }
    let mut largest_gap = 0.0_f64;
    for (gpu_value, reference_value) in gpu_values.iter().zip(reference_values) {
        largest_gap = largest_gap.max((f64::from(*gpu_value) - reference_value).abs());
    }
    largest_gap / reference_scale
}

#[test]
#[serial]
fn should_multiply_matrices_to_float32_accuracy_where_a_plain_product_may_not() {
    let runtime = initialize_test_runtime();
    let math = GpuMath { runtime: &runtime };
    // The width of a Kev hidden state, where the served head's 0.009 gap
    // between TF32 and float32 was found.
    let shape = (24, 1024, 16);
    let left_values = sequence_values(shape.0 * shape.1, 7);
    let right_values = sequence_values(shape.1 * shape.2, 11);
    let left = math.upload(&left_values, &[shape.0, shape.1]).unwrap();
    let right = math.upload(&right_values, &[shape.1, shape.2]).unwrap();
    let reference = host_product(&left_values, &right_values, shape);
    let precise = math
        .download(&math.precise_matmul(&left, &right).unwrap())
        .unwrap();
    assert_eq!(precise.len(), reference.len());
    let precise_gap = largest_relative_gap(&precise, &reference);
    // Float32 itself, summing 1024 products, is good to a few 1e-6 here;
    // TF32 is about 1e-3.
    // The plain product is not compared with: whether it rounds depends on
    // the GPU. On one with TF32 this case measured 2.1e-6 precise against
    // 6.5e-4 plain; on CI's GPU, without TF32, the plain product is already
    // full float32 and can beat the precise one's dropped low-by-low term.
    // What training relies on holds everywhere: float32 accuracy.
    assert!(
        precise_gap < 1e-5,
        "precise product off by {precise_gap:e} relative"
    );
}

#[test]
#[serial]
fn should_give_exact_products_of_values_that_need_no_rounding() {
    let runtime = initialize_test_runtime();
    let math = GpuMath { runtime: &runtime };
    let left = math.upload(&[1.0, 2.0, -0.5, 4.0], &[2, 2]).unwrap();
    let right = math.upload(&[0.25, 0.0, 1.0, -2.0], &[2, 2]).unwrap();
    let product = math
        .download(&math.precise_matmul(&left, &right).unwrap())
        .unwrap();
    assert_eq!(product, vec![2.25, -4.0, 3.875, -8.0]);
}
