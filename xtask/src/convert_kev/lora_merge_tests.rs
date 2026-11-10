//! Unit contracts for the merge arithmetic: `W + scale · B · A`, rounded
//! once to the base's storage type, for any row count and core count.

use super::{LoraMergeInputs, MergeShape, merged_weight_bytes};
use crate::model_files::bfloat16::{bfloat16_bits_to_f32, f32_to_bfloat16_bits};
use crate::model_files::safetensors_file::TensorElementType;

fn bfloat16_bytes(values: &[f32]) -> Vec<u8> {
    let mut encoded_bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        encoded_bytes.extend(f32_to_bfloat16_bits(*value).to_le_bytes());
    }
    encoded_bytes
}

fn bfloat16_values(encoded_bytes: &[u8]) -> Vec<f32> {
    let (whole_values, _) = encoded_bytes.as_chunks::<2>();
    let mut values = Vec::with_capacity(whole_values.len());
    for value_bytes in whole_values {
        values.push(bfloat16_bits_to_f32(u16::from_le_bytes(*value_bytes)));
    }
    values
}

/// `W + scale · B · A` computed directly in float64, row by row.
fn exact_merge(
    base: &[f32],
    down: &[f32],
    up: &[f32],
    scale: f32,
    merge_shape: MergeShape,
) -> Vec<f64> {
    let mut merged_values = Vec::with_capacity(base.len());
    for output_row in 0..merge_shape.output_count {
        for input_column in 0..merge_shape.input_count {
            let mut delta = 0.0_f64;
            for rank_index in 0..merge_shape.rank {
                delta += f64::from(up[output_row * merge_shape.rank + rank_index])
                    * f64::from(down[rank_index * merge_shape.input_count + input_column]);
            }
            merged_values.push(
                f64::from(base[output_row * merge_shape.input_count + input_column])
                    + f64::from(scale) * delta,
            );
        }
    }
    merged_values
}

fn patterned_values(count: usize, step: f32) -> Vec<f32> {
    (0..count)
        .map(|value_index| ((value_index % 7) as f32 - 3.0) * step)
        .collect()
}

#[test]
fn should_merge_into_bfloat16_to_the_nearest_value_of_the_exact_merge() {
    let merge_shape = MergeShape {
        output_count: 37,
        input_count: 64,
        rank: 4,
    };
    let base = patterned_values(37 * 64, 0.01);
    let down = patterned_values(4 * 64, 0.003);
    let up = patterned_values(37 * 4, 0.005);

    let merged_bytes = merged_weight_bytes(&LoraMergeInputs {
        base_weight_bytes: &bfloat16_bytes(&base),
        base_element_type: TensorElementType::BFloat16,
        down_projection: &down,
        up_projection: &up,
        scale: 2.0,
        merge_shape,
    })
    .expect("the merge should succeed");

    let exact_values = exact_merge(
        &bfloat16_values(&bfloat16_bytes(&base)),
        &down,
        &up,
        2.0,
        merge_shape,
    );
    for (merged_value, exact_value) in bfloat16_values(&merged_bytes).iter().zip(&exact_values) {
        let bfloat16_step = (exact_value.abs() / 128.0).max(1.0e-30);
        assert!((f64::from(*merged_value) - exact_value).abs() <= bfloat16_step);
    }
}

#[test]
fn should_merge_into_float32_exactly_up_to_float32_rounding() {
    let merge_shape = MergeShape {
        output_count: 3,
        input_count: 2,
        rank: 1,
    };
    let base = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut base_bytes = Vec::new();
    for value in base {
        base_bytes.extend(value.to_le_bytes());
    }

    let merged_bytes = merged_weight_bytes(&LoraMergeInputs {
        base_weight_bytes: &base_bytes,
        base_element_type: TensorElementType::Float32,
        down_projection: &[1.0, 2.0],
        up_projection: &[1.0, 0.0, -1.0],
        scale: 0.5,
        merge_shape,
    })
    .expect("the merge should succeed");

    let (whole_values, _) = merged_bytes.as_chunks::<4>();
    let merged_values: Vec<f32> = whole_values
        .iter()
        .map(|value_bytes| f32::from_le_bytes(*value_bytes))
        .collect();
    assert_eq!(merged_values, vec![1.5, 3.0, 3.0, 4.0, 4.5, 5.0]);
}

#[test]
fn should_leave_the_weight_unchanged_when_the_adapter_is_zero() {
    let merge_shape = MergeShape {
        output_count: 1,
        input_count: 4,
        rank: 2,
    };
    let base_bytes = bfloat16_bytes(&[0.25, -0.5, 1.0, 0.0]);

    let merged_bytes = merged_weight_bytes(&LoraMergeInputs {
        base_weight_bytes: &base_bytes,
        base_element_type: TensorElementType::BFloat16,
        down_projection: &[0.0; 8],
        up_projection: &[0.0; 2],
        scale: 2.0,
        merge_shape,
    })
    .expect("a single-row merge should succeed");

    assert_eq!(merged_bytes, base_bytes);
}

#[test]
fn should_refuse_factors_that_do_not_fit_the_weight_or_an_unmerged_type() {
    let merge_shape = MergeShape {
        output_count: 2,
        input_count: 2,
        rank: 1,
    };
    let base_bytes = bfloat16_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let misshapen = LoraMergeInputs {
        base_weight_bytes: &base_bytes,
        base_element_type: TensorElementType::BFloat16,
        down_projection: &[1.0, 2.0, 3.0],
        up_projection: &[1.0, 1.0],
        scale: 1.0,
        merge_shape,
    };
    let half_precision_base = LoraMergeInputs {
        base_element_type: TensorElementType::Float16,
        down_projection: &[1.0, 2.0],
        ..misshapen
    };

    assert!(
        merged_weight_bytes(&LoraMergeInputs {
            base_element_type: TensorElementType::BFloat16,
            ..half_precision_base
        })
        .is_ok()
    );
    assert!(merged_weight_bytes(&misshapen).is_err());
    assert!(merged_weight_bytes(&half_precision_base).is_err());
}
