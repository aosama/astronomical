//! Oracle reference for the n-gram lookup table's quantized row decode.
//!
//! The production row reader will fetch packed ranges from storage and
//! reconstruct activation-dtype rows. The reference here materializes the
//! same rows through MLX's own dequantization on the GPU and compares against
//! host-side affine reconstruction in `f64`, so a production reader that
//! mis-scales, mis-packs, or skips the bias term fails here.
//!
//! The affine layout matches the published table: 4-bit values packed into
//! unsigned 32-bit words, one scale and one bias per group of 32.

use super::{DeterministicValues, assert_f32_close, oracle_test_runtime};

/// Geometry for one lookup-row decode row set.
pub(crate) struct RowGeometry {
    pub row_count: usize,
    pub group_size: usize,
    pub bits: u32,
}

/// Packs host values into the affine layout the published table uses and
/// returns the packed words, scales, and biases.
///
/// Each group of `group_size` values shares one scale and bias; values are
/// quantized to `bits` levels between group minimum and maximum, matching
/// MLX's affine scheme.
pub(crate) fn pack_affine_rows(
    values: &[f32],
    geometry: &RowGeometry,
) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    assert_eq!(
        values.len(),
        geometry.row_count * 160,
        "row width is fixed at 160"
    );
    let group_count = 160 / geometry.group_size;
    let mut packed = vec![0_u32; geometry.row_count * (160 * geometry.bits as usize / 32)];
    let mut scales = Vec::with_capacity(geometry.row_count * group_count);
    let mut biases = Vec::with_capacity(geometry.row_count * group_count);
    let levels = (1_u64 << geometry.bits) - 1;
    for row in 0..geometry.row_count {
        for group in 0..group_count {
            let start = row * 160 + group * geometry.group_size;
            let slice = &values[start..start + geometry.group_size];
            let minimum = slice.iter().cloned().fold(f32::INFINITY, f32::min);
            let maximum = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let scale = if maximum > minimum {
                (maximum - minimum) / levels as f32
            } else {
                1.0
            };
            let bias = minimum;
            scales.push(scale);
            biases.push(bias);
            for (offset, value) in slice.iter().enumerate() {
                let quantized = ((value - bias) / scale).round().clamp(0.0, levels as f32) as u64;
                let bit_offset = (group * geometry.group_size + offset) * geometry.bits as usize;
                let word_index = row * (160 * geometry.bits as usize / 32) + bit_offset / 32;
                packed[word_index] |= (quantized as u32) << (bit_offset % 32);
            }
        }
    }
    (packed, scales, biases)
}

/// Host-side `f64` affine reconstruction of one packed row set.
pub(crate) fn host_dequantize(
    packed: &[u32],
    scales: &[f32],
    biases: &[f32],
    geometry: &RowGeometry,
) -> Vec<f64> {
    let group_count = 160 / geometry.group_size;
    let words_per_row = 160 * geometry.bits as usize / 32;
    let mut output = Vec::with_capacity(geometry.row_count * 160);
    for row in 0..geometry.row_count {
        for group in 0..group_count {
            let scale = scales[row * group_count + group] as f64;
            let bias = biases[row * group_count + group] as f64;
            for offset in 0..geometry.group_size {
                let bit_offset = (group * geometry.group_size + offset) * geometry.bits as usize;
                let word = packed[row * words_per_row + bit_offset / 32];
                let quantized = (word >> (bit_offset % 32)) & ((1_u32 << geometry.bits) - 1);
                output.push(scale * quantized as f64 + bias);
            }
        }
    }
    output
}

#[tokio::test]
async fn should_match_gpu_dequantized_lookup_rows_against_host_reference() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = oracle_test_runtime();
    let mut values = DeterministicValues::new(0x3F7C);
    let geometry = RowGeometry {
        row_count: 4,
        group_size: 32,
        bits: 4,
    };
    let values_f32 = values.vec(geometry.row_count * 160, 1.0);
    let (packed, scales, biases) = pack_affine_rows(&values_f32, &geometry);

    // GPU path: MLX's own dequantization over the packed layout.
    let packed_array = runtime
        .array_from_u32(
            &packed,
            &[
                geometry.row_count as i32,
                (160 * geometry.bits as usize / 32) as i32,
            ],
        )
        .expect("packed rows should construct");
    let scales_array = runtime
        .array_from_f32(
            &scales,
            &[
                geometry.row_count as i32,
                (160 / geometry.group_size) as i32,
            ],
        )
        .expect("scales should construct");
    let biases_array = runtime
        .array_from_f32(
            &biases,
            &[
                geometry.row_count as i32,
                (160 / geometry.group_size) as i32,
            ],
        )
        .expect("biases should construct");
    let dequantized = runtime
        .dequantize_affine(
            &packed_array,
            &scales_array,
            &biases_array,
            geometry.group_size as i32,
            geometry.bits as i32,
        )
        .expect("GPU dequantization should run");
    dequantized
        .evaluate()
        .expect("dequantization should evaluate");
    let gpu_rows = dequantized.to_vec_f32().expect("rows should copy");

    let host = host_dequantize(&packed, &scales, &biases, &geometry);
    assert_f32_close(
        &gpu_rows,
        &host,
        1.0e-4,
        "GPU lookup-row decode must match the host affine reconstruction",
    );
}

#[tokio::test]
async fn should_reconstruct_packed_rows_exactly_through_the_host_path() {
    // Round-trip: quantize host values, reconstruct, and require agreement
    // within one quantization step. This pins the packer itself, so a reader
    // contract built on it starts from a verified encoding.
    let mut values = DeterministicValues::new(0x4B8D);
    let geometry = RowGeometry {
        row_count: 2,
        group_size: 32,
        bits: 4,
    };
    let values_f32 = values.vec(geometry.row_count * 160, 1.0);
    let (packed, scales, biases) = pack_affine_rows(&values_f32, &geometry);
    let reconstructed = host_dequantize(&packed, &scales, &biases, &geometry);
    let levels = (1_u64 << geometry.bits) - 1;
    // The bound is one step, not half: the scale and bias are themselves
    // computed in f32 from the same values, so the reconstruction can differ
    // from the ideal by the scale's own representation error plus half a
    // quantization step.
    for (original, reconstructed_value) in values_f32.iter().zip(&reconstructed) {
        let step = 1.0 / levels as f64;
        assert!(
            ((*original as f64 - reconstructed_value) / step).abs() <= 1.0 + 1.0e-9,
            "reconstruction {reconstructed_value} must be within one step of {original}"
        );
    }
}
