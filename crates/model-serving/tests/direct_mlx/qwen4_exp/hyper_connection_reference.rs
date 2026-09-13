//! Oracle reference for hyper-connection stream mixing on the GPU.
//!
//! The pinned algebra lives in the hermetic `stream_algebra` owner as pure
//! host arithmetic. This module proves the same algebra composed from raw
//! MLX operations produces the same result on the GPU, so when the production
//! route arrives — compiled graphs, custom kernels, or plain ops — it has a
//! GPU-side reference that was itself checked against host math. A production
//! route is then parity-tested against these functions rather than against
//! itself.

use astronomical_runtime_integration::MlxRuntime;

use super::{DeterministicValues, assert_f32_close, f32_array, oracle_test_runtime};

/// Mixing geometry for one reference row.
pub(crate) struct StreamGeometry {
    pub stream_count: usize,
    pub stream_width: usize,
    pub low_rank: usize,
    pub epsilon: f64,
}

fn hyper_width(geometry: &StreamGeometry) -> usize {
    geometry.stream_count * geometry.stream_width
}

/// Host-side `f64` gated-residual mix, transcribed from the pinned algebra:
/// grouped RMSNorm, project down, SiLU scaled by one over the stream count,
/// project up, sigmoid, multiply into normalized streams, average over
/// streams.
pub(crate) fn host_gated_mix(
    hyper_input: &[f32],
    norm_weights: &[f32],
    down: &[f32],
    up: &[f32],
    geometry: &StreamGeometry,
) -> (Vec<f64>, Vec<f64>) {
    let width = geometry.stream_width;
    let hyper = hyper_width(geometry);
    let mut normalized = Vec::with_capacity(hyper);
    for stream in 0..geometry.stream_count {
        let base = stream * width;
        let variance: f64 = (base..base + width)
            .map(|index| hyper_input[index] as f64 * hyper_input[index] as f64)
            .sum::<f64>()
            / width as f64;
        let scale = 1.0 / (variance + geometry.epsilon).sqrt();
        for offset in 0..width {
            let index = base + offset;
            normalized.push(hyper_input[index] as f64 * scale * (1.0 + norm_weights[index] as f64));
        }
    }
    let stream_count = geometry.stream_count as f64;
    let mut hidden = Vec::with_capacity(geometry.low_rank);
    for rank in 0..geometry.low_rank {
        let row = &down[rank * hyper..(rank + 1) * hyper];
        let dot: f64 = row
            .iter()
            .zip(&normalized)
            .map(|(weight, value)| (*weight as f64) * value)
            .sum::<f64>()
            / stream_count;
        hidden.push(dot * sigmoid(dot));
    }
    let mut gate = Vec::with_capacity(hyper);
    for stream in 0..hyper {
        let mut product = 0.0_f64;
        for rank in 0..geometry.low_rank {
            product += up[stream * geometry.low_rank + rank] as f64 * hidden[rank];
        }
        gate.push(sigmoid(product));
    }
    let mut mixed = vec![0.0_f64; width];
    for stream in 0..geometry.stream_count {
        for offset in 0..width {
            let index = stream * width + offset;
            mixed[offset] += gate[index] * normalized[index];
        }
    }
    for value in &mut mixed {
        *value /= stream_count;
    }
    (mixed, normalized)
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

/// GPU-side gated mix composed from raw MLX operations, mirroring the host
/// order operation by operation.
pub(crate) fn gpu_gated_mix(
    runtime: &MlxRuntime,
    hyper_input: &[f32],
    norm_weights: &[f32],
    down: &[f32],
    up: &[f32],
    geometry: &StreamGeometry,
) -> Result<(Vec<f32>, Vec<f32>), astronomical_runtime_integration::MlxRuntimeError> {
    let hyper = hyper_width(geometry);
    let width = geometry.stream_width;
    let input = f32_array(runtime, hyper_input, &[hyper as i32])?;
    let norm = f32_array(runtime, norm_weights, &[hyper as i32])?;
    // Grouped RMSNorm over each stream: variance per group, then the
    // per-element affine of one plus the weight.
    let mut normalized_parts = Vec::with_capacity(geometry.stream_count);
    for stream in 0..geometry.stream_count {
        let slice = runtime.slice(
            &input,
            &[(stream * width) as i32],
            &[((stream + 1) * width) as i32],
            &[1],
        )?;
        let weight_slice = runtime.slice(
            &norm,
            &[(stream * width) as i32],
            &[((stream + 1) * width) as i32],
            &[1],
        )?;
        // The checkpoint stores the affine as an additive offset around one,
        // while MLX's fused RMS normalization multiplies by the weight
        // directly, so the reference adds one before the fused op.
        let one = runtime.array_from_f32(&vec![1.0_f32; width], &[width as i32])?;
        let shifted_weight = runtime.add(&weight_slice, &one)?;
        let normalized = runtime.rms_norm(&slice, &shifted_weight, geometry.epsilon as f32)?;
        normalized_parts.push(normalized);
    }
    let stacked = runtime.stack_axis(&normalized_parts.iter().collect::<Vec<_>>(), 0)?;
    let stacked_shape = stacked.shape();
    let element_count: i32 = stacked_shape.iter().product();
    let normalized = runtime.reshape(&stacked, &[element_count])?;
    let down_array = f32_array(runtime, down, &[geometry.low_rank as i32, hyper as i32])?;
    let column = runtime.reshape(&normalized, &[hyper as i32, 1])?;
    let hidden = runtime.matmul(&down_array, &column)?;
    let hidden = runtime.divide(
        &hidden,
        &runtime.array_from_f32(&[geometry.stream_count as f32], &[1])?,
    )?;
    // SiLU: value times sigmoid of value.
    let activated = runtime.sigmoid(&hidden)?;
    let hidden = runtime.multiply(&hidden, &activated)?;
    let up_array = f32_array(runtime, up, &[hyper as i32, geometry.low_rank as i32])?;
    let hidden_column = runtime.reshape(&hidden, &[geometry.low_rank as i32, 1])?;
    let gate = runtime.matmul(&up_array, &hidden_column)?;
    let gate = runtime.sigmoid(&gate)?;
    let gate = runtime.reshape(&gate, &[hyper as i32])?;
    // Multiply the gate into the normalized streams and average over streams.
    let gated = runtime.multiply(&gate, &normalized)?;
    let gated = runtime.reshape(&gated, &[geometry.stream_count as i32, width as i32])?;
    let mean = runtime.sum_axis(&gated, 0, false)?;
    let mixed = runtime.divide(
        &mean,
        &runtime.array_from_f32(&[geometry.stream_count as f32], &[1])?,
    )?;
    mixed.evaluate()?;
    normalized.evaluate()?;
    Ok((mixed.to_vec_f32()?, normalized.to_vec_f32()?))
}

#[tokio::test]
async fn should_match_gpu_stream_mixing_against_the_pinned_host_algebra() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = oracle_test_runtime();
    let mut values = DeterministicValues::new(0x5C9E);
    for geometry in [
        StreamGeometry {
            stream_count: 2,
            stream_width: 8,
            low_rank: 3,
            epsilon: 1.0e-6,
        },
        StreamGeometry {
            stream_count: 4,
            stream_width: 6,
            low_rank: 5,
            epsilon: 1.0e-6,
        },
    ] {
        let hyper = hyper_width(&geometry);
        let hyper_input = values.vec(hyper, 1.0);
        let norm_weights = values.vec(hyper, 0.25);
        let down = values.vec(geometry.low_rank * hyper, 0.5);
        let up = values.vec(hyper * geometry.low_rank, 0.5);
        let (gpu_mixed, gpu_normalized) =
            gpu_gated_mix(&runtime, &hyper_input, &norm_weights, &down, &up, &geometry)
                .expect("GPU mixing should run");
        let (host_mixed, host_normalized) =
            host_gated_mix(&hyper_input, &norm_weights, &down, &up, &geometry);
        assert_f32_close(
            &gpu_normalized,
            &host_normalized,
            1.0e-4,
            "GPU grouped normalization must match the host algebra",
        );
        assert_f32_close(
            &gpu_mixed,
            &host_mixed,
            1.0e-4,
            "GPU gated mixing must match the host algebra",
        );
    }
}

#[tokio::test]
async fn should_match_the_hermetic_owner_values_on_the_gpu() {
    // The hermetic contract pinned exact f32 values for a fixed 2-stream,
    // 4-wide, rank-3 input. The GPU composition must reproduce them, which
    // ties the hermetic owner, this oracle, and the future production route
    // to one set of numbers.
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = oracle_test_runtime();
    let geometry = StreamGeometry {
        stream_count: 2,
        stream_width: 4,
        low_rank: 3,
        epsilon: 1.0e-6,
    };
    let hyper_input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let norm_weights = vec![0.1, -0.2, 0.3, 0.0, 0.5, -0.5, 0.25, 0.75];
    let down = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let up = vec![
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let (gpu_mixed, _) =
        gpu_gated_mix(&runtime, &hyper_input, &norm_weights, &down, &up, &geometry)
            .expect("GPU mixing should run");
    assert_f32_close(
        &gpu_mixed,
        &[0.456_878_934, 0.304_467_565, 0.874_527_157, 1.137_607_931],
        1.0e-5,
        "GPU mixing must reproduce the hermetic contract values",
    );
}
