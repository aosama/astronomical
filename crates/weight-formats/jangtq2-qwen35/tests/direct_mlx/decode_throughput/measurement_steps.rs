//! The timing, memory-bound and fixture steps behind the decode and fused
//! projection measurements in the parent file, kept apart so the parent reads
//! as the list of measured cases.

use jangtq2_qwen35::mlx_decode;
use mlx_bridge::{MlxArray, MlxDtype};
use runtime_integration::MlxRuntime;

use super::{
    MEASURED_ITERATIONS, WARMUP_ITERATIONS, deterministic_packed_words, deterministic_row_scales,
    median_duration_millis,
};

/// Discarded decodes before the measured ones: the warmup stays outside every
/// timed region.
pub(super) fn warm_up_decoder(
    runtime: &MlxRuntime,
    packed_code_words: &MlxArray,
    row_scales: &MlxArray,
    bit_width: u32,
) {
    for _ in 0..WARMUP_ITERATIONS {
        drop(
            mlx_decode::decode_expert_weight_rows(
                runtime,
                packed_code_words,
                row_scales,
                bit_width,
            )
            .expect("the warmup decode should succeed"),
        );
    }
}

pub(super) struct DecodeMeasurement {
    pub(super) measured_durations_ms: Vec<f64>,
    pub(super) largest_measured_peak_growth_bytes: u64,
}

/// Times `MEASURED_ITERATIONS` decodes of one page; only the decode call sits
/// between the timer start and its read.
pub(super) fn measure_decode_iterations(
    runtime: &MlxRuntime,
    packed_code_words: &MlxArray,
    row_scales: &MlxArray,
    bit_width: u32,
    expected_decoded_shape: [i32; 3],
    case_name: &str,
) -> DecodeMeasurement {
    let mut measured_durations_ms = Vec::with_capacity(MEASURED_ITERATIONS);
    // Largest single-decode active-memory growth seen across the measured
    // iterations, each computed against its own baseline (see the loop).
    let mut largest_measured_peak_growth_bytes: u64 = 0;
    for measured_iteration in 0..MEASURED_ITERATIONS {
        // Peak-memory contract: one decode call must grow active memory by
        // no more than the persisted output it returns. If a change
        // reintroduces intermediate arrays, page admission would
        // under-reserve real serving: fail here instead.
        //
        // MLX's reset_peak_memory sets the counter to zero rather than to
        // the current active level, so a later raw reading is an absolute
        // high-water that still contains everything resident at reset; the
        // active baseline read at the same moment turns the reading into
        // the decode call's own growth. MLX also releases a dropped prior
        // output's buffer from a completion thread at an arbitrary later
        // point, so a stale output may still be counted active at reset.
        // Per-iteration rebaselining neutralizes that staleness in both
        // release orderings: if the stale output survives past the new
        // output's allocation, it cancels against the baseline; if it is
        // released first, the new output absorbs its place inside the
        // baseline and the growth reads lower. Asserting the maximum
        // across iterations samples both orderings, keeping the detection
        // floor at the slack rather than at a full absorbed output.
        runtime
            .reset_peak_memory()
            .expect("the MLX peak-memory counter should reset");
        let active_memory_bytes_before_decode = runtime
            .memory_snapshot()
            .expect("the MLX active-memory baseline should read")
            .active_memory_bytes() as u64;
        let started_at = std::time::Instant::now();
        let decoded_weight = mlx_decode::decode_expert_weight_rows(
            runtime,
            packed_code_words,
            row_scales,
            bit_width,
        )
        .expect("the measured decode should succeed");
        let elapsed_ms = started_at.elapsed().as_secs_f64() * 1_000.0;
        assert_eq!(
            decoded_weight.shape(),
            expected_decoded_shape,
            "the decoded page shape must match the format contract"
        );
        assert_eq!(decoded_weight.dtype(), MlxDtype::BFloat16);
        // Read before dropping the output so the persisted output itself is
        // part of the measured growth.
        let measured_peak_growth_bytes = runtime
            .memory_snapshot()
            .expect("the MLX peak-memory reading should read")
            .peak_memory_bytes() as u64
            - active_memory_bytes_before_decode;
        largest_measured_peak_growth_bytes =
            largest_measured_peak_growth_bytes.max(measured_peak_growth_bytes);
        measured_durations_ms.push(elapsed_ms);
        drop(decoded_weight);
        if measured_iteration == 0 {
            eprintln!(
                "[jangtq2-decode-perf:tier=operation] case={case_name} first_iteration_ms={elapsed_ms:.3}"
            );
        }
    }
    DecodeMeasurement {
        measured_durations_ms,
        largest_measured_peak_growth_bytes,
    }
}

pub(super) fn assert_decode_peak_growth_within_persisted_output(
    case_name: &str,
    decoded_element_count: u64,
    largest_measured_peak_growth_bytes: u64,
) {
    // The bound is the decode call's own priced allocation: the persisted
    // output. The packed source, row scales, and the per-width codebook
    // level tables are already resident when the measured decodes start
    // (the warmup uploads the tables; everything else sits inside the
    // subtracted baseline), just as page admission prices the source
    // once at page load rather than per decode. The 20 percent slack
    // covers MLX-internal buffers the decoder cannot see (for example
    // encoder residency); the Metal decoder has no workspace term, so
    // any temporaries an MLX upgrade introduced would fail here before
    // admission under-reserved real serving.
    let persisted_output_bytes = decoded_element_count * std::mem::size_of::<u16>() as u64;
    let bound_bytes_with_slack = persisted_output_bytes + persisted_output_bytes / 5;
    assert!(
        largest_measured_peak_growth_bytes <= bound_bytes_with_slack,
        "case={case_name} decode grew active memory by {largest_measured_peak_growth_bytes} bytes, above the persisted-output-plus-slack {bound_bytes_with_slack}: the page decoder allocated intermediate arrays the admission contract does not price"
    );
    eprintln!(
        "[jangtq2-decode-perf:tier=operation] case={case_name} peak_growth_bytes={largest_measured_peak_growth_bytes} persisted_output_bytes={persisted_output_bytes}"
    );
}

pub(super) struct FusedProjectionInputs {
    pub(super) packed_code_words: MlxArray,
    pub(super) row_scales: MlxArray,
    pub(super) hidden: MlxArray,
    pub(super) flat_route: MlxArray,
    pub(super) gather_route: MlxArray,
    pub(super) expanded_hidden: MlxArray,
}

/// Builds and materializes every array both projection paths read, so no
/// upload or conversion lands inside a timed region.
pub(super) fn synthetic_fused_projection_inputs(
    runtime: &MlxRuntime,
    packed_shape: &[i32],
    bit_width: u32,
) -> FusedProjectionInputs {
    let expert_count = packed_shape[0] as usize;
    let output_rows = packed_shape[1] as usize;
    let packed_words = packed_shape[2] as usize;
    let input_width = packed_words * 32 / bit_width as usize;
    let packed_code_words = MlxArray::from_u32(
        &deterministic_packed_words(
            u64::try_from(expert_count * 31 + packed_words).unwrap_or(0xA5A5),
            expert_count * output_rows * packed_words,
        ),
        packed_shape,
    )
    .expect("the synthetic packed page should form a valid uint32 tensor");
    let row_scales = runtime
        .astype(
            &runtime
                .array_from_f32(
                    &deterministic_row_scales(
                        u64::try_from(expert_count * 17 + output_rows).unwrap_or(0x5A5A),
                        expert_count * output_rows,
                    ),
                    &[expert_count as i32, output_rows as i32],
                )
                .expect("the synthetic row scales should upload"),
            MlxDtype::Float16,
        )
        .expect("the synthetic row scales should convert to float16");
    let activations_host: Vec<f32> = (0..input_width)
        .map(|element_index| ((element_index % 33) as f32 - 16.0) / 16.0)
        .collect();
    let hidden = runtime
        .astype(
            &runtime
                .array_from_f32(&activations_host, &[1, input_width as i32])
                .expect("the activation fixture should upload"),
            MlxDtype::BFloat16,
        )
        .expect("the activations should convert to bfloat16");
    let route_host: Vec<i32> = (0..expert_count as i32).collect();
    let flat_route = MlxArray::from_i32(&route_host, &[expert_count as i32])
        .expect("the flat route should upload");
    let gather_route = MlxArray::from_i32(&route_host, &[1, expert_count as i32])
        .expect("the gather route should upload");
    let expanded_hidden = runtime
        .reshape(&hidden, &[1, 1, 1, input_width as i32])
        .expect("the activations should expand for gather_mm");
    runtime
        .evaluate_arrays(&[
            &packed_code_words,
            &row_scales,
            &hidden,
            &flat_route,
            &gather_route,
            &expanded_hidden,
        ])
        .expect("the inputs should materialize before timing");
    FusedProjectionInputs {
        packed_code_words,
        row_scales,
        hidden,
        flat_route,
        gather_route,
        expanded_hidden,
    }
}

/// Warms up, then returns the median wall time of `MEASURED_ITERATIONS` runs of
/// one fully evaluated projection path.
pub(super) fn median_path_duration_ms(path: &dyn Fn()) -> f64 {
    for _ in 0..WARMUP_ITERATIONS {
        path();
    }
    let mut durations_ms = Vec::with_capacity(MEASURED_ITERATIONS);
    for _ in 0..MEASURED_ITERATIONS {
        let started_at = std::time::Instant::now();
        path();
        durations_ms.push(started_at.elapsed().as_secs_f64() * 1_000.0);
    }
    median_duration_millis(&durations_ms)
}
