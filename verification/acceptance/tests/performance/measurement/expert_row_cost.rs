//! What one more row costs in a mixture-of-experts feed-forward, at the
//! lowest layer that decides it: the gathered quantized projection.
//!
//! A draft round verifies several rows in one pass, and each row routes to
//! its own experts, so each row reads its own expert weights. This times the
//! 35B-A3B's real expert shapes (256 experts a layer, 8 a row, 2,048 wide in
//! and out, 512 wide inside) for 1 to 8 rows at 4-bit and 8-bit, through the
//! same `gather_expert_projection` both engines call, over several distinct
//! layers so every pass reads memory it has not just read. It prints each row
//! count's time scaled to the model's 40 layers, the cost of each added row
//! and the rate the weights were read at. Asserts nothing and needs no model.

use std::time::{Duration, Instant};

use astronomical_model_serving::{
    PerformanceAttribution, StackedExpertProjection, expert_assignment_order,
    gather_expert_projection,
};
use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;
use serial_test::serial;

use super::k_row_verify_cost::{
    MEASURED_PASS_COUNT, WARM_UP_PASS_COUNT, initialize_runtime, random_rows,
};

const EXPERT_COUNT: i32 = 256;
const EXPERTS_PER_ROW: usize = 8;
const HIDDEN_WIDTH: i32 = 2048;
const EXPERT_INNER_WIDTH: i32 = 512;
const QUANTIZATION_GROUP_SIZE: i32 = 64;
const MODEL_LAYER_COUNT: f64 = 40.0;
/// Distinct layers a pass walks: at 8-bit each holds about 0.8 GB, far past
/// what the processor keeps close, so every layer's read comes from memory.
const MEASURED_LAYER_COUNT: usize = 6;
const ROW_COUNTS: [usize; 5] = [1, 2, 3, 4, 8];
const QUANTIZATION_BIT_WIDTHS: [i32; 2] = [4, 8];
const BYTES_PER_GB: f64 = 1_073_741_824.0;

/// One stacked affine projection: `[experts, output, packed input]`.
struct ExpertProjection {
    packed_weights: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
    quantization_bits: i32,
}

struct ExpertLayer {
    up_projection: ExpertProjection,
    gate_projection: ExpertProjection,
    down_projection: ExpertProjection,
}

fn expert_projection(
    runtime: &MlxRuntime,
    random_key: &MlxArray,
    output_width: i32,
    input_width: i32,
    quantization_bits: i32,
) -> ExpertProjection {
    let full_precision = random_rows(
        runtime,
        random_key,
        EXPERT_COUNT * output_width,
        input_width,
    );
    let (packed_rows, scale_rows, bias_rows) = runtime
        .quantize_affine(&full_precision, QUANTIZATION_GROUP_SIZE, quantization_bits)
        .expect("the experts quantize");
    let stacked_shape = [EXPERT_COUNT, output_width, -1];
    let projection = ExpertProjection {
        packed_weights: runtime
            .reshape(&packed_rows, &stacked_shape)
            .expect("the packed experts stack"),
        scales: runtime
            .reshape(&scale_rows, &stacked_shape)
            .expect("the scales stack"),
        biases: runtime
            .reshape(&bias_rows, &stacked_shape)
            .expect("the biases stack"),
        quantization_bits,
    };
    runtime
        .evaluate_arrays(&[
            &projection.packed_weights,
            &projection.scales,
            &projection.biases,
        ])
        .expect("the stacked experts materialize");
    projection
}

fn expert_layer(
    runtime: &MlxRuntime,
    random_key: &MlxArray,
    quantization_bits: i32,
) -> ExpertLayer {
    ExpertLayer {
        up_projection: expert_projection(
            runtime,
            random_key,
            EXPERT_INNER_WIDTH,
            HIDDEN_WIDTH,
            quantization_bits,
        ),
        gate_projection: expert_projection(
            runtime,
            random_key,
            EXPERT_INNER_WIDTH,
            HIDDEN_WIDTH,
            quantization_bits,
        ),
        down_projection: expert_projection(
            runtime,
            random_key,
            HIDDEN_WIDTH,
            EXPERT_INNER_WIDTH,
            quantization_bits,
        ),
    }
}

/// Every row's experts for one layer, `[rows, experts a row]`: all distinct,
/// as rows of a draft tree mostly are, and different from layer to layer.
fn routed_expert_indices(runtime: &MlxRuntime, row_count: usize, layer_index: usize) -> MlxArray {
    // 7 shares no factor with 256, so the stride visits every expert once
    // before it repeats: 64 assignments never name one expert twice.
    const EXPERT_STRIDE: usize = 7;
    let assignment_count = row_count * EXPERTS_PER_ROW;
    let mut expert_ids = Vec::with_capacity(assignment_count);
    for assignment_index in 0..assignment_count {
        let expert_id =
            (layer_index * 31 + assignment_index * EXPERT_STRIDE) % EXPERT_COUNT as usize;
        expert_ids.push(expert_id as u32);
    }
    runtime
        .array_from_u32(&expert_ids, &[row_count as i32, EXPERTS_PER_ROW as i32])
        .expect("the routed experts build")
}

fn gathered(
    runtime: &MlxRuntime,
    activations: &MlxArray,
    projection: &ExpertProjection,
    expert_indices: &MlxArray,
) -> MlxArray {
    gather_expert_projection(
        runtime,
        activations,
        StackedExpertProjection::Affine {
            packed_weights: &projection.packed_weights,
            scales: &projection.scales,
            biases: &projection.biases,
            group_size: QUANTIZATION_GROUP_SIZE,
            bits: projection.quantization_bits,
        },
        expert_indices,
        expert_assignment_order(false, expert_indices.element_count(), EXPERT_COUNT as usize),
        &mut PerformanceAttribution::disabled(),
    )
    .expect("the gathered projection builds")
}

/// One layer's expert chain for every row, summed over each row's experts:
/// `[rows, hidden]`, the next layer's input.
fn layer_output(
    runtime: &MlxRuntime,
    hidden_rows: &MlxArray,
    layer: &ExpertLayer,
    expert_indices: &MlxArray,
) -> MlxArray {
    let row_count = hidden_rows.shape()[0];
    let expert_inputs = runtime
        .reshape(hidden_rows, &[row_count, 1, 1, HIDDEN_WIDTH])
        .expect("the rows expand");
    let up = gathered(
        runtime,
        &expert_inputs,
        &layer.up_projection,
        expert_indices,
    );
    let gate = gathered(
        runtime,
        &expert_inputs,
        &layer.gate_projection,
        expert_indices,
    );
    // The activation's cost is not the question; a product keeps both
    // projections in the graph.
    let activated = runtime.multiply(&gate, &up).expect("the product builds");
    let expert_outputs = gathered(runtime, &activated, &layer.down_projection, expert_indices);
    let per_row_outputs = runtime
        .sum_axis(&expert_outputs, 1, false)
        .expect("each row's experts sum");
    runtime
        .reshape(&per_row_outputs, &[row_count, HIDDEN_WIDTH])
        .expect("the rows flatten")
}

fn median_pass_time(runtime: &MlxRuntime, layers: &[ExpertLayer], row_count: usize) -> Duration {
    let random_key = runtime.random_key(99).expect("a key builds");
    let first_rows = random_rows(runtime, &random_key, row_count as i32, HIDDEN_WIDTH);
    let mut layer_expert_indices = Vec::with_capacity(layers.len());
    for layer_index in 0..layers.len() {
        layer_expert_indices.push(routed_expert_indices(runtime, row_count, layer_index));
    }
    let mut prepared_arrays: Vec<&MlxArray> = layer_expert_indices.iter().collect();
    prepared_arrays.push(&first_rows);
    runtime
        .evaluate_arrays(&prepared_arrays)
        .expect("the inputs materialize");
    let mut pass_times = Vec::with_capacity(MEASURED_PASS_COUNT);
    for pass_index in 0..WARM_UP_PASS_COUNT + MEASURED_PASS_COUNT {
        let pass_started_at = Instant::now();
        let mut hidden_rows =
            layer_output(runtime, &first_rows, &layers[0], &layer_expert_indices[0]);
        for (layer, expert_indices) in layers.iter().zip(&layer_expert_indices).skip(1) {
            hidden_rows = layer_output(runtime, &hidden_rows, layer, expert_indices);
        }
        runtime
            .evaluate_arrays(&[&hidden_rows])
            .expect("the pass evaluates");
        if pass_index >= WARM_UP_PASS_COUNT {
            pass_times.push(pass_started_at.elapsed());
        }
    }
    pass_times.sort();
    pass_times[pass_times.len() / 2]
}

/// The weight bytes one row's experts occupy in one layer.
fn routed_bytes_a_row(layer: &ExpertLayer) -> f64 {
    let mut layer_bytes = 0_usize;
    for projection in [
        &layer.up_projection,
        &layer.gate_projection,
        &layer.down_projection,
    ] {
        layer_bytes += projection.packed_weights.byte_count()
            + projection.scales.byte_count()
            + projection.biases.byte_count();
    }
    layer_bytes as f64 * EXPERTS_PER_ROW as f64 / f64::from(EXPERT_COUNT)
}

#[test]
#[ignore = "performance measurement on the GPU; run through cargo measure-expert-row-cost"]
#[serial]
fn should_measure_what_each_added_row_costs_in_the_expert_feed_forward() {
    let runtime = initialize_runtime();
    let random_key = runtime.random_key(2468).expect("a key builds");
    for quantization_bits in QUANTIZATION_BIT_WIDTHS {
        let mut layers = Vec::with_capacity(MEASURED_LAYER_COUNT);
        for _ in 0..MEASURED_LAYER_COUNT {
            layers.push(expert_layer(&runtime, &random_key, quantization_bits));
        }
        let model_row_bytes = routed_bytes_a_row(&layers[0]) * MODEL_LAYER_COUNT;
        let layer_scale = MODEL_LAYER_COUNT / MEASURED_LAYER_COUNT as f64;
        let mut one_row_milliseconds = 0.0;
        for row_count in ROW_COUNTS {
            let model_milliseconds =
                median_pass_time(&runtime, &layers, row_count).as_secs_f64() * 1e3 * layer_scale;
            if row_count == 1 {
                one_row_milliseconds = model_milliseconds;
            }
            let added_row_milliseconds = if row_count == 1 {
                model_milliseconds
            } else {
                (model_milliseconds - one_row_milliseconds) / (row_count - 1) as f64
            };
            eprintln!(
                "[expert-row-cost] {quantization_bits}-bit rows={row_count}: {model_milliseconds:6.2} ms over 40 layers, {added_row_milliseconds:5.2} ms each added row, {:.2} GB of experts a row read at {:.0} GB/s",
                model_row_bytes / BYTES_PER_GB,
                model_row_bytes * row_count as f64 / BYTES_PER_GB / (model_milliseconds / 1e3),
            );
        }
    }
}
