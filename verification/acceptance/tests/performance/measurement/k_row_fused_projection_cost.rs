//! What fusing a layer's projections saves on the register-ordered kernel,
//! priced before any model code changes.
//!
//! The same 16 distinct 27B layers are projected two ways at each row count:
//! one dispatch and one set of group sums per projection, as the model runs
//! today; and fused, as TensorFold stages its weights: a layer's mixer inputs
//! (DeltaNet q|k|v|z|b|a, attention q|k|v) as one weight, gate|up as one, and
//! the group sums of each distinct input computed once. The difference is the
//! dispatches and the input reads fusion removes.

use std::time::{Duration, Instant};

use astronomical_model_serving::row_batched_matmul::{
    ProjectionInput, RegisterOrderedWeight, RowBatchedMatmulKernels, register_ordered_projection,
};
use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;
use serial_test::serial;

use super::k_row_verify_cost::{
    HIDDEN_WIDTH, MEASURED_LAYER_COUNT, MEASURED_PASS_COUNT, MIXER_OUTPUT_WIDTH, MLP_WIDTH,
    MODEL_LAYER_COUNT, WARM_UP_PASS_COUNT, initialize_runtime,
    quantize_random_weight_in_row_slices, random_rows,
};

const ROW_COUNTS: [i32; 3] = [1, 8, 16];
/// DeltaNet's fused mixer input: q|k|v (10,240), z (6,144), b and a (48 each).
const DELTA_NET_MIXER_WIDTHS: [i32; 4] = [10240, 6144, 48, 48];
/// Attention's: q with its gate (12,288), k and v (1,024 each).
const ATTENTION_MIXER_WIDTHS: [i32; 3] = [12288, 1024, 1024];

/// One layer's projections, each width a separate weight or one fused weight.
struct ProjectionLayer {
    mixer_inputs: Vec<RegisterOrderedWeight>,
    mixer_output: RegisterOrderedWeight,
    /// Gate then up, or gate|up fused.
    feed_forward_inputs: Vec<RegisterOrderedWeight>,
    feed_forward_output: RegisterOrderedWeight,
}

#[test]
#[ignore = "performance measurement on the GPU; run through cargo measure-k-row-verify-cost"]
#[serial]
fn should_price_fused_projections_and_shared_group_sums_on_the_register_kernel() {
    let runtime = initialize_runtime();
    let kernels = RowBatchedMatmulKernels::declare().expect("the kernels are declared");
    let random_key = runtime.random_key(4321).expect("a key builds");
    let mut separate_layers = Vec::with_capacity(MEASURED_LAYER_COUNT);
    let mut fused_layers = Vec::with_capacity(MEASURED_LAYER_COUNT);
    for layer_index in 0..MEASURED_LAYER_COUNT {
        let mixer_widths: &[i32] = if layer_index % 4 == 3 {
            &ATTENTION_MIXER_WIDTHS
        } else {
            &DELTA_NET_MIXER_WIDTHS
        };
        let converter = Converter {
            runtime: &runtime,
            kernels: &kernels,
            random_key: &random_key,
        };
        separate_layers.push(converter.layer(mixer_widths, false));
        fused_layers.push(converter.layer(mixer_widths, true));
    }
    for row_count in ROW_COUNTS {
        let separate = median_pass_time(
            &runtime,
            &kernels,
            &random_key,
            row_count,
            &separate_layers,
            false,
        );
        let fused = median_pass_time(
            &runtime,
            &kernels,
            &random_key,
            row_count,
            &fused_layers,
            true,
        );
        let layer_scale = MODEL_LAYER_COUNT / MEASURED_LAYER_COUNT as f64;
        eprintln!(
            "[k-row] fused K={row_count:>2}: separate {:6.1} ms, fused with shared sums {:6.1} ms over 64 layers",
            separate.as_secs_f64() * 1000.0 * layer_scale,
            fused.as_secs_f64() * 1000.0 * layer_scale,
        );
    }
}

struct Converter<'measurement> {
    runtime: &'measurement MlxRuntime,
    kernels: &'measurement RowBatchedMatmulKernels,
    random_key: &'measurement MlxArray,
}

impl Converter<'_> {
    /// A random 4-bit `[output, input]` weight in register order. Widths
    /// that do not tile 32 columns (the lone 48-wide gates) are rounded up,
    /// as their real fused stack's width (6,240) tiles.
    fn weight(&self, output_width: i32, input_width: i32) -> RegisterOrderedWeight {
        let tiled_width = (output_width + 31) / 32 * 32;
        let (packed_weight, scales, biases) = quantize_random_weight_in_row_slices(
            self.runtime,
            self.random_key,
            tiled_width,
            input_width,
        );
        let converted = RegisterOrderedWeight::convert(
            self.runtime,
            self.kernels,
            &packed_weight,
            &scales,
            &biases,
        )
        .expect("the weight converts");
        self.runtime
            .evaluate_arrays(&[&converted.codes, &converted.scale_bias_words])
            .expect("the converted weight materializes");
        converted
    }

    fn layer(&self, mixer_widths: &[i32], is_fused: bool) -> ProjectionLayer {
        let mixer_inputs = if is_fused {
            vec![self.weight(mixer_widths.iter().sum(), HIDDEN_WIDTH)]
        } else {
            let mut separate = Vec::with_capacity(mixer_widths.len());
            for mixer_width in mixer_widths {
                separate.push(self.weight(*mixer_width, HIDDEN_WIDTH));
            }
            separate
        };
        let feed_forward_inputs = if is_fused {
            vec![self.weight(2 * MLP_WIDTH, HIDDEN_WIDTH)]
        } else {
            vec![
                self.weight(MLP_WIDTH, HIDDEN_WIDTH),
                self.weight(MLP_WIDTH, HIDDEN_WIDTH),
            ]
        };
        ProjectionLayer {
            mixer_inputs,
            mixer_output: self.weight(HIDDEN_WIDTH, MIXER_OUTPUT_WIDTH),
            feed_forward_inputs,
            feed_forward_output: self.weight(HIDDEN_WIDTH, MLP_WIDTH),
        }
    }
}

/// Every projection of every layer once, outputs summed in so none is
/// dropped. `shares_inputs` prepares each distinct input once; otherwise
/// every projection prepares its own, as the model does today.
fn stack_pass(
    runtime: &MlxRuntime,
    kernels: &RowBatchedMatmulKernels,
    layers: &[ProjectionLayer],
    rows: [&MlxArray; 3],
    shares_inputs: bool,
) -> MlxArray {
    let [hidden_rows, mixer_value_rows, feed_forward_rows] = rows;
    let mut projected_sums = Vec::new();
    for layer in layers {
        for (weights, input_rows) in [
            (&layer.mixer_inputs, hidden_rows),
            (&layer.feed_forward_inputs, hidden_rows),
        ] {
            let shared = ProjectionInput::prepare(runtime, kernels, input_rows).expect("prepares");
            for weight in weights.iter() {
                let input = if shares_inputs {
                    None
                } else {
                    Some(ProjectionInput::prepare(runtime, kernels, input_rows).expect("prepares"))
                };
                let projected = register_ordered_projection(
                    runtime,
                    kernels,
                    input.as_ref().unwrap_or(&shared),
                    weight,
                )
                .expect("projects");
                projected_sums.push(runtime.sum(&projected, false).expect("reduces"));
            }
        }
        for (weight, input_rows) in [
            (&layer.mixer_output, mixer_value_rows),
            (&layer.feed_forward_output, feed_forward_rows),
        ] {
            let input = ProjectionInput::prepare(runtime, kernels, input_rows).expect("prepares");
            let projected =
                register_ordered_projection(runtime, kernels, &input, weight).expect("projects");
            projected_sums.push(runtime.sum(&projected, false).expect("reduces"));
        }
    }
    let references: Vec<&MlxArray> = projected_sums.iter().collect();
    runtime
        .sum(&runtime.stack_axis(&references, 0).expect("stacks"), false)
        .expect("reduces")
}

fn median_pass_time(
    runtime: &MlxRuntime,
    kernels: &RowBatchedMatmulKernels,
    random_key: &MlxArray,
    row_count: i32,
    layers: &[ProjectionLayer],
    shares_inputs: bool,
) -> Duration {
    let hidden_rows = random_rows(runtime, random_key, row_count, HIDDEN_WIDTH);
    let mixer_value_rows = random_rows(runtime, random_key, row_count, MIXER_OUTPUT_WIDTH);
    let feed_forward_rows = random_rows(runtime, random_key, row_count, MLP_WIDTH);
    runtime
        .evaluate_arrays(&[&hidden_rows, &mixer_value_rows, &feed_forward_rows])
        .expect("input rows materialize");
    let mut pass_times = Vec::with_capacity(MEASURED_PASS_COUNT);
    for pass_index in 0..WARM_UP_PASS_COUNT + MEASURED_PASS_COUNT {
        let started_at = Instant::now();
        let pass_output = stack_pass(
            runtime,
            kernels,
            layers,
            [&hidden_rows, &mixer_value_rows, &feed_forward_rows],
            shares_inputs,
        );
        runtime
            .evaluate_arrays(&[&pass_output])
            .expect("the pass evaluates");
        if pass_index >= WARM_UP_PASS_COUNT {
            pass_times.push(started_at.elapsed());
        }
    }
    pass_times.sort();
    pass_times[pass_times.len() / 2]
}
