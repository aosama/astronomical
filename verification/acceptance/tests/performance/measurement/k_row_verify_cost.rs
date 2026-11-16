//! What one pass over K token rows costs against a pass over one row, on the
//! dense Qwen3.8 27B's real 4-bit projection shapes, through MLX's affine
//! quantized matmul and through Astronomical's row-batched kernel.
//!
//! Drafting only pays when checking K drafted tokens costs close to writing
//! one, which holds only while a pass is bound by reading the weights. This is
//! the bottom of the call stack: the projections, nothing else. Values are
//! random because a matmul's cost does not depend on them.
//!
//! The stack holds distinct weights for every layer. Calling one matrix again
//! would read it from the GPU's system-level memory instead of DRAM and report
//! a pass faster than serving ever sees, so every layer here owns its bytes and
//! a pass streams the whole stack once, like a real decode step.

use std::time::{Duration, Instant};

use astronomical_model_serving::row_batched_matmul::{
    RegisterOrderedWeight, RowBatchedMatmulKernels, register_ordered_matmul,
    row_batched_quantized_matmul, row_batched_quantized_matmul_accepts,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_bridge::{MlxArray, MlxDtype};
use serial_test::serial;

pub(super) const HIDDEN_WIDTH: i32 = 5120;
pub(super) const MLP_WIDTH: i32 = 17408;
/// Gated DeltaNet value width and the attention output width share 6144.
pub(super) const MIXER_OUTPUT_WIDTH: i32 = 6144;
const VOCABULARY_SIZE: i32 = 248_320;
const QUANTIZATION_GROUP_SIZE: i32 = 64;
const QUANTIZATION_BITS: i32 = 4;
/// The 27B repeats three DeltaNet layers and one attention layer; 16 distinct
/// layers (3.19 GB) is far past the GPU's system-level memory, so every pass
/// reads DRAM. Both kernels read the same arrays, so each layer is held once.
pub(super) const MEASURED_LAYER_COUNT: usize = 16;
pub(super) const MODEL_LAYER_COUNT: f64 = 64.0;
const ROW_COUNTS: [i32; 9] = [1, 2, 4, 8, 12, 16, 32, 64, 128];
pub(super) const WARM_UP_PASS_COUNT: usize = 2;
pub(super) const MEASURED_PASS_COUNT: usize = 5;
const ACTIVE_MEMORY_LIMIT_BYTES: usize = 12 * 1024 * 1024 * 1024;
const MLX_BUFFER_CACHE_LIMIT_BYTES: usize = 1024 * 1024 * 1024;
const BYTES_PER_GB: f64 = 1_073_741_824.0;

/// Which kernel a pass multiplies with. A projection the row-batched kernel
/// refuses (the 48-wide DeltaNet gates) takes MLX's, as serving would.
#[derive(Clone, Copy)]
enum MatmulPath<'kernels> {
    Mlx,
    RowBatched(&'kernels RowBatchedMatmulKernels),
    /// The weights converted once into TensorFold's register order.
    RegisterOrdered(&'kernels RowBatchedMatmulKernels),
}

impl MatmulPath<'_> {
    fn label(self) -> &'static str {
        match self {
            MatmulPath::Mlx => "MLX        ",
            MatmulPath::RowBatched(_) => "row-batched",
            MatmulPath::RegisterOrdered(_) => "register   ",
        }
    }
}

/// One packed 4-bit projection, `[output, input]` before packing, in MLX's
/// layout, which both paths read as stored.
struct PackedProjection {
    weight: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
    /// The same weight in register order, when its shape tiles.
    register_ordered: Option<RegisterOrderedWeight>,
    stored_bytes: usize,
}

/// One decoder layer's projections in the 27B's shapes.
struct MeasuredLayer {
    /// Projections reading the hidden row: DeltaNet qkv, z, a, b or attention q, k, v.
    mixer_inputs: Vec<PackedProjection>,
    mixer_output: PackedProjection,
    mlp_gate: PackedProjection,
    mlp_up: PackedProjection,
    mlp_down: PackedProjection,
}

pub(super) fn initialize_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(ACTIVE_MEMORY_LIMIT_BYTES, MLX_BUFFER_CACHE_LIMIT_BYTES)
            .expect("memory limits are valid"),
    )
    .expect("the MLX runtime initializes")
}

#[test]
#[ignore = "performance measurement on the GPU; run through cargo measure-k-row-verify-cost"]
#[serial]
fn should_measure_k_row_projection_cost_against_one_row_for_the_dense_27b() {
    let runtime = initialize_runtime();
    let kernels = RowBatchedMatmulKernels::declare().expect("the kernels are declared");
    let random_key = runtime.random_key(1234).expect("a key builds");
    let mut layers = Vec::with_capacity(MEASURED_LAYER_COUNT);
    for layer_index in 0..MEASURED_LAYER_COUNT {
        layers.push(build_layer(
            &runtime,
            &kernels,
            &random_key,
            layer_index % 4 == 3,
        ));
    }
    let output_head = pack_projection(
        &runtime,
        &kernels,
        &random_key,
        VOCABULARY_SIZE,
        HIDDEN_WIDTH,
    );
    let stack_bytes: usize = layers.iter().map(layer_stored_bytes).sum();
    eprintln!(
        "[k-row] {MEASURED_LAYER_COUNT} distinct layers, {:.2} GB of packed weights a pass",
        stack_bytes as f64 / BYTES_PER_GB
    );
    for matmul_path in [
        MatmulPath::Mlx,
        MatmulPath::RowBatched(&kernels),
        MatmulPath::RegisterOrdered(&kernels),
    ] {
        let mut one_row_times = None;
        for row_count in ROW_COUNTS {
            let stack_time = median_pass_time(
                &runtime,
                &random_key,
                row_count,
                &MeasuredPass::LayerStack(&layers),
                matmul_path,
            );
            let head_time = median_pass_time(
                &runtime,
                &random_key,
                row_count,
                &MeasuredPass::OutputHead(&output_head),
                matmul_path,
            );
            let (one_row_stack, one_row_head) =
                *one_row_times.get_or_insert((stack_time, head_time));
            report_row_count(
                matmul_path,
                row_count,
                (stack_time, one_row_stack),
                (head_time, one_row_head),
                stack_bytes,
            );
        }
    }
}

/// Every distinct projection shape of the 27B as `(output, input)`. On M5,
/// MLX runs its one-row kernel once per row up to 12 rows for these widths
/// and its matrix-unit kernel from 13 rows, so 12 and 13 straddle the switch.
const PROJECTION_SHAPES: [(i32, i32); 7] = [
    (10240, HIDDEN_WIDTH),
    (6144, HIDDEN_WIDTH),
    (12288, HIDDEN_WIDTH),
    (1024, HIDDEN_WIDTH),
    (HIDDEN_WIDTH, MIXER_OUTPUT_WIDTH),
    (MLP_WIDTH, HIDDEN_WIDTH),
    (HIDDEN_WIDTH, MLP_WIDTH),
];
const PER_SHAPE_ROW_COUNTS: [i32; 6] = [1, 8, 12, 13, 16, 32];

#[test]
#[ignore = "performance measurement on the GPU; run through cargo measure-k-row-verify-cost"]
#[serial]
fn should_measure_k_row_cost_against_one_row_for_each_projection_shape() {
    let runtime = initialize_runtime();
    let kernels = RowBatchedMatmulKernels::declare().expect("the kernels are declared");
    let random_key = runtime.random_key(1234).expect("a key builds");
    for (output_width, input_width) in PROJECTION_SHAPES {
        let mut shape_copies = Vec::with_capacity(MEASURED_LAYER_COUNT);
        for _ in 0..MEASURED_LAYER_COUNT {
            shape_copies.push(pack_projection(
                &runtime,
                &kernels,
                &random_key,
                output_width,
                input_width,
            ));
        }
        for matmul_path in [
            MatmulPath::Mlx,
            MatmulPath::RowBatched(&kernels),
            MatmulPath::RegisterOrdered(&kernels),
        ] {
            let mut row_cost_line = format!(
                "[k-row] {} {output_width}x{input_width}:",
                matmul_path.label()
            );
            let mut one_row_time = None;
            for row_count in PER_SHAPE_ROW_COUNTS {
                let measured_pass = MeasuredPass::ShapeCopies(&shape_copies, input_width);
                let pass_time = median_pass_time(
                    &runtime,
                    &random_key,
                    row_count,
                    &measured_pass,
                    matmul_path,
                );
                let one_row = *one_row_time.get_or_insert(pass_time);
                row_cost_line.push_str(&format!(
                    " K={row_count} {:.2}x",
                    pass_time.as_secs_f64() / one_row.as_secs_f64()
                ));
            }
            eprintln!(
                "{row_cost_line} (one row {:.2} ms for 16 copies)",
                one_row_time.expect("one row was timed").as_secs_f64() * 1000.0
            );
        }
    }
}

fn build_layer(
    runtime: &MlxRuntime,
    kernels: &RowBatchedMatmulKernels,
    random_key: &MlxArray,
    is_attention: bool,
) -> MeasuredLayer {
    let mixer_input_widths: &[i32] = if is_attention {
        &[12288, 1024, 1024]
    } else {
        &[10240, 6144, 48, 48]
    };
    let mut mixer_inputs = Vec::with_capacity(mixer_input_widths.len());
    for output_width in mixer_input_widths {
        mixer_inputs.push(pack_projection(
            runtime,
            kernels,
            random_key,
            *output_width,
            HIDDEN_WIDTH,
        ));
    }
    MeasuredLayer {
        mixer_inputs,
        mixer_output: pack_projection(
            runtime,
            kernels,
            random_key,
            HIDDEN_WIDTH,
            MIXER_OUTPUT_WIDTH,
        ),
        mlp_gate: pack_projection(runtime, kernels, random_key, MLP_WIDTH, HIDDEN_WIDTH),
        mlp_up: pack_projection(runtime, kernels, random_key, MLP_WIDTH, HIDDEN_WIDTH),
        mlp_down: pack_projection(runtime, kernels, random_key, HIDDEN_WIDTH, MLP_WIDTH),
    }
}

fn pack_projection(
    runtime: &MlxRuntime,
    kernels: &RowBatchedMatmulKernels,
    random_key: &MlxArray,
    output_width: i32,
    input_width: i32,
) -> PackedProjection {
    let (weight, scales, biases) =
        quantize_random_weight_in_row_slices(runtime, random_key, output_width, input_width);
    let register_ordered =
        RegisterOrderedWeight::convert(runtime, kernels, &weight, &scales, &biases).ok();
    if let Some(converted) = &register_ordered {
        runtime
            .evaluate_arrays(&[&converted.codes, &converted.scale_bias_words])
            .expect("the converted weight materializes");
    }
    let element_count = output_width as usize * input_width as usize;
    let group_count = element_count / QUANTIZATION_GROUP_SIZE as usize;
    PackedProjection {
        weight,
        scales,
        biases,
        register_ordered,
        // Four bits a value, plus a bf16 scale and bias for every group.
        stored_bytes: element_count / 2 + group_count * 4,
    }
}

/// A random 4-bit weight as MLX packs it: `(weight, scales, biases)`.
///
/// Quantizing builds float32 temporaries of the whole weight, which for the
/// 248,320-row head would be about 5 GB, so large weights are quantized a
/// slice of rows at a time and the packed slices joined; rows are independent.
pub(super) fn quantize_random_weight_in_row_slices(
    runtime: &MlxRuntime,
    random_key: &MlxArray,
    output_width: i32,
    input_width: i32,
) -> (MlxArray, MlxArray, MlxArray) {
    let mut slice_count = (output_width / 32_768).max(1);
    while output_width % slice_count != 0 {
        slice_count += 1;
    }
    let (mut weight_slices, mut scale_slices, mut bias_slices) =
        (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..slice_count {
        let full_precision_slice =
            random_rows(runtime, random_key, output_width / slice_count, input_width);
        let (weight_slice, scale_slice, bias_slice) = runtime
            .quantize_affine(
                &full_precision_slice,
                QUANTIZATION_GROUP_SIZE,
                QUANTIZATION_BITS,
            )
            .expect("the weight quantizes");
        runtime
            .evaluate_arrays(&[&weight_slice, &scale_slice, &bias_slice])
            .expect("the packed slice materializes");
        weight_slices.push(weight_slice);
        scale_slices.push(scale_slice);
        bias_slices.push(bias_slice);
    }
    let weight = join_row_slices(runtime, &weight_slices);
    let scales = join_row_slices(runtime, &scale_slices);
    let biases = join_row_slices(runtime, &bias_slices);
    runtime
        .evaluate_arrays(&[&weight, &scales, &biases])
        .expect("the packed weight materializes");
    (weight, scales, biases)
}

fn join_row_slices(runtime: &MlxRuntime, row_slices: &[MlxArray]) -> MlxArray {
    let slice_references: Vec<&MlxArray> = row_slices.iter().collect();
    runtime
        .concatenate_axis(&slice_references, 0)
        .expect("the slices join")
}

pub(super) fn random_rows(
    runtime: &MlxRuntime,
    random_key: &MlxArray,
    row_count: i32,
    width: i32,
) -> MlxArray {
    runtime
        .random_normal(
            &[row_count, width],
            MlxDtype::BFloat16,
            0.0,
            0.02,
            random_key,
        )
        .expect("random rows build")
}

fn project(
    runtime: &MlxRuntime,
    matmul_path: MatmulPath<'_>,
    rows: &MlxArray,
    projection: &PackedProjection,
) -> MlxArray {
    // A shape the row-batched kernels cannot tile (the 48-wide DeltaNet gates)
    // stays on MLX, as serving does.
    if let MatmulPath::RegisterOrdered(kernels) = matmul_path
        && let Some(register_ordered) = &projection.register_ordered
    {
        let row_count: i32 = rows.shape()[..rows.shape().len() - 1].iter().product();
        if row_count <= 128 {
            return register_ordered_matmul(runtime, kernels, rows, register_ordered)
                .expect("the register-ordered matmul builds");
        }
    }
    if let MatmulPath::RowBatched(kernels) = matmul_path
        && row_batched_quantized_matmul_accepts(
            rows,
            &projection.weight,
            &projection.scales,
            &projection.biases,
        )
    {
        return row_batched_quantized_matmul(
            runtime,
            kernels,
            rows,
            &projection.weight,
            &projection.scales,
            &projection.biases,
        )
        .expect("the row-batched matmul builds");
    }
    runtime
        .quantized_matmul_affine(
            rows,
            &projection.weight,
            &projection.scales,
            &projection.biases,
            true,
            QUANTIZATION_GROUP_SIZE,
            QUANTIZATION_BITS,
        )
        .expect("the quantized matmul builds")
}

/// Every projection of every layer once. The MLP output feeds the next layer
/// so the passes are ordered like a real forward; the mixer outputs are summed
/// in so none is dropped as unused.
fn stack_pass(
    runtime: &MlxRuntime,
    matmul_path: MatmulPath<'_>,
    layers: &[MeasuredLayer],
    hidden_rows: &MlxArray,
    mixer_value_rows: &MlxArray,
) -> MlxArray {
    let mut residual_rows = runtime
        .multiply_scalar(hidden_rows, 1.0)
        .expect("the residual starts from the input rows");
    for layer in layers {
        let mut mixer_sum: Option<MlxArray> = None;
        for mixer_input in &layer.mixer_inputs {
            let projected = runtime
                .sum(
                    &project(runtime, matmul_path, &residual_rows, mixer_input),
                    false,
                )
                .expect("the mixer projection reduces");
            mixer_sum = Some(match mixer_sum {
                Some(running_sum) => runtime.add(&running_sum, &projected).expect("sums add"),
                None => projected,
            });
        }
        let mixed = project(runtime, matmul_path, mixer_value_rows, &layer.mixer_output);
        let gate = runtime
            .silu(&project(runtime, matmul_path, &mixed, &layer.mlp_gate))
            .expect("the gate activates");
        let gated = runtime
            .multiply(&gate, &project(runtime, matmul_path, &mixed, &layer.mlp_up))
            .expect("the gate multiplies");
        let mlp_rows = project(runtime, matmul_path, &gated, &layer.mlp_down);
        let with_mixer = runtime
            .add(
                &mlp_rows,
                mixer_sum.as_ref().expect("every layer has mixer inputs"),
            )
            .expect("the mixer sum broadcasts");
        residual_rows = runtime
            .add(&residual_rows, &with_mixer)
            .expect("the residual adds");
    }
    residual_rows
}

/// What one timed pass runs over the K input rows.
enum MeasuredPass<'weights> {
    LayerStack(&'weights [MeasuredLayer]),
    OutputHead(&'weights PackedProjection),
    /// Distinct copies of one shape, each read once a pass, with its input width.
    ShapeCopies(&'weights [PackedProjection], i32),
}

/// Projects every copy and sums the results to one scalar, so one evaluation
/// runs them all and no output is dropped as unused.
fn shape_copies_pass(
    runtime: &MlxRuntime,
    matmul_path: MatmulPath<'_>,
    shape_copies: &[PackedProjection],
    input_rows: &MlxArray,
) -> MlxArray {
    let mut running_sum: Option<MlxArray> = None;
    for shape_copy in shape_copies {
        let projected_sum = runtime
            .sum(
                &project(runtime, matmul_path, input_rows, shape_copy),
                false,
            )
            .expect("the projection reduces");
        running_sum = Some(match running_sum {
            Some(previous_sum) => runtime
                .add(&previous_sum, &projected_sum)
                .expect("sums add"),
            None => projected_sum,
        });
    }
    running_sum.expect("there is at least one copy")
}

/// The pass's inputs are built and evaluated before the clock starts, so a
/// timed pass holds only the projections and the small reductions.
fn median_pass_time(
    runtime: &MlxRuntime,
    random_key: &MlxArray,
    row_count: i32,
    measured_pass: &MeasuredPass<'_>,
    matmul_path: MatmulPath<'_>,
) -> Duration {
    let second_input_width = match measured_pass {
        MeasuredPass::ShapeCopies(_, input_width) => *input_width,
        _ => MIXER_OUTPUT_WIDTH,
    };
    let hidden_rows = random_rows(runtime, random_key, row_count, HIDDEN_WIDTH);
    let second_rows = random_rows(runtime, random_key, row_count, second_input_width);
    runtime
        .evaluate_arrays(&[&hidden_rows, &second_rows])
        .expect("input rows materialize");
    let mut pass_times = Vec::with_capacity(MEASURED_PASS_COUNT);
    for pass_index in 0..WARM_UP_PASS_COUNT + MEASURED_PASS_COUNT {
        let pass_started_at = Instant::now();
        let pass_output = match measured_pass {
            MeasuredPass::LayerStack(layers) => {
                stack_pass(runtime, matmul_path, layers, &hidden_rows, &second_rows)
            }
            MeasuredPass::OutputHead(output_head) => {
                project(runtime, matmul_path, &hidden_rows, output_head)
            }
            MeasuredPass::ShapeCopies(shape_copies, _) => {
                shape_copies_pass(runtime, matmul_path, shape_copies, &second_rows)
            }
        };
        runtime
            .evaluate_arrays(&[&pass_output])
            .expect("the pass evaluates");
        if pass_index >= WARM_UP_PASS_COUNT {
            pass_times.push(pass_started_at.elapsed());
        }
    }
    pass_times.sort();
    pass_times[pass_times.len() / 2]
}

fn layer_stored_bytes(layer: &MeasuredLayer) -> usize {
    let mixer_input_bytes: usize = layer
        .mixer_inputs
        .iter()
        .map(|projection| projection.stored_bytes)
        .sum();
    mixer_input_bytes
        + layer.mixer_output.stored_bytes
        + layer.mlp_gate.stored_bytes
        + layer.mlp_up.stored_bytes
        + layer.mlp_down.stored_bytes
}

/// `stack` and `head` are each (this row count's time, one row's time).
fn report_row_count(
    matmul_path: MatmulPath<'_>,
    row_count: i32,
    stack: (Duration, Duration),
    head: (Duration, Duration),
    stack_bytes: usize,
) {
    let layer_scale = MODEL_LAYER_COUNT / MEASURED_LAYER_COUNT as f64;
    let read_rate_gb_per_second = stack_bytes as f64 / BYTES_PER_GB / stack.0.as_secs_f64();
    eprintln!(
        "[k-row] {} K={row_count:>2}: stack {:7.2} ms ({:.2}x one row, {:.0} GB/s) -> 64 layers {:6.1} ms | head {:6.2} ms ({:.2}x)",
        matmul_path.label(),
        stack.0.as_secs_f64() * 1000.0,
        stack.0.as_secs_f64() / stack.1.as_secs_f64(),
        read_rate_gb_per_second,
        stack.0.as_secs_f64() * 1000.0 * layer_scale,
        head.0.as_secs_f64() * 1000.0,
        head.0.as_secs_f64() / head.1.as_secs_f64(),
    );
}

const SWEPT_SLICE_COUNTS: [i32; 4] = [1, 2, 4, 8];
const SWEPT_ROW_COUNTS: [i32; 2] = [1, 16];

/// The register-ordered kernel's input slices per shape: the table in
/// `register_ordered_plan` came from TensorFold's measurements on an M5
/// Ultra, so this times every slice count on this chip, at one row (plain
/// decode) and 16 (a draft tree), over 16 distinct copies of each shape.
#[test]
#[ignore = "performance measurement on the GPU; run through cargo measure-k-row-verify-cost"]
#[serial]
fn should_sweep_register_ordered_slice_counts_for_each_projection_shape() {
    let runtime = initialize_runtime();
    let kernels = RowBatchedMatmulKernels::declare().expect("the kernels are declared");
    let random_key = runtime.random_key(1234).expect("a key builds");
    for (output_width, input_width) in PROJECTION_SHAPES {
        let mut shape_copies = Vec::with_capacity(MEASURED_LAYER_COUNT);
        for _ in 0..MEASURED_LAYER_COUNT {
            shape_copies.push(pack_projection(
                &runtime,
                &kernels,
                &random_key,
                output_width,
                input_width,
            ));
        }
        let shipped_slice_count = match &shape_copies[0].register_ordered {
            Some(converted) => converted.input_slice_count,
            None => 0,
        };
        let mut sweep_line =
            format!("[k-row] slices {output_width}x{input_width} (shipped {shipped_slice_count}):");
        for slice_count in SWEPT_SLICE_COUNTS {
            for shape_copy in &mut shape_copies {
                if let Some(converted) = shape_copy.register_ordered.as_mut() {
                    converted.input_slice_count = slice_count;
                }
            }
            for row_count in SWEPT_ROW_COUNTS {
                let pass_time = median_pass_time(
                    &runtime,
                    &random_key,
                    row_count,
                    &MeasuredPass::ShapeCopies(&shape_copies, input_width),
                    MatmulPath::RegisterOrdered(&kernels),
                );
                sweep_line.push_str(&format!(
                    " SK={slice_count} K={row_count} {:.2} ms",
                    pass_time.as_secs_f64() * 1000.0
                ));
            }
        }
        eprintln!("{sweep_line}");
    }
}
