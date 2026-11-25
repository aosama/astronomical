//! What one GPU primitive costs when its tensor is small: the fixed price
//! of a kernel dispatch, which a reply token pays thousands of times.
//!
//! A reply token on a 35B-A3B evaluates about 2,600 GPU primitives and on
//! Flash-Next about 9,700 (`gpu_primitives_per_forward` in a reply's
//! stages). Their weight bytes explain about 6 ms and 11 ms of tokens that
//! take 17 ms and 46 ms; this contract prices the rest from below. Each
//! chain is one kind of primitive repeated over a residual-sized vector
//! (Flash-Next's four lanes of 2,560), built once and evaluated once, and
//! the cost is the evaluation's time over the primitives MLX counted.
//! Building the graph on the host is timed apart.
//!
//! The second case prices a one-row 4-bit weight multiply at the shapes a
//! token really runs, beside the time its weight bytes need at the GPU's
//! memory bandwidth: the multiplies of a small projection are far from that
//! floor, and a token runs hundreds of them.
//!
//! No model on disk. Run with `cargo measure-primitive-dispatch-cost`.
//! Seconds.

use std::time::{Duration, Instant};

use graph_host_step::evaluation_counts;
use mlx_bridge::{MlxArray, MlxDtype};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

const LANE_COUNT: i32 = 4;
const HIDDEN_WIDTH: i32 = 2_560;
const CHAIN_LENGTH: usize = 2_000;
const TIMED_PASS_COUNT: usize = 21;

fn test_runtime() -> MlxRuntime {
    let four_gigabytes = 4 * 1024 * 1024 * 1024;
    let memory_limits = MlxMemoryLimits::new(four_gigabytes, four_gigabytes / 8);
    MlxRuntime::initialize(memory_limits.expect("the limits are valid")).expect("MLX starts")
}

/// One kind of primitive, repeated.
#[derive(Clone, Copy, Debug)]
enum ChainedPrimitive {
    /// A view of the same bytes in another shape: no kernel runs.
    Reshape,
    /// One element-wise kernel over the vector.
    MultiplyByScalar,
    /// A unary element-wise kernel.
    Sigmoid,
    /// A dtype change, as a float32 norm makes on a bfloat16 residual.
    CastThereAndBack,
}

fn residual(runtime: &MlxRuntime) -> MlxArray {
    let width = (LANE_COUNT * HIDDEN_WIDTH) as usize;
    let mut values = Vec::with_capacity(width);
    for feature_index in 0..width {
        values.push((feature_index % 97) as f32 / 97.0 - 0.5);
    }
    let float_residual = runtime
        .array_from_f32(&values, &[1, 1, LANE_COUNT * HIDDEN_WIDTH])
        .expect("the residual uploads");
    runtime
        .astype(&float_residual, MlxDtype::BFloat16)
        .expect("the residual narrows")
}

fn chain_of(runtime: &MlxRuntime, primitive: ChainedPrimitive, start: &MlxArray) -> MlxArray {
    let almost_one = runtime
        .astype(
            &runtime.array_from_f32(&[0.999], &[1]).expect("a scalar"),
            MlxDtype::BFloat16,
        )
        .expect("the scalar narrows");
    let flat_shape = [1, 1, LANE_COUNT * HIDDEN_WIDTH];
    let lane_shape = [1, 1, LANE_COUNT, HIDDEN_WIDTH];
    let mut current = start.retain().expect("the start is retained");
    for step_index in 0..CHAIN_LENGTH {
        current = match primitive {
            ChainedPrimitive::Reshape if step_index % 2 == 0 => {
                runtime.reshape(&current, &lane_shape)
            }
            ChainedPrimitive::Reshape => runtime.reshape(&current, &flat_shape),
            ChainedPrimitive::MultiplyByScalar => runtime.multiply(&current, &almost_one),
            ChainedPrimitive::Sigmoid => runtime.sigmoid(&current),
            ChainedPrimitive::CastThereAndBack if step_index % 2 == 0 => {
                runtime.astype(&current, MlxDtype::Float32)
            }
            ChainedPrimitive::CastThereAndBack => runtime.astype(&current, MlxDtype::BFloat16),
        }
        .expect("the chain extends");
    }
    current
}

struct ChainCost {
    build: Duration,
    evaluation: Duration,
    gpu_primitive_count: u64,
}

fn price_one_pass(
    runtime: &MlxRuntime,
    primitive: ChainedPrimitive,
    start: &MlxArray,
) -> ChainCost {
    let build_started_at = Instant::now();
    let chain_end = chain_of(runtime, primitive, start);
    let build = build_started_at.elapsed();
    let counts_before = evaluation_counts();
    let evaluation_started_at = Instant::now();
    runtime
        .evaluate_arrays(&[&chain_end])
        .expect("the chain evaluates");
    let evaluation = evaluation_started_at.elapsed();
    ChainCost {
        build,
        evaluation,
        gpu_primitive_count: evaluation_counts().since(counts_before).gpu_primitives,
    }
}

/// The median pass, so one slow pass does not set the price.
fn median_cost(runtime: &MlxRuntime, primitive: ChainedPrimitive, start: &MlxArray) -> ChainCost {
    let mut passes = Vec::with_capacity(TIMED_PASS_COUNT);
    for _ in 0..TIMED_PASS_COUNT {
        passes.push(price_one_pass(runtime, primitive, start));
    }
    passes.sort_by_key(|pass| pass.evaluation);
    passes.swap_remove(TIMED_PASS_COUNT / 2)
}

#[test]
#[ignore = "prices a small GPU primitive on the local GPU"]
#[serial]
fn should_price_one_small_gpu_primitive_of_each_kind() {
    let runtime = test_runtime();
    let start = residual(&runtime);
    runtime
        .evaluate_arrays(&[&start])
        .expect("the start evaluates");
    for primitive in [
        ChainedPrimitive::Reshape,
        ChainedPrimitive::MultiplyByScalar,
        ChainedPrimitive::Sigmoid,
        ChainedPrimitive::CastThereAndBack,
    ] {
        // One pass first: it compiles the kernel and fills the buffer pool.
        price_one_pass(&runtime, primitive, &start);
        let cost = median_cost(&runtime, primitive, &start);
        let primitive_count = cost.gpu_primitive_count.max(1) as f64;
        eprintln!(
            "[primitive-dispatch-cost] primitive={primitive:?} gpu_primitives={} evaluation_microseconds_a_primitive={:.2} build_microseconds_a_primitive={:.2}",
            cost.gpu_primitive_count,
            cost.evaluation.as_secs_f64() * 1e6 / primitive_count,
            cost.build.as_secs_f64() * 1e6 / CHAIN_LENGTH as f64,
        );
        assert!(
            cost.gpu_primitive_count >= CHAIN_LENGTH as u64,
            "every step of the {primitive:?} chain must be a primitive MLX evaluated"
        );
    }
}

const WEIGHTS_A_SCALE_GROUP: i32 = 64;
const BITS_A_WEIGHT: i32 = 4;
const MULTIPLIES_A_PASS: usize = 200;
/// Weight bytes a second the dense 27B's one-row pass moves: about 15 GB in
/// 52 ms (`cargo measure-k-row-verify-cost`).
const MEASURED_BYTES_A_SECOND: f64 = 290.0e9;

/// A one-row projection a reply token runs: `[1, 1, input] -> [1, 1, output]`.
struct ProjectionShape {
    name: &'static str,
    input_width: i32,
    output_width: i32,
}

const PROJECTION_SHAPES: [ProjectionShape; 7] = [
    ProjectionShape {
        name: "35B-A3B router",
        input_width: 2_048,
        output_width: 256,
    },
    ProjectionShape {
        name: "35B-A3B expert gate",
        input_width: 2_048,
        output_width: 512,
    },
    ProjectionShape {
        name: "35B-A3B expert down",
        input_width: 512,
        output_width: 2_048,
    },
    ProjectionShape {
        name: "35B-A3B attention query",
        input_width: 2_048,
        output_width: 4_096,
    },
    ProjectionShape {
        name: "Flash-Next lane write-back",
        input_width: 10_240,
        output_width: 4,
    },
    ProjectionShape {
        name: "Flash-Next attention query",
        input_width: 2_560,
        output_width: 6_144,
    },
    ProjectionShape {
        name: "vocabulary head",
        input_width: 2_048,
        output_width: 248_320,
    },
];

struct QuantizedProjection {
    activations: MlxArray,
    packed_weight: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
    weight_byte_count: usize,
}

fn quantized_projection(runtime: &MlxRuntime, shape: &ProjectionShape) -> QuantizedProjection {
    let random_key = runtime.random_key(2_567).expect("a random key");
    let weight = runtime
        .random_normal(
            &[shape.output_width, shape.input_width],
            MlxDtype::Float32,
            0.0,
            0.02,
            &random_key,
        )
        .expect("the weight is drawn");
    let (packed_weight, scales, biases) = runtime
        .quantize_affine(&weight, WEIGHTS_A_SCALE_GROUP, BITS_A_WEIGHT)
        .expect("the weight quantizes");
    let activations = runtime
        .astype(
            &runtime
                .random_normal(
                    &[1, 1, shape.input_width],
                    MlxDtype::Float32,
                    0.0,
                    1.0,
                    &random_key,
                )
                .expect("the activations are drawn"),
            MlxDtype::BFloat16,
        )
        .expect("the activations narrow");
    runtime
        .evaluate_arrays(&[&activations, &packed_weight, &scales, &biases])
        .expect("the projection materializes");
    let weight_byte_count = packed_weight.byte_count() + scales.byte_count() + biases.byte_count();
    QuantizedProjection {
        activations,
        packed_weight,
        scales,
        biases,
        weight_byte_count,
    }
}

/// `activations` through `projection`, lazily.
fn one_row_multiply(
    runtime: &MlxRuntime,
    activations: &MlxArray,
    projection: &QuantizedProjection,
) -> MlxArray {
    runtime
        .quantized_matmul_affine(
            activations,
            &projection.packed_weight,
            &projection.scales,
            &projection.biases,
            true,
            WEIGHTS_A_SCALE_GROUP,
            BITS_A_WEIGHT,
        )
        .expect("the multiply builds")
}

/// Seconds for `MULTIPLIES_A_PASS` independent multiplies in one evaluation.
fn multiply_pass_seconds(runtime: &MlxRuntime, projection: &QuantizedProjection) -> f64 {
    let mut products = Vec::with_capacity(MULTIPLIES_A_PASS);
    for _ in 0..MULTIPLIES_A_PASS {
        products.push(one_row_multiply(
            runtime,
            &projection.activations,
            projection,
        ));
    }
    let product_references: Vec<&MlxArray> = products.iter().collect();
    let started_at = Instant::now();
    runtime
        .evaluate_arrays(&product_references)
        .expect("the multiplies evaluate");
    started_at.elapsed().as_secs_f64()
}

/// The median of the timed passes in microseconds a multiply, after one
/// untimed pass: independent multiplies of `there`, or with `back` a chain
/// alternating the two.
fn median_microseconds_a_multiply(
    runtime: &MlxRuntime,
    there: &QuantizedProjection,
    back: Option<&QuantizedProjection>,
) -> f64 {
    let mut pass_seconds = Vec::with_capacity(TIMED_PASS_COUNT + 1);
    for _ in 0..=TIMED_PASS_COUNT {
        pass_seconds.push(match back {
            Some(back) => chained_pass_seconds(runtime, there, back),
            None => multiply_pass_seconds(runtime, there),
        });
    }
    // The first pass compiled the kernel.
    pass_seconds.remove(0);
    pass_seconds.sort_by(f64::total_cmp);
    pass_seconds[pass_seconds.len() / 2] * 1e6 / MULTIPLIES_A_PASS as f64
}

#[test]
#[ignore = "prices a one-row quantized multiply on the local GPU"]
#[serial]
fn should_price_a_one_row_quantized_multiply_against_its_weight_bytes() {
    let runtime = test_runtime();
    for shape in &PROJECTION_SHAPES {
        let projection = quantized_projection(&runtime, shape);
        let microseconds_a_multiply = median_microseconds_a_multiply(&runtime, &projection, None);
        let floor_microseconds =
            projection.weight_byte_count as f64 / MEASURED_BYTES_A_SECOND * 1e6;
        eprintln!(
            "[primitive-dispatch-cost] multiply=\"{}\" weight_megabytes={:.2} microseconds_a_multiply={microseconds_a_multiply:.1} weight_bytes_floor_microseconds={floor_microseconds:.1} times_the_floor={:.1}",
            shape.name,
            projection.weight_byte_count as f64 / 1_048_576.0,
            microseconds_a_multiply / floor_microseconds.max(1e-9),
        );
        assert!(
            microseconds_a_multiply > 0.0,
            "the {} multiply must take time",
            shape.name
        );
    }
}

/// A projection and the one that brings its output back to the input width,
/// so the two can be chained as a token chains its layers.
const CHAINED_PAIRS: [(&str, i32, i32); 3] = [
    ("35B-A3B expert gate then down", 2_048, 512),
    ("35B-A3B attention query then output", 2_048, 4_096),
    ("Flash-Next expert gate then down", 2_560, 640),
];

/// Seconds for `MULTIPLIES_A_PASS` multiplies in which each reads the one
/// before it, alternating the pair.
fn chained_pass_seconds(
    runtime: &MlxRuntime,
    there: &QuantizedProjection,
    back: &QuantizedProjection,
) -> f64 {
    let mut current = there.activations.retain().expect("the start is retained");
    for step_index in 0..MULTIPLIES_A_PASS {
        let projection = if step_index % 2 == 0 { there } else { back };
        current = one_row_multiply(runtime, &current, projection);
    }
    let started_at = Instant::now();
    runtime
        .evaluate_arrays(&[&current])
        .expect("the chain evaluates");
    started_at.elapsed().as_secs_f64()
}

#[test]
#[ignore = "prices chained one-row quantized multiplies on the local GPU"]
#[serial]
fn should_price_quantized_multiplies_that_each_wait_for_the_one_before() {
    let runtime = test_runtime();
    for (pair_name, outer_width, inner_width) in CHAINED_PAIRS {
        let there = quantized_projection(
            &runtime,
            &ProjectionShape {
                name: pair_name,
                input_width: outer_width,
                output_width: inner_width,
            },
        );
        let back = quantized_projection(
            &runtime,
            &ProjectionShape {
                name: pair_name,
                input_width: inner_width,
                output_width: outer_width,
            },
        );
        let microseconds_a_multiply = median_microseconds_a_multiply(&runtime, &there, Some(&back));
        let floor_microseconds = (there.weight_byte_count + back.weight_byte_count) as f64
            / 2.0
            / MEASURED_BYTES_A_SECOND
            * 1e6;
        eprintln!(
            "[primitive-dispatch-cost] chained=\"{pair_name}\" microseconds_a_multiply={microseconds_a_multiply:.1} weight_bytes_floor_microseconds={floor_microseconds:.1} times_the_floor={:.1}",
            microseconds_a_multiply / floor_microseconds.max(1e-9),
        );
        assert!(
            microseconds_a_multiply > 0.0,
            "the {pair_name} chain must take time"
        );
    }
}

/// Experts in the priced table. A gathered multiply reads only the experts
/// it is given, so a table of 64 prices a layer of 256 or 512.
const TABLE_EXPERT_COUNT: i32 = 64;

/// A routed expert layer's shape: `routed` experts of `outer -> inner -> outer`.
struct ExpertLayerShape {
    name: &'static str,
    layer_expert_count: i32,
    routed_expert_count: i32,
    outer_width: i32,
    inner_width: i32,
}

const EXPERT_LAYER_SHAPES: [ExpertLayerShape; 2] = [
    ExpertLayerShape {
        name: "35B-A3B",
        layer_expert_count: 256,
        routed_expert_count: 8,
        outer_width: 2_048,
        inner_width: 512,
    },
    ExpertLayerShape {
        name: "Flash-Next",
        layer_expert_count: 512,
        routed_expert_count: 10,
        outer_width: 2_560,
        inner_width: 640,
    },
];

/// A 4-bit expert table `[experts, output, input]`.
fn expert_table(runtime: &MlxRuntime, input_width: i32, output_width: i32) -> QuantizedProjection {
    let random_key = runtime.random_key(2_568).expect("a random key");
    let weight = runtime
        .random_normal(
            &[TABLE_EXPERT_COUNT, output_width, input_width],
            MlxDtype::Float32,
            0.0,
            0.02,
            &random_key,
        )
        .expect("the table is drawn");
    let (packed_weight, scales, biases) = runtime
        .quantize_affine(&weight, WEIGHTS_A_SCALE_GROUP, BITS_A_WEIGHT)
        .expect("the table quantizes");
    let activations = runtime
        .astype(
            &runtime
                .random_normal(
                    &[1, 1, input_width],
                    MlxDtype::Float32,
                    0.0,
                    1.0,
                    &random_key,
                )
                .expect("the activations are drawn"),
            MlxDtype::BFloat16,
        )
        .expect("the activations narrow");
    runtime
        .evaluate_arrays(&[&activations, &packed_weight, &scales, &biases])
        .expect("the table materializes");
    let weight_byte_count = packed_weight.byte_count() + scales.byte_count() + biases.byte_count();
    QuantizedProjection {
        activations,
        packed_weight,
        scales,
        biases,
        weight_byte_count,
    }
}

/// One gathered multiply: every routed expert's row through its own matrix.
fn gathered_multiply(
    runtime: &MlxRuntime,
    rows: &MlxArray,
    table: &QuantizedProjection,
    routed_expert_ids: &MlxArray,
) -> MlxArray {
    runtime
        .gather_quantized_matmul_affine(
            rows,
            &table.packed_weight,
            &table.scales,
            &table.biases,
            None,
            Some(routed_expert_ids),
            true,
            WEIGHTS_A_SCALE_GROUP,
            BITS_A_WEIGHT,
            true,
        )
        .expect("the gathered multiply builds")
}

/// Seconds for a chain of gathered multiplies alternating the gate and the
/// down table over the routed experts' rows.
fn gathered_chain_seconds(
    runtime: &MlxRuntime,
    shape: &ExpertLayerShape,
    gate: &QuantizedProjection,
    down: &QuantizedProjection,
) -> f64 {
    let mut routed_ids = Vec::with_capacity(shape.routed_expert_count as usize);
    for routed_index in 0..shape.routed_expert_count {
        routed_ids.push((routed_index * 5) as u32 % TABLE_EXPERT_COUNT as u32);
    }
    routed_ids.sort_unstable();
    let routed_expert_ids = runtime
        .array_from_u32(&routed_ids, &[shape.routed_expert_count])
        .expect("the routed ids upload");
    let mut current = runtime
        .broadcast_to(
            &gate.activations,
            &[shape.routed_expert_count, 1, shape.outer_width],
        )
        .expect("the rows broadcast");
    for step_index in 0..MULTIPLIES_A_PASS {
        let table = if step_index % 2 == 0 { gate } else { down };
        current = gathered_multiply(runtime, &current, table, &routed_expert_ids);
    }
    let started_at = Instant::now();
    runtime
        .evaluate_arrays(&[&current])
        .expect("the chain evaluates");
    started_at.elapsed().as_secs_f64()
}

/// Seconds for `MULTIPLIES_A_PASS` route selections in one evaluation:
/// softmax, the partition that finds the best experts, and their scores.
fn route_selection_seconds(runtime: &MlxRuntime, shape: &ExpertLayerShape) -> f64 {
    let random_key = runtime.random_key(2_569).expect("a random key");
    let router_logits = runtime
        .random_normal(
            &[1, 1, shape.layer_expert_count],
            MlxDtype::BFloat16,
            0.0,
            2.0,
            &random_key,
        )
        .expect("the logits are drawn");
    runtime
        .evaluate_arrays(&[&router_logits])
        .expect("the logits evaluate");
    let first_selected = shape.layer_expert_count - shape.routed_expert_count;
    let mut selections = Vec::with_capacity(MULTIPLIES_A_PASS);
    for _ in 0..MULTIPLIES_A_PASS {
        let probabilities = runtime
            .softmax_axis(&router_logits, -1)
            .expect("softmax builds");
        let partitioned = runtime
            .argpartition_axis(&probabilities, first_selected, -1)
            .expect("the partition builds");
        let selected = runtime
            .slice(
                &partitioned,
                &[0, 0, first_selected],
                &[1, 1, shape.layer_expert_count],
                &[1, 1, 1],
            )
            .expect("the slice builds");
        selections.push(
            runtime
                .take_along_axis(&probabilities, &selected, -1)
                .expect("the scores build"),
        );
    }
    let selection_references: Vec<&MlxArray> = selections.iter().collect();
    let started_at = Instant::now();
    runtime
        .evaluate_arrays(&selection_references)
        .expect("the selections evaluate");
    started_at.elapsed().as_secs_f64()
}

#[test]
#[ignore = "prices the pieces of a routed expert layer on the local GPU"]
#[serial]
fn should_price_the_pieces_of_a_routed_expert_layer() {
    let twelve_gigabytes = 12 * 1024 * 1024 * 1024;
    let memory_limits = MlxMemoryLimits::new(twelve_gigabytes, twelve_gigabytes / 8);
    let runtime =
        MlxRuntime::initialize(memory_limits.expect("the limits are valid")).expect("MLX starts");
    for shape in &EXPERT_LAYER_SHAPES {
        let gate = expert_table(&runtime, shape.outer_width, shape.inner_width);
        let down = expert_table(&runtime, shape.inner_width, shape.outer_width);
        let mut chain_seconds = Vec::with_capacity(TIMED_PASS_COUNT);
        let mut route_seconds = Vec::with_capacity(TIMED_PASS_COUNT);
        gathered_chain_seconds(&runtime, shape, &gate, &down);
        route_selection_seconds(&runtime, shape);
        for _ in 0..TIMED_PASS_COUNT {
            chain_seconds.push(gathered_chain_seconds(&runtime, shape, &gate, &down));
            route_seconds.push(route_selection_seconds(&runtime, shape));
        }
        chain_seconds.sort_by(f64::total_cmp);
        route_seconds.sort_by(f64::total_cmp);
        let per_pass = 1e6 / MULTIPLIES_A_PASS as f64;
        let routed_bytes = (gate.weight_byte_count + down.weight_byte_count) as f64
            / 2.0
            / TABLE_EXPERT_COUNT as f64
            * shape.routed_expert_count as f64;
        eprintln!(
            "[primitive-dispatch-cost] expert_layer=\"{}\" gathered_multiply_microseconds={:.1} routed_weight_bytes_floor_microseconds={:.1} route_selection_microseconds={:.1}",
            shape.name,
            chain_seconds[TIMED_PASS_COUNT / 2] * per_pass,
            routed_bytes / MEASURED_BYTES_A_SECOND * 1e6,
            route_seconds[TIMED_PASS_COUNT / 2] * per_pass,
        );
    }
}
