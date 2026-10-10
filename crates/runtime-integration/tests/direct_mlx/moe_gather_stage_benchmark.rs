use std::time::Instant;

use astronomical_runtime_integration::MlxRuntime;

use crate::common::runtime_test_support;
use astronomical_mlx_c_rust::{MlxArray, MlxDtype};

const WARMUP_ITERATIONS: usize = 2;
const MEASUREMENT_ITERATIONS: usize = 5;
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

/// Ornith-1.5-35B-A3B-OptiQ-4bit MoE geometry from its config.json text_config.
const EXPERT_COUNT: usize = 256;
const TOP_K: usize = 8;
const HIDDEN_SIZE: usize = 2_048;
const MOE_INTERMEDIATE: usize = 512;
const LAYER_COUNT: usize = 40;
/// End-to-end prefill wall time per 4096-token chunk measured in the item-3
/// tile sweep (1,980 tokens/s warm); used only as the headroom reference.
const E2E_CHUNK_4096_MILLIS: f64 = 2_068.0;

const CHUNK_TOKEN_COUNTS: [usize; 2] = [2_048, 4_096];

/// Sizes the three MoE gather-stage deltas against the stock NAX gather
/// kernel at real Ornith shapes: (1) the replicated-row copy produced by
/// sorting assignments, (2) the separate SwiGLU elementwise pass over the
/// fused gate/up output, and (3) the masked tensor-unit work the stock
/// kernel spends on 64-row blocks that span two experts. Each stage is
/// timed in isolation so a later kernel patch can attribute its gain to a
/// specific stage instead of guessing from end-to-end wall time.
#[test]
#[ignore = "measures MoE gather stages on real GPU kernels; run directly with one test thread"]
fn should_measure_moe_gather_stage_costs_at_ornith_shapes() {
    let runtime = runtime_test_support::runtime();

    for token_count in CHUNK_TOKEN_COUNTS {
        let sorted_row_count = token_count * TOP_K;
        eprintln!(
            "[moe-gather-stage] chunk_tokens={token_count} sorted_rows={sorted_row_count} rows_per_expert={}",
            sorted_row_count / EXPERT_COUNT
        );

        let sorted_row_count_i = sorted_row_count as i32;
        let expert_ids = build_uniform_expert_ids(sorted_row_count);
        let argsort_permutation = runtime
            .argsort_axis(&expert_ids, 0)
            .expect("the expert id argsort should build");
        let permutation_flat = runtime
            .reshape(&argsort_permutation, &[sorted_row_count_i])
            .expect("the argsort permutation should flatten");
        let sorted_expert_ids = runtime
            .take_axis(&expert_ids, &permutation_flat, 0)
            .expect("the sorted expert ids should build");
        let sorted_ids_flat = runtime
            .reshape(&sorted_expert_ids, &[sorted_row_count_i])
            .expect("the sorted ids should flatten");
        runtime
            .evaluate_arrays(&[&permutation_flat, &sorted_ids_flat])
            .expect("the routing indices should materialize");

        let replicated_rows = measure_assignment_replication(
            &runtime,
            token_count,
            &permutation_flat,
            sorted_row_count,
        );
        let fused_weights =
            build_fused_gate_up_weights(&runtime, EXPERT_COUNT, MOE_INTERMEDIATE * 2, HIDDEN_SIZE);
        let down_weights =
            build_fused_gate_up_weights(&runtime, EXPERT_COUNT, HIDDEN_SIZE, MOE_INTERMEDIATE);

        let activations = build_token_states(&runtime, sorted_row_count, HIDDEN_SIZE);
        let gate_up_realistic_millis = measure_gather_qmm(
            &runtime,
            &activations,
            &fused_weights,
            Some(&sorted_ids_flat),
            "gate_up_realistic",
        );
        let single_expert_ids = build_single_expert_ids(sorted_row_count);
        let gate_up_single_expert_millis = measure_gather_qmm(
            &runtime,
            &activations,
            &fused_weights,
            Some(&single_expert_ids),
            "gate_up_single_expert",
        );

        let fused_output = runtime
            .gather_quantized_matmul_affine(
                &activations,
                &fused_weights.packed_words,
                &fused_weights.scales,
                &fused_weights.biases,
                None,
                Some(&sorted_ids_flat),
                true,
                GROUP_SIZE,
                BITS,
                true,
            )
            .expect("the fused gate/up product should build");
        runtime
            .evaluate_arrays(&[&fused_output])
            .expect("the fused gate/up product should materialize");
        let swiglu_millis = measure_swiglu_pass(&runtime, &fused_output, sorted_row_count);

        let swiglu_output = build_swiglu_output(&runtime, sorted_row_count, MOE_INTERMEDIATE);
        let down_realistic_millis = measure_gather_qmm(
            &runtime,
            &swiglu_output,
            &down_weights,
            Some(&sorted_ids_flat),
            "down_realistic",
        );

        let per_layer_millis = replicated_rows.median_millis
            + gate_up_realistic_millis
            + swiglu_millis
            + down_realistic_millis;
        let masked_work_millis = gate_up_realistic_millis - gate_up_single_expert_millis;
        eprintln!(
            "[moe-gather-stage:result] chunk_tokens={token_count} replication_ms={:.3} replication_write_mb={:.1} gate_up_realistic_ms={gate_up_realistic_millis:.3} gate_up_single_expert_ms={gate_up_single_expert_millis:.3} masked_work_ms={masked_work_millis:.3} swiglu_ms={swiglu_millis:.3} down_ms={down_realistic_millis:.3} per_layer_total_ms={per_layer_millis:.3}",
            replicated_rows.median_millis,
            replicated_rows.copied_bytes as f64 / 1_000_000.0,
        );
        eprintln!(
            "[moe-gather-stage:headroom] chunk_tokens={token_count} layers={LAYER_COUNT} moe_stages_total_ms={:.1} e2e_reference_ms={E2E_CHUNK_4096_MILLIS:.1} moe_share_percent={:.1} replication_total_ms={:.1} swiglu_total_ms={:.1} masked_work_total_ms={:.1}",
            per_layer_millis * LAYER_COUNT as f64,
            per_layer_millis * LAYER_COUNT as f64 / E2E_CHUNK_4096_MILLIS * 100.0,
            replicated_rows.median_millis * LAYER_COUNT as f64,
            swiglu_millis * LAYER_COUNT as f64,
            masked_work_millis * LAYER_COUNT as f64,
        );

        runtime
            .clear_allocator_cache()
            .expect("the sweep should release reclaimable MLX allocations");
    }
}

struct ReplicationMeasurement {
    median_millis: f64,
    copied_bytes: usize,
}

/// Times the take_axis replication that materializes [T*k, 1, K] sorted
/// rows — the copy a row-map kernel variant would eliminate.
fn measure_assignment_replication(
    runtime: &MlxRuntime,
    token_count: usize,
    argsort_permutation: &MlxArray,
    sorted_row_count: usize,
) -> ReplicationMeasurement {
    let token_states = build_token_states(runtime, token_count, HIDDEN_SIZE);
    let copied_bytes = sorted_row_count * HIDDEN_SIZE * 2;
    let median_millis = measure_stage("assignment_replication", || {
        let sorted_states = runtime
            .take_axis(&token_states, argsort_permutation, 0)
            .expect("the sorted state replication should build");
        runtime
            .evaluate_arrays(&[&sorted_states])
            .expect("the sorted state replication should evaluate");
    });
    ReplicationMeasurement {
        median_millis,
        copied_bytes,
    }
}

fn measure_gather_qmm(
    runtime: &MlxRuntime,
    activations: &MlxArray,
    weights: &QuantizedWeights,
    rhs_indices: Option<&MlxArray>,
    stage_label: &str,
) -> f64 {
    measure_stage(stage_label, || {
        let product = runtime
            .gather_quantized_matmul_affine(
                activations,
                &weights.packed_words,
                &weights.scales,
                &weights.biases,
                None,
                rhs_indices,
                true,
                GROUP_SIZE,
                BITS,
                true,
            )
            .expect("the gathered quantized matmul should build");
        runtime
            .evaluate_arrays(&[&product])
            .expect("the gathered quantized matmul should evaluate");
    })
}

/// Times the split + silu(gate) * up elementwise chain over the fused
/// [M, 1, 2*inter] output — the pass a SwiGLU kernel epilogue would absorb.
fn measure_swiglu_pass(
    runtime: &MlxRuntime,
    fused_output: &MlxArray,
    sorted_row_count: usize,
) -> f64 {
    let rows = sorted_row_count as i32;
    let fused_width = (MOE_INTERMEDIATE * 2) as i32;
    measure_stage("swiglu_elementwise", || {
        let gate = runtime
            .slice(
                fused_output,
                &[0, 0, 0],
                &[rows, 1, MOE_INTERMEDIATE as i32],
                &[1, 1, 1],
            )
            .expect("the gate slice should build");
        let up = runtime
            .slice(
                fused_output,
                &[0, 0, MOE_INTERMEDIATE as i32],
                &[rows, 1, fused_width],
                &[1, 1, 1],
            )
            .expect("the up slice should build");
        let activated_gate = runtime
            .silu(&gate)
            .expect("the silu activation should build");
        let swiglu = runtime
            .multiply(&activated_gate, &up)
            .expect("the swiglu product should build");
        runtime
            .evaluate_arrays(&[&swiglu])
            .expect("the swiglu chain should evaluate");
    })
}

fn measure_stage(stage_label: &str, stage: impl Fn()) -> f64 {
    for warmup_iteration in 1..=WARMUP_ITERATIONS {
        eprintln!(
            "[moe-gather-stage] warmup {warmup_iteration}/{WARMUP_ITERATIONS} stage={stage_label}"
        );
        stage();
    }
    let mut elapsed_millis = Vec::with_capacity(MEASUREMENT_ITERATIONS);
    for measurement_iteration in 1..=MEASUREMENT_ITERATIONS {
        eprintln!(
            "[moe-gather-stage] measurement {measurement_iteration}/{MEASUREMENT_ITERATIONS} stage={stage_label}"
        );
        let started_at = Instant::now();
        stage();
        elapsed_millis.push(started_at.elapsed().as_secs_f64() * 1_000.0);
    }
    elapsed_millis.sort_by(f64::total_cmp);
    let middle_index = elapsed_millis.len() / 2;
    (elapsed_millis[middle_index - 1] + elapsed_millis[middle_index]) / 2.0
}

struct QuantizedWeights {
    packed_words: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
}

fn build_fused_gate_up_weights(
    runtime: &MlxRuntime,
    expert_count: usize,
    output_columns: usize,
    input_columns: usize,
) -> QuantizedWeights {
    let rows = expert_count * output_columns;
    let words_per_row = input_columns * BITS as usize / 32;
    let weight_words = deterministic_weight_words(rows * words_per_row);
    let packed_words = runtime
        .array_from_u32(
            &weight_words,
            &[
                expert_count as i32,
                output_columns as i32,
                words_per_row as i32,
            ],
        )
        .expect("the packed quantized weights should be valid");

    let groups_per_row = input_columns / GROUP_SIZE as usize;
    let scale_values = vec![0.01f32; rows * groups_per_row];
    let float32_scales = runtime
        .array_from_f32(
            &scale_values,
            &[
                expert_count as i32,
                output_columns as i32,
                groups_per_row as i32,
            ],
        )
        .expect("the affine scales should be valid");
    let scales = runtime
        .astype(&float32_scales, MlxDtype::Float16)
        .expect("the scales should cast to float16");
    let biases = runtime
        .astype(&float32_scales, MlxDtype::Float16)
        .expect("the zero biases should cast to float16");

    runtime
        .evaluate_arrays(&[&packed_words, &scales, &biases])
        .expect("the quantized weights should materialize");
    QuantizedWeights {
        packed_words,
        scales,
        biases,
    }
}

fn build_token_states(runtime: &MlxRuntime, row_count: usize, input_columns: usize) -> MlxArray {
    let activation_values = deterministic_activation_values(row_count * input_columns);
    let float32_activations = runtime
        .array_from_f32(
            &activation_values,
            &[row_count as i32, 1, input_columns as i32],
        )
        .expect("the activation matrix should be valid");
    let activations = runtime
        .astype(&float32_activations, MlxDtype::BFloat16)
        .expect("the activations should cast to bfloat16");
    runtime
        .evaluate_arrays(&[&activations])
        .expect("the activations should materialize");
    activations
}

fn build_swiglu_output(runtime: &MlxRuntime, row_count: usize, intermediate: usize) -> MlxArray {
    let activation_values = deterministic_activation_values(row_count * intermediate);
    let float32_activations = runtime
        .array_from_f32(
            &activation_values,
            &[row_count as i32, 1, intermediate as i32],
        )
        .expect("the swiglu output matrix should be valid");
    let activations = runtime
        .astype(&float32_activations, MlxDtype::BFloat16)
        .expect("the swiglu output should cast to bfloat16");
    runtime
        .evaluate_arrays(&[&activations])
        .expect("the swiglu output should materialize");
    activations
}

/// All rows address expert 0, so the kernel never masks work for mixed
/// 64-row blocks; the gap against the realistic run sizes the masked-work
/// overhead a segmented single-expert tile scheduler would recover.
fn build_single_expert_ids(row_count: usize) -> MlxArray {
    let runtime = runtime_test_support::runtime();
    let expert_ids = vec![0u32; row_count];
    runtime
        .array_from_u32(&expert_ids, &[row_count as i32])
        .expect("the single-expert ids should be valid")
}

fn build_uniform_expert_ids(row_count: usize) -> MlxArray {
    let runtime = runtime_test_support::runtime();
    let mut generator_state = 0x9E37_79B9_7F4A_7C15u64;
    let expert_ids: Vec<u32> = (0..row_count)
        .map(|_| {
            generator_state = generator_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((generator_state >> 33) as u32) % EXPERT_COUNT as u32
        })
        .collect();
    runtime
        .array_from_u32(&expert_ids, &[row_count as i32])
        .expect("the expert ids should be valid")
}

fn deterministic_weight_words(element_count: usize) -> Vec<u32> {
    let mut generator_state = 0x2545_F491_4F6C_DD1Du64;
    (0..element_count)
        .map(|_| {
            generator_state = generator_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (generator_state >> 32) as u32
        })
        .collect()
}

fn deterministic_activation_values(element_count: usize) -> Vec<f32> {
    let mut generator_state = 0x2545_F491_4F6C_DD1Du64;
    (0..element_count)
        .map(|_| {
            generator_state = generator_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit_interval = ((generator_state >> 33) as f32) / ((1u32 << 31) as f32);
            unit_interval * 2.0 - 1.0
        })
        .collect()
}
