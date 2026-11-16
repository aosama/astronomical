//! The `k` largest logits of every row, found in one pass over the
//! vocabulary.
//!
//! Speculative decoding needs a few tokens per row from 248,320-wide rows
//! twice a round: the drafter's candidates and the target's sampling
//! distribution. MLX's argpartition orders the whole row to do it, which
//! measured 2.7 ms for the drafter's 7 rows and 3.2 ms for a 16-row tree on
//! an M5 Pro, against a few tenths of a millisecond to read those rows once.
//!
//! One threadgroup takes a row: each thread keeps the best `K` of its
//! strided share in registers, each simdgroup merges its 32 lists by `K`
//! rounds of `simd_max`, and the first simdgroup merges the 16 simdgroups'
//! lists the same way. The result is exact and sorted best first; equal
//! logits go to the lower token id.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
};

use super::{sampling_failure, sampling_runtime_error};
use crate::InferenceEngineError;

/// Sixteen simdgroups: within the 640 threads a threadgroup may hold on
/// every Metal GPU this runs on (GitHub's virtual Macs allow no more), where
/// 1,024 was refused outright.
const THREADS_PER_ROW: i32 = 512;
/// The largest `k` a call may ask for: registers hold `K` pairs a thread.
pub(crate) const MAXIMUM_TOP_K: i32 = 32;

/// Input `logits` [1, rows, V]; output `token_ids` [1, rows, K] uint32.
const VOCABULARY_TOP_K_SOURCE: &str = r#"
    constexpr uint THREADS = 512;
    constexpr uint SIMDGROUPS = THREADS / 32;
    const uint row = threadgroup_position_in_grid.y;
    const uint thread_index = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simdgroup = simdgroup_index_in_threadgroup;
    const device auto* row_logits = logits + (size_t)row * V;

    float best_values[K];
    uint best_ids[K];
    for (int slot = 0; slot < K; ++slot) {
        best_values[slot] = -INFINITY;
        best_ids[slot] = 0xFFFFFFFFu;
    }
    for (uint token = thread_index; token < V; token += THREADS) {
        const float value = float(row_logits[token]);
        if (value < best_values[K - 1] ||
            (value == best_values[K - 1] && token > best_ids[K - 1])) {
            continue;
        }
        int slot = K - 1;
        while (slot > 0 && (value > best_values[slot - 1] ||
               (value == best_values[slot - 1] && token < best_ids[slot - 1]))) {
            best_values[slot] = best_values[slot - 1];
            best_ids[slot] = best_ids[slot - 1];
            --slot;
        }
        best_values[slot] = value;
        best_ids[slot] = token;
    }

    threadgroup float simdgroup_values[THREADS / 32 * K];
    threadgroup uint simdgroup_ids[THREADS / 32 * K];
    int head = 0;
    for (int round = 0; round < K; ++round) {
        const float candidate = head < K ? best_values[head] : -INFINITY;
        const uint candidate_id = head < K ? best_ids[head] : 0xFFFFFFFFu;
        const float round_best = simd_max(candidate);
        const uint round_id = simd_min(candidate == round_best ? candidate_id : 0xFFFFFFFFu);
        if (candidate == round_best && candidate_id == round_id) {
            ++head;
        }
        if (lane == 0) {
            simdgroup_values[simdgroup * K + round] = round_best;
            simdgroup_ids[simdgroup * K + round] = round_id;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup != 0) {
        return;
    }
    head = 0;
    for (int round = 0; round < K; ++round) {
        const bool lane_has_list = lane < SIMDGROUPS && head < K;
        const float candidate = lane_has_list ? simdgroup_values[lane * K + head] : -INFINITY;
        const uint candidate_id = lane_has_list ? simdgroup_ids[lane * K + head] : 0xFFFFFFFFu;
        const float round_best = simd_max(candidate);
        const uint round_id = simd_min(candidate == round_best ? candidate_id : 0xFFFFFFFFu);
        if (candidate == round_best && candidate_id == round_id) {
            ++head;
        }
        if (lane == 0) {
            token_ids[(size_t)row * K + round] = round_id;
        }
    }
"#;

/// The `kept_count` largest-logit token ids of every row of `logits`
/// [1, rows, vocabulary], best first: [1, rows, kept_count] uint32.
pub(crate) fn vocabulary_top_k(
    runtime: &MlxRuntime,
    logits: &MlxArray,
    kept_count: i32,
) -> Result<MlxArray, InferenceEngineError> {
    let shape = logits.shape();
    if shape.len() != 3 || shape[0] != 1 || !(1..=MAXIMUM_TOP_K).contains(&kept_count) {
        return Err(sampling_failure(
            "top-k reads [1, rows, vocabulary] logits and keeps 1 to 32 tokens a row",
        ));
    }
    let (row_count, vocabulary_size) = (shape[1], shape[2]);
    // Declared per call: MLX compiles the source once per process.
    let kernel = MlxMetalKernel::new(
        "astro_vocabulary_top_k",
        &["logits"],
        &["token_ids"],
        VOCABULARY_TOP_K_SOURCE,
    )
    .map_err(sampling_runtime_error)?;
    let mut outputs = runtime
        .apply_metal_kernel(
            &kernel,
            &[logits],
            &[MlxMetalKernelOutput::new(
                vec![1, row_count, kept_count],
                MlxDtype::UInt32,
            )],
            [THREADS_PER_ROW, row_count, 1],
            [THREADS_PER_ROW, 1, 1],
            &[
                MlxMetalKernelTemplateArgument::Integer {
                    name: "K",
                    integer_template_argument: kept_count,
                },
                MlxMetalKernelTemplateArgument::Integer {
                    name: "V",
                    integer_template_argument: vocabulary_size,
                },
            ],
        )
        .map_err(sampling_runtime_error)?;
    outputs
        .pop()
        .ok_or_else(|| sampling_failure("the top-k kernel returned no output"))
}
