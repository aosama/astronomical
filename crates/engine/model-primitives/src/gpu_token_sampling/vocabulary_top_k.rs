//! The `k` largest logits of every row, found in one pass over the
//! vocabulary.
//!
//! Speculative decoding needs a few tokens per row from 248,320-wide rows
//! twice a round: the drafter's candidates and the target's sampling
//! distribution. MLX's argpartition orders the whole row to do it, which
//! measured 2.7 ms for the drafter's 7 rows and 3.2 ms for a 16-row tree on
//! an M5 Pro, against a few tenths of a millisecond to read those rows once.
//!
//! The row is cut into chunks, one threadgroup each, so a round's few rows
//! still fill the GPU: with one threadgroup a row, a call on a draft
//! round's 3 or 4 rows ran 3 or 4 threadgroups and took about 0.75 ms
//! alone, where reading the rows once takes 5 microseconds
//! (`cargo measure-vocabulary-top-k-cost`). In each threadgroup every
//! thread keeps the best `K` of its strided share in registers, each
//! simdgroup merges its 32 lists by `K` rounds of `simd_max`, and the first
//! simdgroup merges the 8 simdgroups' lists the same way, writing the
//! chunk's best `K` with their logits. A second launch merges a row's
//! chunk lists the same way. The result is exact and sorted best first;
//! equal logits go to the lower token id.

use mlx_bridge::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
};
use runtime_integration::MlxRuntime;

use super::{sampling_failure, sampling_runtime_error};
use crate::gpu_token_sampling::TokenSamplingError;

/// Eight simdgroups. A threadgroup's thread limit falls as each thread's
/// registers grow, and a thread here holds `2 * K` of them: GitHub's
/// virtual Macs allowed 640 threads at `K = 20` and only 384 at `K = 32`,
/// where an M5 Pro allows 1,024 at both. 256 fits every limit seen.
const THREADS_PER_ROW: i32 = 256;
/// The largest `k` a call may ask for: registers hold `K` pairs a thread.
pub const MAXIMUM_TOP_K: i32 = 32;

/// The `K` best of candidates `[0, CANDIDATES)` read through
/// `CANDIDATE_VALUE(index)` and `CANDIDATE_ID(index)`, written through
/// `WRITE_BEST(round, value, id)` by the first simdgroup. Shared by both
/// launches so the chunk pass and the merge rank candidates the same way.
const BEST_K_OF_CANDIDATES: &str = r#"
    constexpr uint THREADS = 256;
    constexpr uint SIMDGROUPS = THREADS / 32;
    const uint thread_index = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simdgroup = simdgroup_index_in_threadgroup;

    float best_values[K];
    uint best_ids[K];
    for (int slot = 0; slot < K; ++slot) {
        best_values[slot] = -INFINITY;
        best_ids[slot] = 0xFFFFFFFFu;
    }
    for (uint index = thread_index; index < CANDIDATES; index += THREADS) {
        const float value = CANDIDATE_VALUE(index);
        const uint token = CANDIDATE_ID(index);
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
            WRITE_BEST(round, round_best, round_id);
        }
    }
"#;

/// Input `logits` [1, rows, V]; outputs each chunk's best `K` logits and
/// token ids, [rows, C, K]. Threadgroup `x` is the chunk, `y` the row.
const CHUNK_PASS_PRELUDE: &str = r#"
    const uint row = threadgroup_position_in_grid.y;
    const uint chunk = threadgroup_position_in_grid.x;
    constexpr uint SPAN = (V + C - 1) / C;
    const uint chunk_start = chunk * SPAN;
    const uint chunk_end = min(chunk_start + SPAN, uint(V));
    const uint CANDIDATES = chunk_end > chunk_start ? chunk_end - chunk_start : 0;
    const device auto* chunk_logits = logits + (size_t)row * V + chunk_start;
    const size_t list_base = ((size_t)row * C + chunk) * K;
    #define CANDIDATE_VALUE(index) float(chunk_logits[index])
    #define CANDIDATE_ID(index) (chunk_start + (index))
    #define WRITE_BEST(round, value, id) \
        { chunk_values[list_base + (round)] = (value); chunk_ids[list_base + (round)] = (id); }
"#;

/// Input `logits` [1, rows, V]; output the row's best `K` token ids,
/// [1, rows, K]. One threadgroup a row, for one or two rows.
const SINGLE_PASS_PRELUDE: &str = r#"
    const uint row = threadgroup_position_in_grid.y;
    constexpr uint CANDIDATES = V;
    const device auto* row_logits = logits + (size_t)row * V;
    #define CANDIDATE_VALUE(index) float(row_logits[index])
    #define CANDIDATE_ID(index) (index)
    #define WRITE_BEST(round, value, id) { token_ids[(size_t)row * K + (round)] = (id); }
"#;

/// Inputs every chunk's best `K` of a row; output the row's best `K` token
/// ids, [1, rows, K]. One threadgroup a row.
const MERGE_PASS_PRELUDE: &str = r#"
    const uint row = threadgroup_position_in_grid.y;
    constexpr uint CANDIDATES = C * K;
    const auto row_values = chunk_values + (size_t)row * CANDIDATES;
    const auto row_ids = chunk_ids + (size_t)row * CANDIDATES;
    #define CANDIDATE_VALUE(index) row_values[index]
    #define CANDIDATE_ID(index) row_ids[index]
    #define WRITE_BEST(round, value, id) { token_ids[(size_t)row * K + (round)] = (id); }
"#;

/// The `kept_count` largest-logit token ids of every row of `logits`
/// [1, rows, vocabulary], best first: [1, rows, kept_count] uint32.
pub fn vocabulary_top_k(
    runtime: &MlxRuntime,
    logits: &MlxArray,
    kept_count: i32,
) -> Result<MlxArray, TokenSamplingError> {
    let shape = logits.shape();
    if shape.len() != 3 || shape[0] != 1 || !(1..=MAXIMUM_TOP_K).contains(&kept_count) {
        return Err(sampling_failure(
            "top-k reads [1, rows, vocabulary] logits and keeps 1 to 32 tokens a row",
        ));
    }
    let (row_count, vocabulary_size) = (shape[1], shape[2]);
    let chunk_count = chunk_count_for(row_count, vocabulary_size);
    if chunk_count == 1 {
        return single_pass_top_k(runtime, logits, row_count, kept_count, vocabulary_size);
    }
    let row_shape = RowShape {
        row_count,
        kept_count,
        vocabulary_size,
        chunk_count,
    };
    let (chunk_values, chunk_ids) = chunk_lists(runtime, logits, &row_shape)?;
    merged_chunk_lists(runtime, &chunk_values, &chunk_ids, &row_shape)
}

/// The shape one chunked call runs at.
struct RowShape {
    row_count: i32,
    kept_count: i32,
    vocabulary_size: i32,
    chunk_count: i32,
}

impl RowShape {
    fn template_arguments(&self) -> [MlxMetalKernelTemplateArgument; 3] {
        [
            integer_template_argument("K", self.kept_count),
            integer_template_argument("V", self.vocabulary_size),
            integer_template_argument("C", self.chunk_count),
        ]
    }
}

/// Every chunk's best `K` logits and token ids, [rows, C, K] each.
fn chunk_lists(
    runtime: &MlxRuntime,
    logits: &MlxArray,
    row_shape: &RowShape,
) -> Result<(MlxArray, MlxArray), TokenSamplingError> {
    // Declared per call: MLX compiles each source once per process.
    let chunk_kernel = MlxMetalKernel::new(
        "astro_vocabulary_top_k_chunks",
        &["logits"],
        &["chunk_values", "chunk_ids"],
        &[CHUNK_PASS_PRELUDE, BEST_K_OF_CANDIDATES].concat(),
    )
    .map_err(sampling_runtime_error)?;
    let list_shape = vec![
        row_shape.row_count,
        row_shape.chunk_count,
        row_shape.kept_count,
    ];
    let mut outputs = runtime
        .apply_metal_kernel(
            &chunk_kernel,
            &[logits],
            &[
                MlxMetalKernelOutput::new(list_shape.clone(), MlxDtype::Float32),
                MlxMetalKernelOutput::new(list_shape, MlxDtype::UInt32),
            ],
            [
                THREADS_PER_ROW * row_shape.chunk_count,
                row_shape.row_count,
                1,
            ],
            [THREADS_PER_ROW, 1, 1],
            &row_shape.template_arguments(),
        )
        .map_err(sampling_runtime_error)?
        .into_iter();
    match (outputs.next(), outputs.next()) {
        (Some(chunk_values), Some(chunk_ids)) => Ok((chunk_values, chunk_ids)),
        _ => Err(sampling_failure(
            "the top-k chunk pass returned the wrong outputs",
        )),
    }
}

/// A row's best `K` token ids from its chunks' lists, [1, rows, K].
fn merged_chunk_lists(
    runtime: &MlxRuntime,
    chunk_values: &MlxArray,
    chunk_ids: &MlxArray,
    row_shape: &RowShape,
) -> Result<MlxArray, TokenSamplingError> {
    let merge_kernel = MlxMetalKernel::new(
        "astro_vocabulary_top_k_merge",
        &["chunk_values", "chunk_ids"],
        &["token_ids"],
        &[MERGE_PASS_PRELUDE, BEST_K_OF_CANDIDATES].concat(),
    )
    .map_err(sampling_runtime_error)?;
    let mut outputs = runtime
        .apply_metal_kernel(
            &merge_kernel,
            &[chunk_values, chunk_ids],
            &[MlxMetalKernelOutput::new(
                vec![1, row_shape.row_count, row_shape.kept_count],
                MlxDtype::UInt32,
            )],
            [THREADS_PER_ROW, row_shape.row_count, 1],
            [THREADS_PER_ROW, 1, 1],
            &row_shape.template_arguments(),
        )
        .map_err(sampling_runtime_error)?;
    outputs
        .pop()
        .ok_or_else(|| sampling_failure("the top-k kernel returned no output"))
}

/// One launch, one threadgroup a row: what one or two rows run, since a
/// second launch cost a multi-token-prediction reply on Flash-Next about
/// 1% (25.7 against 25.3 tokens a second) where it saved nothing.
fn single_pass_top_k(
    runtime: &MlxRuntime,
    logits: &MlxArray,
    row_count: i32,
    kept_count: i32,
    vocabulary_size: i32,
) -> Result<MlxArray, TokenSamplingError> {
    let kernel = MlxMetalKernel::new(
        "astro_vocabulary_top_k",
        &["logits"],
        &["token_ids"],
        &[SINGLE_PASS_PRELUDE, BEST_K_OF_CANDIDATES].concat(),
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
                integer_template_argument("K", kept_count),
                integer_template_argument("V", vocabulary_size),
            ],
        )
        .map_err(sampling_runtime_error)?;
    outputs
        .pop()
        .ok_or_else(|| sampling_failure("the top-k kernel returned no output"))
}

/// Chunks a row is cut into: enough threadgroups in all to fill the GPU
/// for a round's few rows, no more for many rows, and never a chunk so
/// short that a thread has only a few tokens.
const THREADGROUPS_TO_FILL_THE_GPU: i32 = 128;
/// Rows a single launch serves; more are cut into chunks.
const ROWS_ONE_PASS_SERVES: i32 = 2;
const TOKENS_A_CHUNK_AT_LEAST: i32 = 4_096;
const MOST_CHUNKS_A_ROW: i32 = 32;

fn chunk_count_for(row_count: i32, vocabulary_size: i32) -> i32 {
    if row_count <= ROWS_ONE_PASS_SERVES {
        return 1;
    }
    let chunks_to_fill_the_gpu = THREADGROUPS_TO_FILL_THE_GPU / row_count.max(1);
    let chunks_the_row_allows = vocabulary_size / TOKENS_A_CHUNK_AT_LEAST;
    chunks_to_fill_the_gpu
        .min(chunks_the_row_allows)
        .clamp(1, MOST_CHUNKS_A_ROW)
}

fn integer_template_argument(name: &'static str, value: i32) -> MlxMetalKernelTemplateArgument {
    MlxMetalKernelTemplateArgument::Integer {
        name,
        integer_template_argument: value,
    }
}
