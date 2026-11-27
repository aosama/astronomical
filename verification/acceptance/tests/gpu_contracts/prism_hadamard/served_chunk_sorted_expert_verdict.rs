//! Whether a served prompt chunk on an installed mixture-of-experts pack reads
//! faster with the sorted-expert matmul taking its gate and up projections,
//! and whether the chunk's logits keep every bit.
//!
//! One loaded model serves a 2,048-token Romeo and Juliet chunk two ways: with
//! the sorted-expert matmul paused (MLX's route: a sorted copy of the rows,
//! then `gather_qmm` for gate and for up) and with it on (gate and up read
//! straight from the token rows). First each way's final logits are compared
//! element by element, and any difference fails the run; then the profiler
//! times the two as procedures in twelve interleaved rounds, order flipped
//! every round, with the GPU's busy and idle time.
//!
//! Read on an M5 Pro with Ornith 1.5 35B-A3B (OptiQ 4-bit), 2026-11-27: the
//! same logits to the bit, and the chunk 1.0% faster (796 against 807 ms),
//! faster in 11 of 12 rounds. Inside a real layer MLX's gate and up gathers
//! already overlap, so the kernel's 13% at the multiply becomes about 1% of
//! the chunk.
//!
//! Run with `cargo measure-sorted-expert-matmul-served-chunk --configuration
//! PATH` (a `model_directory` of a 4-bit mixture-of-experts pack). About a
//! minute.

use mlx_bridge::MlxArray;
use mlx_performance_profiler::{MeasuredProcedure, ReportLocation, compare_procedures};
use model_serving::{Qwen3_5TextModel, RequestDecoderStateStack};
use serial_test::serial;

use super::prefill_chunk_breakdown::{chunk_token_ids, forward_prepared_chunk};
use super::shared_core::{empty_shared_core_state, loaded_shared_core};
use super::support::{PREFILL_CHUNK_TOKENS, pack_directory};
use crate::sorted_expert_fixtures::mismatched_values;

const REPORT_SUBDIRECTORY: &str = "sorted-expert-matmul-served-chunk";

/// One whole chunk through every layer, from an empty decoder state, with
/// the sorted-expert matmul on or paused.
struct ServedChunk<'model> {
    name: &'static str,
    shared_core: &'model Qwen3_5TextModel,
    chunk_token_ids: &'model [u32],
    uses_sorted_expert_matmul: bool,
    decoder_state: Option<RequestDecoderStateStack>,
}

impl MeasuredProcedure for ServedChunk<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn prepare(&mut self) -> Result<(), String> {
        self.shared_core
            .optional_accelerators
            .sorted_expert_matmul_paused_for_measurement
            .set(!self.uses_sorted_expert_matmul);
        self.decoder_state = Some(empty_shared_core_state(self.shared_core));
        Ok(())
    }

    fn run(&mut self) -> Result<(), String> {
        self.forward().map(|_final_logits| ())
    }
}

impl ServedChunk<'_> {
    fn forward(&mut self) -> Result<MlxArray, String> {
        forward_prepared_chunk(
            self.shared_core,
            self.chunk_token_ids,
            &mut self.decoder_state,
        )
    }

    /// The chunk's final logits this way, evaluated.
    fn final_logits(&mut self) -> MlxArray {
        self.prepare().expect("the chunk prepares");
        let final_logits = self.forward().expect("the chunk forwards");
        self.shared_core
            .runtime
            .evaluate_arrays(&[&final_logits])
            .expect("the logits evaluate");
        final_logits
    }
}

#[test]
#[serial]
#[ignore = "loads an installed mixture-of-experts pack on the GPU; run through cargo measure-sorted-expert-matmul-served-chunk with model_directory configured"]
fn should_give_a_verdict_on_the_sorted_expert_matmul_in_a_served_chunk() {
    let shared_core = loaded_shared_core(&pack_directory());
    assert!(
        shared_core
            .optional_accelerators
            .sorted_expert_matmul_kernels
            .is_some(),
        "the pack's model kept no sorted-expert matmul: its capability probe failed or it has no sorted route"
    );
    let chunk_token_ids = chunk_token_ids(PREFILL_CHUNK_TOKENS);
    let mut mlx_route = served_chunk(&shared_core, &chunk_token_ids, false);
    let mut kernel_route = served_chunk(&shared_core, &chunk_token_ids, true);
    let mismatched_logits = mismatched_values(
        &shared_core.runtime,
        &mlx_route.final_logits(),
        &kernel_route.final_logits(),
    );
    let report_directory = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/tmp")
        .join(REPORT_SUBDIRECTORY);
    std::fs::create_dir_all(&report_directory).expect("the report folder is created");
    let comparison = compare_procedures(
        &mut mlx_route,
        &mut kernel_route,
        ReportLocation {
            title: "A served prompt chunk: gate and up through the sorted-expert matmul against MLX",
            directory: &report_directory,
            file_base: "served-chunk",
        },
    )
    .expect("the profiler compares the two routes");
    shared_core
        .optional_accelerators
        .sorted_expert_matmul_paused_for_measurement
        .set(false);
    println!(
        "[sorted-expert-matmul-served-chunk] tokens={PREFILL_CHUNK_TOKENS} mismatched_logits={mismatched_logits} duration mlx={:.1}ms kernel={:.1}ms relative={:+.1}% verdict={:?} rounds_faster={}/{} gpu_busy mlx={:.1}ms kernel={:.1}ms",
        comparison.duration.baseline_median_microseconds / 1_000.0,
        comparison.duration.candidate_median_microseconds / 1_000.0,
        comparison.duration.relative_difference * 100.0,
        comparison.duration.verdict,
        comparison.duration.rounds_candidate_faster,
        comparison.duration.round_count,
        comparison.gpu_busy.baseline_median_microseconds / 1_000.0,
        comparison.gpu_busy.candidate_median_microseconds / 1_000.0,
    );
    assert_eq!(
        mismatched_logits, 0,
        "the sorted-expert matmul changed the chunk's logits"
    );
}

fn served_chunk<'model>(
    shared_core: &'model Qwen3_5TextModel,
    chunk_token_ids: &'model [u32],
    uses_sorted_expert_matmul: bool,
) -> ServedChunk<'model> {
    ServedChunk {
        name: if uses_sorted_expert_matmul {
            "sorted_expert_matmul"
        } else {
            "mlx_gather_qmm"
        },
        shared_core,
        chunk_token_ids,
        uses_sorted_expert_matmul,
        decoder_state: None,
    }
}
