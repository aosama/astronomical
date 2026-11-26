//! Where one draft round's verify pass spends its GPU time, kernel by
//! kernel (issue #2601): the dense 27B's served round, a drafted tree of the
//! engine's own size after a short code prompt, measured by the MLX
//! performance profiler. Prints the pass's median and GPU time and writes the
//! job's page, with its execution graph, and its per-kernel JSON.
//!
//! Run with `cargo measure-dflash2-verify-pass-profile`; append
//! `--configuration PATH` naming another target with its drafter.

use std::path::Path;

use mlx_bridge::MlxArray;
use mlx_performance_profiler::{MeasuredJob, ReportLocation, measure_job};
use model_serving::DraftTreeTarget;
use model_serving::speculative_decoding::tree_rows::TreeRows;
use model_serving::{Qwen3_5ResidentModel, RequestDecoderStateStack};
use serial_test::serial;

use super::dflash2_kept_per_round::load_pair_verifying_trees_as_served;
use super::dflash2_tree_oracle::{Drafting, ORACLE_CELLS, draft_tree, prefill};

/// "code completion 0.7": the served temperature for code.
const CODE_CELL_INDEX: usize = 1;
const REPORT_SUBDIRECTORY: &str = "dflash2-verify-pass-profile";

/// One verify pass from the same stored state every time it is built.
struct VerifyPass<'pair> {
    target: &'pair Qwen3_5ResidentModel,
    tree_rows: &'pair TreeRows,
    anchor_position: u32,
    decoder_state: &'pair mut RequestDecoderStateStack,
    tap_layer_ids: &'pair [i32],
}

impl MeasuredJob for VerifyPass<'_> {
    fn name(&self) -> &str {
        "verify_pass"
    }

    fn fresh_outputs(&mut self) -> Result<Vec<MlxArray>, String> {
        let stored_before = self
            .decoder_state
            .checkpoint()
            .map_err(|checkpoint_error| checkpoint_error.to_string())?;
        let verification = self
            .target
            .verify_draft_tree(
                self.tree_rows,
                self.anchor_position,
                self.decoder_state,
                self.tap_layer_ids,
            )
            .map_err(|verify_error| verify_error.to_string())?;
        // The pass appended the tree's keys; the next build starts from the
        // same state.
        self.decoder_state
            .restore_checkpoint(stored_before)
            .map_err(|restore_error| restore_error.to_string())?;
        let mut outputs = vec![verification.per_row_logits];
        outputs.extend(verification.hidden_state_taps);
        Ok(outputs)
    }
}

#[test]
#[ignore = "loads a real target and its DFlash2 drafter; run through cargo measure-dflash2-verify-pass-profile"]
#[serial]
fn should_profile_one_draft_rounds_verify_pass_kernel_by_kernel() {
    let mut pair = load_pair_verifying_trees_as_served();
    // Serving hands the GPU each few layers while it builds the next; the
    // profiler sees only the graph it is handed, so the pass is built whole.
    pair.target.set_tree_pass_built_whole_for_measurement(true);
    let cell = ORACLE_CELLS[CODE_CELL_INDEX];
    let (mut round, mut decoder_state, _) = prefill(&pair, &cell);
    let tree = draft_tree(&pair, &cell, &mut round, Drafting::Tree);
    let tree_rows = TreeRows::from_draft_tree(round.last_token_id, &tree);
    eprintln!(
        "[verify-pass-profile] tree_rows={} context_tokens={}",
        tree_rows.row_count(),
        round.fed_token_count
    );
    let report_directory = Path::new(env!("CARGO_TARGET_TMPDIR")).join(REPORT_SUBDIRECTORY);
    std::fs::create_dir_all(&report_directory).expect("the report directory is made");
    let summary = measure_job(
        &mut VerifyPass {
            target: &pair.target,
            tree_rows: &tree_rows,
            anchor_position: round.fed_token_count,
            decoder_state: &mut decoder_state,
            tap_layer_ids: &pair.drafter_configuration.target_tap_layer_ids,
        },
        ReportLocation {
            title: "One draft round's verify pass",
            directory: &report_directory,
            file_base: "verify_pass",
        },
    )
    .expect("the verify pass is profiled");
    eprintln!(
        "[verify-pass-profile] job median {:.2} ms (lowest {:.2}, highest {:.2}) gpu {:.2} ms nodes {} pages={}",
        summary.job_wall_median_microseconds / 1_000.0,
        summary.job_wall_lowest_microseconds / 1_000.0,
        summary.job_wall_highest_microseconds / 1_000.0,
        summary.gpu_execution_microseconds / 1_000.0,
        summary.executed_node_count,
        report_directory.display()
    );
}
