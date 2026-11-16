//! Gate 4 end to end on the real 27B pair: every round drafts TensorFold's
//! tree, verifies all of it in one target pass, picks with the keyed sampler
//! (or greedily), and commits the kept chain. Prints kept per round and the
//! decode speed (prefill excluded) for TensorFold's code-completion and chat
//! cells, where its engine measured 78.7 and 44.0 tok/s on this Mac.
//!
//! Each stage still waits for the GPU (the lattice and the picks are read
//! back), so this is the speed a straightforward tree loop reaches, not a
//! pipelined serving loop's.

use std::time::Instant;

use astronomical_model_serving::RequestDecoderStateStack;
use astronomical_model_serving::speculative_decoding::tree_rows::TreeRows;
use mlx_bridge::MlxArray;
use serial_test::serial;

use astronomical_model_serving::speculative_decoding::context_copy::{
    ContextCopier, ContextCopySettings,
};

use super::dflash2_kept_per_round::{
    Cell, JourneyPair, OUTPUT_TOKENS, RoundState, completion_token_ids, load_pair, prompt_token_ids,
};
use super::dflash2_tree_oracle::{
    Drafting, FIRST_KEYED_SEED, ORACLE_CELLS, draft_tree, prefill, target_picks,
};

/// Seeds pooled for a sampled cell, as TensorFold's three repetitions.
const SAMPLED_SEED_COUNT: u64 = 3;
/// TensorFold's copied-row budget: it starts at 16 rows past the tree,
/// doubles while copies land whole, halves when most miss, and stops at 48.
const FIRST_COPY_ROWS: usize = 16;
const MAXIMUM_COPY_ROWS: usize = 48;

#[test]
#[ignore = "loads the real 27B and its DFlash2 drafter; run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_decode_code_cells_with_draft_trees_verified_in_one_target_pass() {
    decode_cells_with_trees(&ORACLE_CELLS[..3]);
}

/// The chat cells on their own: a run heats the GPU, and rounds late in a
/// 90-second run measured 6 to 10 ms slower than the same rounds from a
/// cool start, so each family of cells gets its own process.
#[test]
#[ignore = "loads the real 27B and its DFlash2 drafter; run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_decode_chat_cells_with_draft_trees_verified_in_one_target_pass() {
    // Sampled first: it is the cell compared with TensorFold's 44.0.
    decode_cells_with_trees(&[ORACLE_CELLS[4], ORACLE_CELLS[3]]);
}

fn decode_cells_with_trees(cells: &[Cell]) {
    let mut pair = load_pair();
    pair.target
        .enable_draft_tree_verification()
        .expect("the tree kernel declares");
    for (cell_index, cell) in cells.iter().enumerate() {
        let seed_count = if cell.temperature_thousandths == 0 {
            1
        } else {
            SAMPLED_SEED_COUNT
        };
        // Alternating the two in one process keeps drift out of the comparison.
        let mut pooled = [PooledTreeDecode::default(), PooledTreeDecode::default()];
        for seed_offset in 0..seed_count {
            let seeded_cell = Cell {
                keyed_seed: FIRST_KEYED_SEED + seed_offset,
                ..*cell
            };
            for (pooled_decode, copies_context) in pooled.iter_mut().zip([false, true]) {
                pooled_decode.add(&decode_with_trees(&pair, &seeded_cell, copies_context));
            }
        }
        for (pooled_decode, label) in pooled.iter().zip(["tree", "tree + context copies"]) {
            eprintln!(
                "[dflash2-tree-rounds {}/{}] {} ({seed_count} seeds), {label}: {:.2} kept per \
                 round, {:.1} tok/s decode ({} tokens in {} rounds, {:.1} ms a round)",
                cell_index + 1,
                cells.len(),
                cell.label,
                pooled_decode.emitted as f64 / pooled_decode.rounds as f64,
                pooled_decode.emitted as f64 / pooled_decode.decode_seconds,
                pooled_decode.emitted,
                pooled_decode.rounds,
                pooled_decode.decode_seconds * 1000.0 / pooled_decode.rounds as f64,
            );
        }
        print_segments(pooled[0].rounds + pooled[1].rounds);
    }
}

/// Tree decodes pooled over seeds.
#[derive(Default)]
struct PooledTreeDecode {
    rounds: usize,
    emitted: usize,
    decode_seconds: f64,
}

impl PooledTreeDecode {
    fn add(&mut self, tree: &TreeDecode) {
        self.rounds += tree.rounds;
        self.emitted += tree.emitted_token_ids.len() - 1;
        self.decode_seconds += tree.decode_seconds;
    }
}

struct TreeDecode {
    rounds: usize,
    /// The prefill's pick, then every token the rounds kept.
    emitted_token_ids: Vec<u32>,
    decode_seconds: f64,
}

/// The request's text so far and its copier, when the round adds copies.
struct CopyContext {
    history: Vec<u32>,
    copier: ContextCopier,
    copy_row_budget: usize,
}

fn decode_with_trees(pair: &JourneyPair, cell: &Cell, copies_context: bool) -> TreeDecode {
    let (mut round, mut decoder_state, first_token_id) = prefill(pair, cell);
    let mut copy_context = if copies_context {
        let mut history = if cell.is_raw_completion {
            completion_token_ids(pair, cell.prompt)
        } else {
            prompt_token_ids(pair, cell.prompt)
        };
        history.push(first_token_id);
        Some(CopyContext {
            history,
            copier: ContextCopier::new(ContextCopySettings::default()),
            copy_row_budget: FIRST_COPY_ROWS,
        })
    } else {
        None
    };
    let decode_started_at = Instant::now();
    let mut emitted_token_ids = vec![first_token_id];
    let mut rounds = 0;
    while emitted_token_ids.len() <= OUTPUT_TOKENS {
        let emitted = tree_round(
            pair,
            cell,
            &mut round,
            &mut decoder_state,
            copy_context.as_mut(),
        );
        if let Some(copy_context) = copy_context.as_mut() {
            copy_context.history.extend_from_slice(&emitted);
        }
        emitted_token_ids.extend(emitted);
        rounds += 1;
    }
    TreeDecode {
        rounds,
        emitted_token_ids,
        decode_seconds: decode_started_at.elapsed().as_secs_f64(),
    }
}

/// Drafts a tree, hangs a context copy below it when given a copier,
/// verifies every row, keeps the chain the target's picks follow and
/// commits it. Returns the tokens the round emitted.
fn tree_round(
    pair: &JourneyPair,
    cell: &Cell,
    round: &mut RoundState,
    decoder_state: &mut RequestDecoderStateStack,
    mut copy_context: Option<&mut CopyContext>,
) -> Vec<u32> {
    let segment_started_at = Instant::now();
    let tree = draft_tree(pair, cell, round, Drafting::Tree);
    let segment_started_at = record_segment(0, segment_started_at);
    let mut rows = TreeRows::from_draft_tree(round.last_token_id, &tree);
    let first_copied_row = rows.row_count();
    let mut copied_row_count = 0;
    if let Some(copy_context) = copy_context.as_deref_mut() {
        let copied = copy_context.copier.propose(
            &copy_context.history,
            copy_context.copy_row_budget + tree.token_ids.len(),
        );
        copied_row_count = rows.add_copied_chain(&copied);
    }
    let verification = pair
        .target
        .verify_draft_tree(
            &rows,
            round.fed_token_count,
            decoder_state,
            &pair.drafter_configuration.target_tap_layer_ids,
        )
        .expect("the tree verifies");
    let segment_started_at = record_segment(1, segment_started_at);
    // A row's pick takes the position after the row's own.
    let mut pick_positions = Vec::with_capacity(rows.row_count());
    for row_position in rows.positions(round.fed_token_count as i32) {
        pick_positions.push(row_position as u64 + 1);
    }
    let picks = target_picks(pair, cell, &verification.per_row_logits, &pick_positions);
    let segment_started_at = record_segment(2, segment_started_at);
    let kept = rows.kept_chain(&picks);
    if let Some(copy_context) = copy_context {
        let mut landed = 0;
        for kept_row in &kept.rows {
            if *kept_row >= first_copied_row {
                landed += 1;
            }
        }
        copy_context.copier.observe(copied_row_count, landed);
        adapt_copy_row_budget(copy_context, copied_row_count, landed);
    }
    round.pending_tap_rows = kept_tap_rows(pair, &verification.hidden_state_taps, &kept.rows);
    pair.target
        .commit_draft_tree(decoder_state, verification, &kept.rows)
        .expect("the kept chain commits");
    record_segment(3, segment_started_at);
    let mut emitted = Vec::with_capacity(kept.rows.len());
    for kept_row in &kept.rows[1..] {
        emitted.push(rows.token_ids[*kept_row]);
    }
    emitted.push(kept.next_token_id);
    round.fed_token_count += kept.rows.len() as u32;
    round.last_token_id = kept.next_token_id;
    emitted
}

/// Wall time an unsynced round spends in each segment, in microseconds:
/// drafting through the lattice readback, the verify's graph build, the
/// wait for the picks, and the commit's graph build. While the first and
/// third wait on the GPU, the second overlaps it; the fourth leaves it idle.
static SEGMENT_MICROSECONDS: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
const SEGMENT_LABELS: [&str; 4] = [
    "draft + lattice",
    "verify build",
    "picks wait",
    "commit build",
];

fn record_segment(segment_index: usize, segment_started_at: Instant) -> Instant {
    SEGMENT_MICROSECONDS[segment_index].fetch_add(
        segment_started_at.elapsed().as_micros() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    Instant::now()
}

/// Prints each segment's milliseconds a round and clears the totals.
fn print_segments(round_count: usize) {
    let mut line = String::new();
    for (label, total) in SEGMENT_LABELS.iter().zip(SEGMENT_MICROSECONDS.iter()) {
        let microseconds = total.swap(0, std::sync::atomic::Ordering::Relaxed);
        line.push_str(&format!(
            " {label} {:.1} ms,",
            microseconds as f64 / 1000.0 / round_count as f64
        ));
    }
    eprintln!("[dflash2-tree-rounds] a round, unsynced:{line}");
}

/// TensorFold's rule: double the copied rows while copies land whole, halve
/// them (not below the first budget) when fewer than half land.
fn adapt_copy_row_budget(copy_context: &mut CopyContext, copied_row_count: usize, landed: usize) {
    if copied_row_count == 0 {
        return;
    }
    if landed == copied_row_count {
        copy_context.copy_row_budget = (copy_context.copy_row_budget * 2).min(MAXIMUM_COPY_ROWS);
    } else if 2 * landed < copied_row_count {
        copy_context.copy_row_budget = (copy_context.copy_row_budget / 2).max(FIRST_COPY_ROWS);
    }
}

/// The drafter's next input: every tap's kept rows, in chain order, joined.
pub(super) fn kept_tap_rows(
    pair: &JourneyPair,
    layer_taps: &[MlxArray],
    kept_rows: &[usize],
) -> MlxArray {
    let runtime = pair.target.runtime();
    let mut kept_row_ids = Vec::with_capacity(kept_rows.len());
    for kept_row in kept_rows {
        kept_row_ids.push(*kept_row as i32);
    }
    let kept_row_array = runtime
        .array_from_i32(&kept_row_ids, &[kept_rows.len() as i32])
        .expect("the kept rows build");
    let mut kept_taps = Vec::with_capacity(layer_taps.len());
    for layer_tap in layer_taps {
        kept_taps.push(
            runtime
                .take_axis(layer_tap, &kept_row_array, 1)
                .expect("the kept tap rows gather"),
        );
    }
    let references: Vec<&MlxArray> = kept_taps.iter().collect();
    runtime
        .concatenate_axis(&references, -1)
        .expect("the taps join")
}
