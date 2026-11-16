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

use super::dflash2_kept_per_round::{Cell, JourneyPair, OUTPUT_TOKENS, RoundState, load_pair};
use super::dflash2_tree_oracle::{Drafting, ORACLE_CELLS, draft_tree, prefill, target_picks};

#[test]
#[ignore = "loads the real 27B and its DFlash2 drafter; run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_decode_with_draft_trees_verified_in_one_target_pass() {
    let mut pair = load_pair();
    pair.target
        .enable_draft_tree_verification()
        .expect("the tree kernel declares");
    for (cell_index, cell) in ORACLE_CELLS.iter().enumerate() {
        let (rounds, emitted, decode_seconds) = decode_with_trees(&pair, cell);
        eprintln!(
            "[dflash2-tree-rounds {}/{}] {}: {:.2} kept per round, {:.1} tok/s decode \
             ({emitted} tokens in {rounds} rounds, {:.1} ms a round)",
            cell_index + 1,
            ORACLE_CELLS.len(),
            cell.label,
            emitted as f64 / rounds as f64,
            emitted as f64 / decode_seconds,
            decode_seconds * 1000.0 / rounds as f64,
        );
    }
}

/// Returns (rounds, tokens emitted by them, decode seconds).
fn decode_with_trees(pair: &JourneyPair, cell: &Cell) -> (usize, usize, f64) {
    let (mut round, mut decoder_state, _first_token_id) = prefill(pair, cell);
    let decode_started_at = Instant::now();
    let (mut rounds, mut emitted) = (0, 0);
    while emitted < OUTPUT_TOKENS {
        emitted += tree_round(pair, cell, &mut round, &mut decoder_state);
        rounds += 1;
    }
    (rounds, emitted, decode_started_at.elapsed().as_secs_f64())
}

/// Drafts a tree, verifies it, keeps the chain the target's picks follow and
/// commits it. Returns the tokens the round emitted.
fn tree_round(
    pair: &JourneyPair,
    cell: &Cell,
    round: &mut RoundState,
    decoder_state: &mut RequestDecoderStateStack,
) -> usize {
    let tree = draft_tree(pair, cell, round, Drafting::Tree);
    let rows = TreeRows::from_draft_tree(round.last_token_id, &tree);
    let verification = pair
        .target
        .verify_draft_tree(
            &rows,
            round.fed_token_count,
            decoder_state,
            &pair.drafter_configuration.target_tap_layer_ids,
        )
        .expect("the tree verifies");
    // A row's pick takes the position after the row's own.
    let mut pick_positions = Vec::with_capacity(rows.row_count());
    for row_position in rows.positions(round.fed_token_count as i32) {
        pick_positions.push(row_position as u64 + 1);
    }
    let picks = target_picks(pair, cell, &verification.per_row_logits, &pick_positions);
    let kept = rows.kept_chain(&picks);
    round.pending_tap_rows = kept_tap_rows(pair, &verification.hidden_state_taps, &kept.rows);
    pair.target
        .commit_draft_tree(decoder_state, verification, &kept.rows)
        .expect("the kept chain commits");
    round.fed_token_count += kept.rows.len() as u32;
    round.last_token_id = kept.next_token_id;
    kept.rows.len()
}

/// The drafter's next input: every tap's kept rows, in chain order, joined.
fn kept_tap_rows(pair: &JourneyPair, layer_taps: &[MlxArray], kept_rows: &[usize]) -> MlxArray {
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

const BREAKDOWN_ROUND_COUNT: usize = 24;
const BREAKDOWN_WARM_UP_ROUNDS: usize = 4;
const BREAKDOWN_STAGES: [&str; 6] = [
    "drafter pass (4-bit, 8 rows)",
    "draft head + lattice readback",
    "tree search (CPU)",
    "target tree verify (16 rows)",
    "picks readback (keyed sample)",
    "commit kept chain",
];

#[test]
#[ignore = "loads the real 27B and its DFlash2 drafter; run through cargo measure-dflash2-tree-rounds"]
#[serial]
fn should_measure_where_a_tree_round_spends_its_time() {
    let mut pair = load_pair();
    pair.target
        .enable_draft_tree_verification()
        .expect("the tree kernel declares");
    let cell = &ORACLE_CELLS[3];
    let (mut round, mut decoder_state, _first_token_id) = prefill(&pair, cell);
    let mut stage_times: Vec<Vec<f64>> = vec![Vec::new(); BREAKDOWN_STAGES.len()];
    for round_index in 0..BREAKDOWN_ROUND_COUNT {
        let round_stage_times = timed_tree_round(&pair, cell, &mut round, &mut decoder_state);
        if round_index >= BREAKDOWN_WARM_UP_ROUNDS {
            for (stage_index, stage_time) in round_stage_times.iter().enumerate() {
                stage_times[stage_index].push(*stage_time);
            }
        }
    }
    let mut round_total = 0.0;
    for (label, times) in BREAKDOWN_STAGES.iter().zip(stage_times.iter_mut()) {
        times.sort_by(f64::total_cmp);
        let median = times[times.len() / 2];
        round_total += median;
        eprintln!("[dflash2-tree-round-cost] {label:<34} {median:7.1} ms");
    }
    eprintln!(
        "[dflash2-tree-round-cost] {:<34} {round_total:7.1} ms (chat 1.0, every stage synced)",
        "tree round total"
    );
}

/// One tree round with every stage evaluated on its own: milliseconds per
/// stage in `BREAKDOWN_STAGES` order.
fn timed_tree_round(
    pair: &JourneyPair,
    cell: &Cell,
    round: &mut RoundState,
    decoder_state: &mut RequestDecoderStateStack,
) -> Vec<f64> {
    let runtime = pair.target.runtime();
    let mut stage_times = Vec::with_capacity(BREAKDOWN_STAGES.len());
    let mut stage_started_at = Instant::now();
    let staged = super::dflash2_tree_oracle::staged_draft_tree(pair, cell, round);
    stage_times.push(staged.drafter_milliseconds);
    stage_times.push(staged.lattice_milliseconds);
    stage_times.push(staged.search_milliseconds);
    let rows = TreeRows::from_draft_tree(round.last_token_id, &staged.tree);
    stage_started_at = restart(&mut stage_times, stage_started_at, false);
    let verification = pair
        .target
        .verify_draft_tree(
            &rows,
            round.fed_token_count,
            decoder_state,
            &pair.drafter_configuration.target_tap_layer_ids,
        )
        .expect("the tree verifies");
    let mut verified_arrays: Vec<&MlxArray> = verification.hidden_state_taps.iter().collect();
    verified_arrays.push(&verification.per_row_logits);
    runtime
        .evaluate_arrays(&verified_arrays)
        .expect("the verify evaluates");
    stage_started_at = restart(&mut stage_times, stage_started_at, true);
    let mut pick_positions = Vec::with_capacity(rows.row_count());
    for row_position in rows.positions(round.fed_token_count as i32) {
        pick_positions.push(row_position as u64 + 1);
    }
    let picks = target_picks(pair, cell, &verification.per_row_logits, &pick_positions);
    let kept = rows.kept_chain(&picks);
    stage_started_at = restart(&mut stage_times, stage_started_at, true);
    round.pending_tap_rows = kept_tap_rows(pair, &verification.hidden_state_taps, &kept.rows);
    pair.target
        .commit_draft_tree(decoder_state, verification, &kept.rows)
        .expect("the kept chain commits");
    evaluate_decoder_state(pair, decoder_state, &round.pending_tap_rows);
    restart(&mut stage_times, stage_started_at, true);
    round.fed_token_count += kept.rows.len() as u32;
    round.last_token_id = kept.next_token_id;
    stage_times
}

/// Records the time since `stage_started_at` when `should_record`, and
/// returns a fresh start.
fn restart(stage_times: &mut Vec<f64>, stage_started_at: Instant, should_record: bool) -> Instant {
    if should_record {
        stage_times.push(stage_started_at.elapsed().as_secs_f64() * 1000.0);
    }
    Instant::now()
}

/// Waits for every layer's state, so the commit's GPU work is timed.
fn evaluate_decoder_state(
    pair: &JourneyPair,
    decoder_state: &mut RequestDecoderStateStack,
    pending_tap_rows: &MlxArray,
) {
    use astronomical_model_serving::AstroDecoderStateCacheState;
    let mut state_arrays: Vec<MlxArray> = Vec::new();
    for layer_index in 0..decoder_state.layer_count() {
        match decoder_state.layer_mut(layer_index) {
            Some(AstroDecoderStateCacheState::Composite {
                convolution,
                recurrent,
            }) => {
                for array in [convolution.state(), recurrent.state()]
                    .into_iter()
                    .flatten()
                {
                    state_arrays.push(array.retain().expect("the state retains"));
                }
            }
            Some(AstroDecoderStateCacheState::AppendOnlyAttention { attention }) => {
                for array in [attention.keys_state(), attention.values_state()]
                    .into_iter()
                    .flatten()
                {
                    state_arrays.push(array.retain().expect("the state retains"));
                }
            }
            None => {}
        }
    }
    let mut references: Vec<&MlxArray> = state_arrays.iter().collect();
    references.push(pending_tap_rows);
    pair.target
        .runtime()
        .evaluate_arrays(&references)
        .expect("the committed state evaluates");
}
