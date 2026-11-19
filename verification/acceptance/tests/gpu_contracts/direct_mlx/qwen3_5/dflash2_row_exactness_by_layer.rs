//! Where inside the target a window of rows first leaves the one-token
//! step: every decoder layer's output hidden states, fed the target's greedy
//! continuation one token a step against as one plain window of 4 and of 16
//! rows. A row that drifts at a layer carries that drift into every later
//! one, so the first differing layer, and whether it is a DeltaNet or a
//! full-attention layer, names the operation to make row-exact first.
//! `batched_kernel_row_exactness` measures each kernel alone below this.
//! It reports; it asserts nothing.

use model_serving::{PerformanceAttribution, VerificationLogitsMode};
use serial_test::serial;

use super::dflash2_journey_support::host_values;
use super::dflash2_kept_per_round::{JourneyPair, load_pair};
use super::dflash2_tree_oracle::{ORACLE_CELLS, prefill};

const LONGEST_WINDOW_ROW_COUNT: usize = 16;
const WINDOW_ROW_COUNTS: [usize; 2] = [4, 16];
const GREEDY_CHAT_CELL_INDEX: usize = 3;
/// Every fourth layer is full attention on the Qwen 3.5 family.
const FULL_ATTENTION_INTERVAL: usize = 4;

/// `[layer][row]` hidden states, each `hidden size` long.
type LayerRows = Vec<Vec<Vec<f32>>>;

#[test]
#[ignore = "loads a real target and its DFlash2 drafter; run through cargo measure-dflash2-row-exactness"]
#[serial]
fn should_locate_the_layer_where_window_rows_first_leave_the_one_token_step() {
    let pair = load_pair();
    // The drafter names the target layer count it pairs with.
    let layer_count = pair.drafter_configuration.target_layer_count as usize;
    let every_layer: Vec<i32> = (0..layer_count as i32).collect();
    let (tokens, step_layers) = one_token_step_layers(&pair, &every_layer);
    for window_row_count in WINDOW_ROW_COUNTS {
        let window_layers = window_forward_layers(&pair, &every_layer, &tokens[..window_row_count]);
        report_layers(window_row_count, &step_layers, &window_layers);
    }
}

/// The greedy continuation fed one token a step, with every layer's output
/// for each step: `[layer][step]`.
fn one_token_step_layers(pair: &JourneyPair, tap_layer_ids: &[i32]) -> (Vec<u32>, LayerRows) {
    let (round, mut state, first_token_id) = prefill(pair, &ORACLE_CELLS[GREEDY_CHAT_CELL_INDEX]);
    let mut fed_token_count = round.fed_token_count;
    let mut tokens = vec![first_token_id];
    let mut layers: LayerRows = vec![Vec::new(); tap_layer_ids.len()];
    for step_index in 0..LONGEST_WINDOW_ROW_COUNT {
        let step = pair
            .target
            .build_speculative_verification_forward(
                &tokens[step_index..=step_index],
                fed_token_count,
                &mut state,
                tap_layer_ids,
                VerificationLogitsMode::AllPositions,
                &mut PerformanceAttribution::disabled(),
            )
            .expect("the one-token step runs");
        for (layer_index, tap) in step.hidden_state_taps.iter().enumerate() {
            layers[layer_index].push(host_values(pair.target.runtime(), tap));
        }
        let logits = host_values(
            pair.target.runtime(),
            &step.per_position_logits.expect("the step has logits"),
        );
        tokens.push(highest_logit_token(&logits));
        fed_token_count += 1;
    }
    (tokens, layers)
}

/// The same tokens as one plain forward over a window: `[layer][row]`.
fn window_forward_layers(pair: &JourneyPair, tap_layer_ids: &[i32], tokens: &[u32]) -> LayerRows {
    let (round, mut state, _) = prefill(pair, &ORACLE_CELLS[GREEDY_CHAT_CELL_INDEX]);
    let forward = pair
        .target
        .build_speculative_verification_forward(
            tokens,
            round.fed_token_count,
            &mut state,
            tap_layer_ids,
            VerificationLogitsMode::LastRowOnly,
            &mut PerformanceAttribution::disabled(),
        )
        .expect("the window forward runs");
    let mut layers = Vec::with_capacity(forward.hidden_state_taps.len());
    for tap in &forward.hidden_state_taps {
        let values = host_values(pair.target.runtime(), tap);
        let hidden_size = values.len() / tokens.len();
        layers.push(values.chunks(hidden_size).map(<[f32]>::to_vec).collect());
    }
    layers
}

fn report_layers(window_row_count: usize, step_layers: &LayerRows, window_layers: &LayerRows) {
    let mut first_differing_layer = None;
    let mut summary = String::new();
    for (layer_index, window_rows) in window_layers.iter().enumerate() {
        let mut largest = 0.0_f32;
        let mut differing_row_count = 0;
        for (row_index, window_row) in window_rows.iter().enumerate() {
            let difference = largest_difference(&step_layers[layer_index][row_index], window_row);
            if difference > 0.0 {
                differing_row_count += 1;
            }
            largest = largest.max(difference);
        }
        if differing_row_count > 0 && first_differing_layer.is_none() {
            first_differing_layer = Some(layer_index);
        }
        let layer_kind = if (layer_index + 1) % FULL_ATTENTION_INTERVAL == 0 {
            "A"
        } else {
            "D"
        };
        summary.push_str(&format!(
            " {layer_index}{layer_kind}:{largest:.3}/{differing_row_count}"
        ));
    }
    eprintln!(
        "[dflash2-row-exactness-by-layer] {window_row_count}-row window: first differing layer {first_differing_layer:?} (D = DeltaNet, A = full attention; max diff/rows differing):{summary}"
    );
}

fn highest_logit_token(row: &[f32]) -> u32 {
    let mut best_index = 0;
    for (token_index, logit) in row.iter().enumerate() {
        if *logit > row[best_index] {
            best_index = token_index;
        }
    }
    best_index as u32
}

fn largest_difference(reference: &[f32], candidate: &[f32]) -> f32 {
    let mut largest = 0.0_f32;
    for (reference_value, candidate_value) in reference.iter().zip(candidate) {
        largest = largest.max((reference_value - candidate_value).abs());
    }
    largest
}
