//! Gate 2 baseline: where one draft-and-verify round's time goes on a real
//! target and its DFlash2 drafter (the 27B pair unless the test run
//! configures another target), beside a plain one-token decode step, at a
//! short context.
//!
//! Each stage is evaluated on its own so its time can be read; that costs a
//! GPU synchronization per stage, so the stages sum to a little more than a
//! fused round would. This is a performance measurement and asserts nothing.

use std::time::{Duration, Instant};

use mlx_bridge::MlxArray;
use model_serving::RequestDecoderStateStack;
use model_serving::speculative_decoding::dflash_drafter::{
    DFlashDraftTarget, DrafterContextStates, PathChoice, drafted_hidden_rows, select_path,
};
use serial_test::serial;

use super::dflash2_journey_support::{
    TargetForward, TargetLogitRows, forward_with_taps, host_values, tap_rows,
    verify_block_with_taps,
};
use super::dflash2_kept_per_round::{JourneyPair, load_pair, prompt_token_ids};

const MEASURED_ROUND_COUNT: usize = 12;
const WARM_UP_ROUND_COUNT: usize = 2;
/// Any vocabulary token works as the anchor; the stage times do not depend on it.
const ANCHOR_TOKEN_ID: u32 = 17;
const PROMPT: &str =
    "Write a short Python function that computes the Fibonacci sequence and explain it.";

/// Median milliseconds per stage over the measured rounds.
#[derive(Default)]
struct StageTimes {
    plain_decode_step: Vec<Duration>,
    drafter_pass: Vec<Duration>,
    draft_head: Vec<Duration>,
    selector_and_readback: Vec<Duration>,
    verify_eight_rows: Vec<Duration>,
    verify_logits_readback: Vec<Duration>,
    rewind_restore_and_rerun: Vec<Duration>,
    /// Diagnostic: the same verify without the per-row DeltaNet states, the
    /// share of the verify that row-state capture costs.
    verify_without_row_states: Vec<Duration>,
}

#[test]
#[ignore = "loads a real target and its DFlash2 drafter (the 27B pair unless configured); run through cargo measure-dflash2-round-cost"]
#[serial]
fn should_measure_where_a_drafted_round_spends_its_time_on_the_27b() {
    let pair = load_pair();
    let prompt = prompt_token_ids(&pair, PROMPT);
    let mut times = StageTimes::default();
    for round_index in 0..WARM_UP_ROUND_COUNT + MEASURED_ROUND_COUNT {
        let round_times = time_one_round(&pair, &prompt);
        if round_index >= WARM_UP_ROUND_COUNT {
            for (stage_times, stage_time) in times.all_mut().into_iter().zip(round_times) {
                stage_times.push(stage_time);
            }
        }
        eprintln!(
            "[dflash2-round-cost] round {}/{}",
            round_index + 1,
            WARM_UP_ROUND_COUNT + MEASURED_ROUND_COUNT
        );
    }
    report(&mut times);
}

impl StageTimes {
    fn all_mut(&mut self) -> [&mut Vec<Duration>; 8] {
        [
            &mut self.plain_decode_step,
            &mut self.drafter_pass,
            &mut self.draft_head,
            &mut self.selector_and_readback,
            &mut self.verify_eight_rows,
            &mut self.verify_logits_readback,
            &mut self.rewind_restore_and_rerun,
            &mut self.verify_without_row_states,
        ]
    }
}

/// One fresh request: prefill, then one plain step and one drafted round,
/// each stage evaluated and timed. Returns the eight stage times in order.
fn time_one_round(pair: &JourneyPair, prompt: &[u32]) -> [Duration; 8] {
    let runtime = pair.target.runtime();
    let mut decoder_state = RequestDecoderStateStack::empty_from_astro_decoder_state_cache_layout_with_full_attention_kv_state_growth_tokens(
        &pair.target.astro_decoder_state_cache_layout(),
        256,
    )
    .expect("the decoder state allocates");
    let tap_layer_ids = &pair.drafter_configuration.target_tap_layer_ids;
    let prefill = forward(
        pair,
        prompt,
        0,
        &mut decoder_state,
        tap_layer_ids,
        TargetLogitRows::LastRowOnly,
    );
    let pending_taps = tap_rows(runtime, &prefill.hidden_state_taps, prompt.len() as i32);
    evaluate(runtime, &[&pending_taps]);
    let fed = prompt.len() as u32;
    let plain_decode_step = time_plain_step(pair, &mut decoder_state, fed);
    let (drafter_pass, draft_head, selector_and_readback, drafted_ids) =
        time_drafting(pair, &pending_taps);
    let [
        verify_eight_rows,
        verify_logits_readback,
        rewind_restore_and_rerun,
    ] = time_verify_and_rewind(pair, &mut decoder_state, fed, &drafted_ids);
    // Last, since it advances the state with nothing to rewind to; each
    // round is a fresh request.
    let verify_without_row_states =
        time_verify_without_row_states(pair, &mut decoder_state, fed + 4, &drafted_ids);
    [
        plain_decode_step,
        drafter_pass,
        draft_head,
        selector_and_readback,
        verify_eight_rows,
        verify_logits_readback,
        rewind_restore_and_rerun,
        verify_without_row_states,
    ]
}

fn time_verify_without_row_states(
    pair: &JourneyPair,
    decoder_state: &mut RequestDecoderStateStack,
    fed: u32,
    drafted_ids: &[u32],
) -> Duration {
    let mut verified_ids = vec![ANCHOR_TOKEN_ID];
    verified_ids.extend_from_slice(drafted_ids);
    let started = Instant::now();
    let verification = forward(
        pair,
        &verified_ids,
        fed,
        decoder_state,
        &pair.drafter_configuration.target_tap_layer_ids,
        TargetLogitRows::EveryRow,
    );
    let mut evaluated_arrays: Vec<&MlxArray> = verification.hidden_state_taps.iter().collect();
    evaluated_arrays.push(&verification.logits);
    evaluate(pair.target.runtime(), &evaluated_arrays);
    started.elapsed()
}

fn forward(
    pair: &JourneyPair,
    token_ids: &[u32],
    starting_position: u32,
    decoder_state: &mut RequestDecoderStateStack,
    tap_layer_ids: &[i32],
    logit_rows: TargetLogitRows,
) -> TargetForward {
    forward_with_taps(
        &pair.target,
        token_ids,
        starting_position,
        decoder_state,
        tap_layer_ids,
        logit_rows,
    )
}

/// One plain row, then the state is put back for the drafted round.
fn time_plain_step(
    pair: &JourneyPair,
    decoder_state: &mut RequestDecoderStateStack,
    fed: u32,
) -> Duration {
    let checkpoint = decoder_state.checkpoint().expect("checkpoint");
    let started = Instant::now();
    let plain = forward(
        pair,
        &[ANCHOR_TOKEN_ID],
        fed,
        decoder_state,
        &[],
        TargetLogitRows::LastRowOnly,
    );
    evaluate(pair.target.runtime(), &[&plain.logits]);
    let plain_decode_step = started.elapsed();
    decoder_state
        .restore_checkpoint(checkpoint)
        .expect("restore");
    plain_decode_step
}

/// Drafter pass, draft head, then the selector walk with the path read back.
fn time_drafting(
    pair: &JourneyPair,
    pending_taps: &MlxArray,
) -> (Duration, Duration, Duration, Vec<u32>) {
    let runtime = pair.target.runtime();
    let configuration = &pair.drafter_configuration;
    let mut context_states = DrafterContextStates::new(configuration, 0);
    let mut block_ids = vec![ANCHOR_TOKEN_ID];
    block_ids.extend(std::iter::repeat_n(
        configuration.mask_token_id,
        configuration.proposal_count() as usize,
    ));
    let block = runtime
        .array_from_u32(&block_ids, &[1, configuration.block_size])
        .expect("block");
    let target: &dyn DFlashDraftTarget = &*pair.target;
    let started = Instant::now();
    let hidden = drafted_hidden_rows(
        runtime,
        &pair.drafter_weights,
        configuration,
        target,
        &mut context_states,
        &block,
        pending_taps,
    )
    .expect("the drafter runs");
    evaluate(runtime, &[&hidden]);
    let drafter_pass = started.elapsed();
    let started = Instant::now();
    let draft_logits = target.head_logits(runtime, &hidden).expect("head");
    evaluate(runtime, &[&draft_logits]);
    let draft_head = started.elapsed();
    let started = Instant::now();
    let anchor = runtime
        .array_from_u32(&[ANCHOR_TOKEN_ID], &[1])
        .expect("anchor");
    let path = select_path(
        runtime,
        pair.drafter_weights
            .candidate_selector
            .as_ref()
            .expect("a DFlash2 drafter has a candidate selector"),
        configuration,
        &hidden,
        &draft_logits,
        &anchor,
        &mut PathChoice::Greedy,
    )
    .expect("selector");
    let drafted_ids: Vec<u32> = host_values(runtime, &path.token_ids)
        .iter()
        .map(|value| *value as u32)
        .collect();
    (drafter_pass, draft_head, started.elapsed(), drafted_ids)
}

/// The 8-row verify (capturing DeltaNet state after each row), its readback,
/// and the rewind to four kept rows from those captures.
fn time_verify_and_rewind(
    pair: &JourneyPair,
    decoder_state: &mut RequestDecoderStateStack,
    fed: u32,
    drafted_ids: &[u32],
) -> [Duration; 3] {
    let runtime = pair.target.runtime();
    let mut verified_ids = vec![ANCHOR_TOKEN_ID];
    verified_ids.extend_from_slice(drafted_ids);
    let started = Instant::now();
    // The streaming model evaluates the block, its row states among it,
    // before it returns, as a drafting engine's verify does.
    let verified = verify_block_with_taps(
        &pair.target,
        &verified_ids,
        fed,
        decoder_state,
        &pair.drafter_configuration.target_tap_layer_ids,
    );
    let mut evaluated_arrays: Vec<&MlxArray> = verified.hidden_state_taps.iter().collect();
    evaluated_arrays.push(&verified.verification.per_row_logits);
    evaluate(runtime, &evaluated_arrays);
    let verify_eight_rows = started.elapsed();
    let started = Instant::now();
    let _argmax = host_values(
        runtime,
        &runtime
            .argmax_axis(&verified.verification.per_row_logits, -1)
            .expect("argmax"),
    );
    let verify_logits_readback = started.elapsed();
    let started = Instant::now();
    pair.target
        .keep_drafted_rows(verified.verification, 4, decoder_state)
        .expect("the state rewinds");
    [verify_eight_rows, verify_logits_readback, started.elapsed()]
}

fn evaluate(runtime: &runtime_integration::MlxRuntime, arrays: &[&MlxArray]) {
    runtime
        .evaluate_arrays(arrays)
        .expect("the stage evaluates");
}

fn report(times: &mut StageTimes) {
    let labels = [
        "plain one-row decode step",
        "drafter pass (4-bit, 8 rows)",
        "target head on 7 draft rows",
        "selector walk + path readback",
        "target verify, 8 rows + row states",
        "verify logits readback (argmax)",
        "rewind to 4 kept rows (no rerun)",
    ];
    let mut round_total = Duration::ZERO;
    for (label, stage_times) in labels.iter().zip(times.all_mut()) {
        stage_times.sort();
        let median = stage_times[stage_times.len() / 2];
        if !label.starts_with("plain") {
            round_total += median;
        }
        eprintln!(
            "[dflash2-round-cost] {label:<38} {:7.1} ms",
            median.as_secs_f64() * 1000.0
        );
    }
    eprintln!(
        "[dflash2-round-cost] {:<38} {:7.1} ms",
        "drafted round total",
        round_total.as_secs_f64() * 1000.0
    );
    let diagnostic_times = &mut times.verify_without_row_states;
    diagnostic_times.sort();
    eprintln!(
        "[dflash2-round-cost] {:<38} {:7.1} ms (diagnostic, not in the total)",
        "target verify, 8 rows, no row states",
        diagnostic_times[diagnostic_times.len() / 2].as_secs_f64() * 1000.0
    );
}
