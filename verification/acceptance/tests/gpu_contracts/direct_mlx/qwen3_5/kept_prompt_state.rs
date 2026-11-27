//! A prompt read once per loaded pair, the way the `astro_ssd_prompt_cache`
//! spares a served request its prompt: decode journeys read the same story
//! prompt (about 2,000 tokens on a dense target, 10,000 on a mixture of
//! experts) again for every cell and seed, and that reading, not decoding,
//! was most of their wall time.
//!
//! The pair keeps the last prompt it read: its tokens, the decoder state
//! after it, its last logits and the drafter's tap rows. A later read of the
//! same tokens skips the forward and hands back a fresh decoder state seated
//! from the kept one. Seating shares array handles, which is safe because a
//! decode only writes rows after the prompt and never changes a kept row:
//! attention appends from the prompt's offset, and the recurrent and
//! convolution states are replaced, not written in place. Decode
//! measurements may reuse a prompt this way (AGENTS.md); a journey that
//! times prompt reading must not, and none does.

use std::cell::RefCell;

use mlx_bridge::MlxArray;
use model_serving::RequestDecoderStateStack;

use super::dflash2_journey_support::{TargetLogitRows, forward_with_taps, tap_rows};
use super::dflash2_kept_per_round::{
    Cell, JourneyPair, completion_token_ids, load_pair, prompt_token_ids,
};

/// Slab growth for a request's full-attention state, as the journeys have
/// always allocated it.
const FULL_ATTENTION_KV_STATE_GROWTH_TOKENS: i32 = 256;

/// The last prompt a pair read, kept for the next read of the same tokens.
pub(crate) struct KeptPromptState {
    prompt_token_ids: Vec<u32>,
    decoder_state: RequestDecoderStateStack,
    last_logits: MlxArray,
    tap_rows: MlxArray,
}

/// One read prompt, owned by the caller: a decoder state it may decode on,
/// the logits that pick the first token, and the drafter's tap rows.
pub(crate) struct ReadPrompt {
    pub(crate) decoder_state: RequestDecoderStateStack,
    pub(crate) last_logits: MlxArray,
    pub(crate) tap_rows: MlxArray,
    pub(crate) prompt_token_count: usize,
}

/// The cell's prompt as its request sends it: a raw completion or a chat
/// turn.
pub(crate) fn cell_prompt_token_ids(pair: &JourneyPair, cell: &Cell) -> Vec<u32> {
    if cell.is_raw_completion {
        completion_token_ids(pair, cell.prompt)
    } else {
        prompt_token_ids(pair, cell.prompt)
    }
}

/// Reads the cell's prompt, reusing the pair's kept state when it holds
/// these very tokens.
pub(crate) fn read_prompt(pair: &JourneyPair, cell: &Cell) -> ReadPrompt {
    let prompt_token_ids = cell_prompt_token_ids(pair, cell);
    let mut kept_prompt_state = pair.kept_prompt_state.borrow_mut();
    let holds_this_prompt = kept_prompt_state
        .as_ref()
        .is_some_and(|kept| kept.prompt_token_ids == prompt_token_ids);
    if !holds_this_prompt {
        *kept_prompt_state = Some(read_prompt_fresh(pair, prompt_token_ids));
    }
    let kept = kept_prompt_state
        .as_ref()
        .expect("the pair keeps the prompt it just read");
    seated_copy(pair, kept)
}

/// Forwards the prompt from an empty decoder state.
fn read_prompt_fresh(pair: &JourneyPair, prompt_token_ids: Vec<u32>) -> KeptPromptState {
    let mut decoder_state = empty_decoder_state(pair);
    let prompt_forward = forward_with_taps(
        &pair.target,
        &prompt_token_ids,
        0,
        &mut decoder_state,
        &pair.drafter_configuration.target_tap_layer_ids,
        TargetLogitRows::LastRowOnly,
    );
    let tap_rows = tap_rows(
        pair.target.runtime(),
        &prompt_forward.hidden_state_taps,
        prompt_token_ids.len() as i32,
    );
    KeptPromptState {
        prompt_token_ids,
        decoder_state,
        last_logits: prompt_forward.logits,
        tap_rows,
    }
}

/// A fresh decoder state holding the kept prompt, with the kept logits and
/// tap rows, for one caller to decode on.
fn seated_copy(pair: &JourneyPair, kept: &KeptPromptState) -> ReadPrompt {
    let mut decoder_state = empty_decoder_state(pair);
    decoder_state
        .restore_allocation_checkpoint(
            kept.decoder_state
                .allocation_checkpoint()
                .expect("the kept prompt state retains"),
        )
        .expect("the kept prompt state seats");
    ReadPrompt {
        decoder_state,
        last_logits: kept.last_logits.retain().expect("the logits retain"),
        tap_rows: kept.tap_rows.retain().expect("the tap rows retain"),
        prompt_token_count: kept.prompt_token_ids.len(),
    }
}

fn empty_decoder_state(pair: &JourneyPair) -> RequestDecoderStateStack {
    RequestDecoderStateStack::empty_from_astro_decoder_state_cache_layout_with_full_attention_kv_state_growth_tokens(
        &pair.target.astro_decoder_state_cache_layout(),
        FULL_ATTENTION_KV_STATE_GROWTH_TOKENS,
    )
    .expect("the decoder state allocates")
}

/// The empty slot a newly loaded pair starts with.
pub(crate) fn no_kept_prompt_state() -> RefCell<Option<KeptPromptState>> {
    RefCell::new(None)
}

/// Greedy tokens decoded one step at a time on a read prompt's state.
fn greedy_steps(pair: &JourneyPair, read: ReadPrompt, step_count: usize) -> Vec<u32> {
    let ReadPrompt {
        mut decoder_state,
        last_logits,
        prompt_token_count,
        ..
    } = read;
    let mut token_ids = vec![highest_logit_token(pair, &last_logits)];
    for step_index in 0..step_count {
        let step = forward_with_taps(
            &pair.target,
            &[token_ids[step_index]],
            (prompt_token_count + step_index) as u32,
            &mut decoder_state,
            &[],
            TargetLogitRows::LastRowOnly,
        );
        token_ids.push(highest_logit_token(pair, &step.logits));
    }
    token_ids
}

fn highest_logit_token(pair: &JourneyPair, logits: &MlxArray) -> u32 {
    let row = super::dflash2_journey_support::host_values(pair.target.runtime(), logits);
    let mut best_token_id = 0;
    for (token_id, logit) in row.iter().enumerate() {
        if *logit > row[best_token_id] {
            best_token_id = token_id;
        }
    }
    best_token_id as u32
}

/// Greedy steps on a prompt read into a state of its own, then twice on
/// copies seated from the pair's kept state, the second after the first had
/// written its rows: every run must choose the same tokens.
#[test]
#[ignore = "loads a real target and its DFlash2 drafter; run through cargo measure-dflash2-kept-prompt-state"]
#[serial_test::serial]
fn should_decode_the_same_tokens_from_a_kept_prompt_as_from_a_fresh_read() {
    const GREEDY_STEP_COUNT: usize = 32;
    let pair = load_pair();
    let cell = super::dflash2_journey_support::story_cell("story greedy", 0);
    let fresh_tokens = {
        let fresh = read_prompt_fresh(&pair, cell_prompt_token_ids(&pair, &cell));
        let fresh_read = ReadPrompt {
            prompt_token_count: fresh.prompt_token_ids.len(),
            decoder_state: fresh.decoder_state,
            last_logits: fresh.last_logits,
            tap_rows: fresh.tap_rows,
        };
        greedy_steps(&pair, fresh_read, GREEDY_STEP_COUNT)
    };
    let first_kept_tokens = greedy_steps(&pair, read_prompt(&pair, &cell), GREEDY_STEP_COUNT);
    let second_kept_tokens = greedy_steps(&pair, read_prompt(&pair, &cell), GREEDY_STEP_COUNT);
    println!("[kept-prompt-state] fresh={fresh_tokens:?}");
    assert_eq!(first_kept_tokens, fresh_tokens);
    assert_eq!(second_kept_tokens, fresh_tokens);
}
