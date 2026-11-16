//! Splitting scored questions into a fitting part and a held-out part.
//!
//! The held-out part is what decides whether a fitted temperature is kept,
//! so it must be questions the fit never saw. Splitting question by question
//! would put two questions about the same state on both sides, and the
//! held-out score would then partly measure memory of that state; Laya's
//! training notes count such rows as overlap for the same reason. The split
//! is therefore by state: every question about one state lands on one side.
//!
//! The split is deterministic so that running the same examples twice gives
//! the same verdict. States are ordered by a stable hash of their JSON text,
//! which scatters them independently of file order, and every fifth one is
//! held out.

use astronomical_decision_contract::OrderedJsonValue;

use crate::scored_question::ScoredQuestion;

/// One state in this many is held out: 20%, the usual calibration share.
const HOLDOUT_EVERY_NTH_STATE: usize = 5;
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Scored questions, divided so no state appears on both sides.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HoldoutSplit {
    pub fitting_questions: Vec<ScoredQuestion>,
    pub holdout_questions: Vec<ScoredQuestion>,
}

/// A key equal for equal states and stable across runs and builds:
/// 64-bit FNV-1a over the state's compact JSON text.
///
/// `std`'s default hasher is randomly seeded per process, which would move
/// states between sides from one run to the next.
#[must_use]
pub fn state_group_key(state: &OrderedJsonValue) -> u64 {
    let state_text = serde_json::to_string(state).unwrap_or_default();
    let mut state_hash = FNV_OFFSET_BASIS;
    for state_byte in state_text.as_bytes() {
        state_hash ^= u64::from(*state_byte);
        state_hash = state_hash.wrapping_mul(FNV_PRIME);
    }
    state_hash
}

/// Divides `scored_questions` by state; fewer than five distinct states
/// leaves nothing held out, which the calibration gate then refuses.
#[must_use]
pub fn split_by_state(scored_questions: Vec<ScoredQuestion>) -> HoldoutSplit {
    let mut distinct_state_keys = Vec::new();
    for scored_question in &scored_questions {
        distinct_state_keys.push(scored_question.state_group_key);
    }
    distinct_state_keys.sort_unstable();
    distinct_state_keys.dedup();
    let mut held_out_state_keys = Vec::new();
    for (state_position, state_key) in distinct_state_keys.iter().enumerate() {
        if state_position % HOLDOUT_EVERY_NTH_STATE == HOLDOUT_EVERY_NTH_STATE - 1 {
            held_out_state_keys.push(*state_key);
        }
    }
    let mut holdout_split = HoldoutSplit::default();
    for scored_question in scored_questions {
        if held_out_state_keys
            .binary_search(&scored_question.state_group_key)
            .is_ok()
        {
            holdout_split.holdout_questions.push(scored_question);
        } else {
            holdout_split.fitting_questions.push(scored_question);
        }
    }
    holdout_split
}

#[cfg(test)]
#[path = "holdout_split_tests.rs"]
mod tests;
