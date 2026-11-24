//! Which part of a training each labelled question is in, decided by state.
//!
//! - **training** (about 70%): the only questions whose labels move weights
//! - **choosing** (about 15%): picks when to stop and which seed wins, and
//!   fits the trained model's temperatures
//! - **judging** (about 15%): read once, at the end, to judge the result
//!
//! Judging questions never influence a choice; if they did, the judgement
//! would partly measure how well the choices fitted them. As in calibration,
//! the split is by state, so two questions about one state never land on two
//! sides (Laya's training notes count such rows as overlap), and it is
//! deterministic: the same examples always split the same way.
//!
//! Any amount of data trains; nothing here refuses a file. Below
//! [`HOLD_BACK_STATE_COUNT`] different states, holding states back would
//! leave a part with none, so every state trains instead and the report
//! says the result was only compared on the examples it learned from.

use std::collections::HashMap;

/// Out of every this many distinct states (ordered by their stable hash),
/// the first three are judged and the next three choose.
const STATE_POSITIONS_PER_CYCLE: usize = 20;
const JUDGING_POSITIONS: usize = 3;
const CHOOSING_POSITIONS: usize = 3;

/// The fewest different states that are split into parts: one whole cycle,
/// so the judging and choosing parts each hold three states and training
/// keeps the other fourteen. Fewer states all train, unjudged.
pub const HOLD_BACK_STATE_COUNT: usize = STATE_POSITIONS_PER_CYCLE;

/// The part of a training a state's questions belong to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SplitPart {
    Training,
    Choosing,
    Judging,
}

/// Whether this many different states are enough to hold some back.
#[must_use]
pub const fn holds_states_back(distinct_state_count: usize) -> bool {
    distinct_state_count >= HOLD_BACK_STATE_COUNT
}

/// Each distinct state's part, by its position among all states ordered by
/// their stable hash: of every 20 positions, the first three are judged and
/// the next three choose. With fewer than [`HOLD_BACK_STATE_COUNT`] states,
/// every state trains.
#[must_use]
pub fn split_parts_by_state(state_keys: &[u64]) -> HashMap<u64, SplitPart> {
    let mut distinct_state_keys = state_keys.to_vec();
    distinct_state_keys.sort_unstable();
    distinct_state_keys.dedup();
    let holds_back = holds_states_back(distinct_state_keys.len());
    let mut parts = HashMap::with_capacity(distinct_state_keys.len());
    for (state_position, state_key) in distinct_state_keys.iter().enumerate() {
        let cycle_position = state_position % STATE_POSITIONS_PER_CYCLE;
        let part = if !holds_back {
            SplitPart::Training
        } else if cycle_position < JUDGING_POSITIONS {
            SplitPart::Judging
        } else if cycle_position < JUDGING_POSITIONS + CHOOSING_POSITIONS {
            SplitPart::Choosing
        } else {
            SplitPart::Training
        };
        parts.insert(*state_key, part);
    }
    parts
}

#[cfg(test)]
#[path = "training_split_tests.rs"]
mod tests;
