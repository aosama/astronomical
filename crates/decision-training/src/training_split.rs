//! Which part of a training each labelled question is in, decided by state.
//!
//! - **training** (about 70%): the only questions whose labels move weights
//! - **choosing** (about 15%): picks when to stop and which seed wins, and
//!   fits the trained model's temperatures
//! - **judging** (about 15%): read once, at the end, by the never-worse gate
//!
//! Judging questions never influence a choice; if they did, the gate would
//! partly measure how well the choices fitted them. As in calibration, the
//! split is by state, so two questions about one state never land on two
//! sides (Laya's training notes count such rows as overlap), and it is
//! deterministic: the same examples always split the same way.

use std::collections::HashMap;

use astronomical_decision_contract::decision_training_job::TrainingScope;

/// Out of every this many distinct states (ordered by their stable hash),
/// the first three are judged and the next three choose.
const STATE_POSITIONS_PER_CYCLE: usize = 20;
const JUDGING_POSITIONS: usize = 3;
const CHOOSING_POSITIONS: usize = 3;

/// Training refuses fewer labelled questions than this: below it, the
/// judging part holds too few questions for the gate to tell an
/// improvement from luck. Questions, not file lines, because one line may
/// ask several.
pub const MINIMUM_TRAINING_QUESTIONS: usize = 200;
/// ... or fewer distinct states, which would leave each of the three parts
/// only a handful of states to stand for everything.
pub const MINIMUM_TRAINING_STATES: usize = 30;
/// A whole-model training refuses fewer questions than this. Its add-ons
/// let every layer change, so a few hundred questions are memorised rather
/// than learned, and the judging part needs about 150 questions to tell
/// the smaller gains of a sound training from luck.
pub const MINIMUM_WHOLE_MODEL_TRAINING_QUESTIONS: usize = 1_000;
/// ... or fewer distinct states, so no handful of texts dominates what
/// every layer learns.
pub const MINIMUM_WHOLE_MODEL_TRAINING_STATES: usize = 100;

/// The part of a training a state's questions belong to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SplitPart {
    Training,
    Choosing,
    Judging,
}

/// Each distinct state's part, by its position among all states ordered by
/// their stable hash: of every 20 positions, the first three are judged and
/// the next three choose.
#[must_use]
pub fn split_parts_by_state(state_keys: &[u64]) -> HashMap<u64, SplitPart> {
    let mut distinct_state_keys = state_keys.to_vec();
    distinct_state_keys.sort_unstable();
    distinct_state_keys.dedup();
    let mut parts = HashMap::with_capacity(distinct_state_keys.len());
    for (state_position, state_key) in distinct_state_keys.iter().enumerate() {
        let cycle_position = state_position % STATE_POSITIONS_PER_CYCLE;
        let part = if cycle_position < JUDGING_POSITIONS {
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

/// Why there is too little data to train on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TooLittleData {
    TooFewQuestions {
        have: usize,
        need: usize,
        scope: TrainingScope,
    },
    TooFewStates {
        have: usize,
        need: usize,
        scope: TrainingScope,
    },
}

/// Refuses to train on fewer questions or states than the gate can judge
/// for a training of `scope`.
pub fn check_training_data(
    labelled_question_count: usize,
    distinct_state_count: usize,
    scope: TrainingScope,
) -> Result<(), TooLittleData> {
    let (minimum_questions, minimum_states) = match scope {
        TrainingScope::FinalLayers => (MINIMUM_TRAINING_QUESTIONS, MINIMUM_TRAINING_STATES),
        TrainingScope::WholeModel => (
            MINIMUM_WHOLE_MODEL_TRAINING_QUESTIONS,
            MINIMUM_WHOLE_MODEL_TRAINING_STATES,
        ),
    };
    if labelled_question_count < minimum_questions {
        return Err(TooLittleData::TooFewQuestions {
            have: labelled_question_count,
            need: minimum_questions,
            scope,
        });
    }
    if distinct_state_count < minimum_states {
        return Err(TooLittleData::TooFewStates {
            have: distinct_state_count,
            need: minimum_states,
            scope,
        });
    }
    Ok(())
}

#[cfg(test)]
#[path = "training_split_tests.rs"]
mod tests;
