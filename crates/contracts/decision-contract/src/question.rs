//! The typed questions a decision request asks, independent of any model family.
//!
//! A question names the answers it allows up front, so a model can return a
//! probability for each one instead of text. The three kinds are the ones the
//! System One request shape defines: `choice`, `score` and `noul`.

use crate::ordered_json::OrderedJsonValue;

/// The shape of a question's allowed answers.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DecisionQuestionKind {
    /// One of several named options.
    Choice,
    /// A position on an ordered scale of described levels.
    Score,
    /// Whether a statement holds: the probability that it is true.
    Noul,
}

impl DecisionQuestionKind {
    /// The name this kind has in a System One request and response.
    #[must_use]
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        }
    }

    /// The kind a System One `type` field names, or `None` for an unknown name.
    #[must_use]
    pub fn from_wire_name(wire_name: &str) -> Option<Self> {
        match wire_name {
            "choice" => Some(Self::Choice),
            "score" => Some(Self::Score),
            "noul" => Some(Self::Noul),
            _ => None,
        }
    }
}

/// One validated question of a decision request.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionQuestion {
    /// The caller's identifier; answers are returned under this name.
    pub name: String,
    /// What to decide. A string is read as written; structured instructions
    /// are rendered as JSON text by each model family's renderer. `None` when
    /// the request gave none, which some families allow and others reject.
    pub instructions: Option<OrderedJsonValue>,
    pub body: DecisionQuestionBody,
}

/// The allowed answers of one question.
#[derive(Clone, Debug, PartialEq)]
pub enum DecisionQuestionBody {
    Choice {
        /// In the caller's order, which is the order answers are reported in.
        options: Vec<ChoiceOption>,
    },
    Score {
        /// Level descriptions, lowest first; a level's index is its score.
        levels: Vec<OrderedJsonValue>,
    },
    Noul {
        /// When the statement does not hold; `None` uses the family's default wording.
        false_description: Option<OrderedJsonValue>,
        /// When the statement holds; `None` uses the family's default wording.
        true_description: Option<OrderedJsonValue>,
        /// Replacement option names for families that render `false:` / `true:`.
        labels: Option<NoulLabels>,
    },
}

/// One option of a `choice` question.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceOption {
    /// The answer key returned when this option is chosen.
    pub key: String,
    /// When this option applies; `None` when the key alone describes it.
    pub description: Option<OrderedJsonValue>,
}

/// The two option names a `noul` question is rendered with.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NoulLabels {
    pub false_label: String,
    pub true_label: String,
}

impl DecisionQuestion {
    #[must_use]
    pub const fn kind(&self) -> DecisionQuestionKind {
        match self.body {
            DecisionQuestionBody::Choice { .. } => DecisionQuestionKind::Choice,
            DecisionQuestionBody::Score { .. } => DecisionQuestionKind::Score,
            DecisionQuestionBody::Noul { .. } => DecisionQuestionKind::Noul,
        }
    }

    /// How many answers the question allows; a `noul` always allows two.
    #[must_use]
    pub fn option_count(&self) -> usize {
        match &self.body {
            DecisionQuestionBody::Choice { options } => options.len(),
            DecisionQuestionBody::Score { levels } => levels.len(),
            DecisionQuestionBody::Noul { .. } => NOUL_OPTION_COUNT,
        }
    }
}

/// A `noul` question's two answers: the statement is false, or it is true.
pub const NOUL_OPTION_COUNT: usize = 2;
/// The position of "false" in the canonical `noul` probability order.
pub const NOUL_FALSE_INDEX: usize = 0;
/// The position of "true" in the canonical `noul` probability order.
pub const NOUL_TRUE_INDEX: usize = 1;

#[cfg(test)]
#[path = "question_tests.rs"]
mod tests;
