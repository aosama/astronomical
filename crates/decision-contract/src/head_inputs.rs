//! What a decision model's trainable final layers read, returned beside an
//! answer only when a caller asks for it.
//!
//! Training the final layers ("the head") of a decision model never changes
//! the large network before them, so what that network hands the head is
//! fixed for a given example. Reading it once per example lets training run
//! on the CPU in seconds instead of re-running the model for every step.
//! Each family's head reads different things, so each has its own variant.
//!
//! Every list of per-option values is in the contract's canonical option
//! order, the same order as [`crate::QuestionOutcome::option_logits`]. Clef
//! scores choices in alphabetical order internally; its engine reorders
//! these exactly as it reorders the logits.
//!
//! Vectors travel as base64 of their little-endian 32-bit floats: exact, and
//! about half the size of decimal text, which keeps a Kev-9B question with
//! many 4,096-wide options well inside one inter-process message.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Whether a decision request also returns each question's head inputs.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadInputRequest {
    /// Answers only; what every ordinary request asks for.
    #[default]
    Omit,
    /// Answers plus what the trainable final layers read.
    Include,
}

/// One question's head inputs, per model family.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "family", rename_all = "snake_case", deny_unknown_fields)]
pub enum HeadInputs {
    /// Kev's pointer head reads the decision token's final hidden state and
    /// each option's closing token's.
    Kev {
        decision_state: FloatVector,
        option_states: Vec<FloatVector>,
    },
    /// Laya's scorer reads each option's marker state after the head's
    /// transformer layers.
    Laya { marker_states: Vec<FloatVector> },
    /// Clef's residual scorer reads each option's question field and its
    /// normalized option vector; its final mix also takes the option's
    /// lexical prior and the cosine between field and option.
    Clef {
        field_vectors: Vec<FloatVector>,
        option_vectors: Vec<FloatVector>,
        lexical_priors: Vec<f32>,
        cosines: Vec<f32>,
    },
}

impl HeadInputs {
    /// How many options the inputs describe, or `None` when the per-option
    /// lists disagree, which no engine produces.
    #[must_use]
    pub fn option_count(&self) -> Option<usize> {
        match self {
            Self::Kev { option_states, .. } => Some(option_states.len()),
            Self::Laya { marker_states } => Some(marker_states.len()),
            Self::Clef {
                field_vectors,
                option_vectors,
                lexical_priors,
                cosines,
            } => {
                let option_count = option_vectors.len();
                let lists_agree = field_vectors.len() == option_count
                    && lexical_priors.len() == option_count
                    && cosines.len() == option_count;
                lists_agree.then_some(option_count)
            }
        }
    }
}

/// A vector of 32-bit floats that serializes as base64 of its little-endian bytes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FloatVector(pub Vec<f32>);

impl Serialize for FloatVector {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut little_endian_bytes = Vec::with_capacity(self.0.len() * 4);
        for float_value in &self.0 {
            little_endian_bytes.extend_from_slice(&float_value.to_le_bytes());
        }
        serializer.serialize_str(&BASE64_STANDARD.encode(little_endian_bytes))
    }
}

impl<'de> Deserialize<'de> for FloatVector {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_str(FloatVectorVisitor)
    }
}

struct FloatVectorVisitor;

impl Visitor<'_> for FloatVectorVisitor {
    type Value = FloatVector;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("base64 of little-endian 32-bit floats")
    }

    fn visit_str<E: de::Error>(self, encoded_text: &str) -> Result<FloatVector, E> {
        let Ok(little_endian_bytes) = BASE64_STANDARD.decode(encoded_text) else {
            return Err(E::custom("a float vector is not valid base64"));
        };
        let (whole_floats, leftover_bytes) = little_endian_bytes.as_chunks::<4>();
        if !leftover_bytes.is_empty() {
            return Err(E::custom(
                "a float vector's length is not a multiple of 4 bytes",
            ));
        }
        let mut float_values = Vec::with_capacity(whole_floats.len());
        for float_bytes in whole_floats {
            float_values.push(f32::from_le_bytes(*float_bytes));
        }
        Ok(FloatVector(float_values))
    }
}

#[cfg(test)]
#[path = "head_inputs_tests.rs"]
mod tests;
