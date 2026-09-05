//! Sample-time token masking for extra-body structured generation.
//!
//! OpenAI `response_format` may degrade to a prompt. Extra-body constraints
//! must mask illegal logits or the request must fail before generation.

mod json_prefix;

use astronomical_ipc_protocol::StructuredGenerationConstraint;

use json_prefix::JsonPrefixStatus;

/// Compiled constraint used while sampling visible (non-thinking) tokens.
#[derive(Clone, Debug)]
pub(crate) struct StructuredTokenConstraint {
    vocabulary_pieces: Vec<String>,
    end_of_sequence_token_ids: Vec<u32>,
    kind: ConstraintKind,
}

#[derive(Clone, Debug)]
enum ConstraintKind {
    Choice {
        remaining_token_suffixes: Vec<Vec<u32>>,
    },
    Json {
        decoded_text: String,
    },
}

impl StructuredTokenConstraint {
    pub(crate) fn compile(
        constraint: &StructuredGenerationConstraint,
        vocabulary_pieces: Vec<String>,
        end_of_sequence_token_ids: Vec<u32>,
        encoded_choice_sequences: Vec<Vec<u32>>,
    ) -> Self {
        let kind = match constraint {
            StructuredGenerationConstraint::Choice { .. } => ConstraintKind::Choice {
                remaining_token_suffixes: encoded_choice_sequences,
            },
            StructuredGenerationConstraint::JsonObject
            | StructuredGenerationConstraint::JsonSchema { .. } => ConstraintKind::Json {
                decoded_text: String::new(),
            },
        };
        Self {
            vocabulary_pieces,
            end_of_sequence_token_ids,
            kind,
        }
    }

    pub(crate) fn logit_bias_values(&self) -> Vec<f32> {
        let vocabulary_size = self.vocabulary_size();
        let mut biases = vec![f32::NEG_INFINITY; vocabulary_size];
        match &self.kind {
            ConstraintKind::Choice {
                remaining_token_suffixes,
            } => {
                for remaining_token_suffix in remaining_token_suffixes {
                    if let Some(next_token_id) = remaining_token_suffix.first() {
                        if let Some(bias) = biases.get_mut(*next_token_id as usize) {
                            *bias = 0.0;
                        }
                    }
                }
            }
            ConstraintKind::Json { .. } => {
                for (token_id, token_piece) in self.vocabulary_pieces.iter().enumerate() {
                    let normalized_piece = normalize_token_piece(token_piece);
                    if self.piece_is_allowed(&normalized_piece) {
                        biases[token_id] = 0.0;
                    }
                }
            }
        }
        for end_of_sequence_token_id in &self.end_of_sequence_token_ids {
            if self.is_complete() {
                if let Some(bias) = biases.get_mut(*end_of_sequence_token_id as usize) {
                    *bias = 0.0;
                }
            }
        }
        if !biases.iter().any(|bias| *bias == 0.0) {
            // Fail open to EOS rather than a zero-mass categorical row.
            for end_of_sequence_token_id in &self.end_of_sequence_token_ids {
                if let Some(bias) = biases.get_mut(*end_of_sequence_token_id as usize) {
                    *bias = 0.0;
                }
            }
        }
        biases
    }

    fn vocabulary_size(&self) -> usize {
        self.vocabulary_pieces.len()
    }

    pub(crate) fn accept_visible_token(&mut self, token_id: u32) {
        if self.end_of_sequence_token_ids.contains(&token_id) {
            return;
        }
        let Some(token_piece) = self.vocabulary_pieces.get(token_id as usize) else {
            return;
        };
        let normalized_piece = normalize_token_piece(token_piece);
        match &mut self.kind {
            ConstraintKind::Choice {
                remaining_token_suffixes,
            } => {
                remaining_token_suffixes.retain_mut(|remaining_token_suffix| {
                    if remaining_token_suffix.first().copied() == Some(token_id) {
                        remaining_token_suffix.remove(0);
                        true
                    } else {
                        false
                    }
                });
            }
            ConstraintKind::Json { decoded_text } => {
                decoded_text.push_str(&normalized_piece);
            }
        }
    }

    fn piece_is_allowed(&self, normalized_piece: &str) -> bool {
        match &self.kind {
            ConstraintKind::Choice { .. } => false,
            ConstraintKind::Json { decoded_text } => {
                if normalized_piece.is_empty() {
                    return true;
                }
                let mut candidate =
                    String::with_capacity(decoded_text.len() + normalized_piece.len());
                candidate.push_str(decoded_text);
                candidate.push_str(normalized_piece);
                json_prefix::status(&candidate) != JsonPrefixStatus::Invalid
            }
        }
    }

    fn is_complete(&self) -> bool {
        match &self.kind {
            ConstraintKind::Choice {
                remaining_token_suffixes,
            } => remaining_token_suffixes
                .iter()
                .any(|remaining_token_suffix| remaining_token_suffix.is_empty()),
            ConstraintKind::Json { decoded_text } => {
                json_prefix::status(decoded_text) == JsonPrefixStatus::Complete
            }
        }
    }
}

fn normalize_token_piece(token_piece: &str) -> String {
    token_piece.replace('\u{0120}', " ").replace('▁', " ")
}

#[cfg(test)]
mod tests {
    use astronomical_ipc_protocol::StructuredGenerationConstraint;

    use super::StructuredTokenConstraint;

    #[test]
    fn should_mask_choice_tokens_to_juliet_or_romeo() {
        let constraint = StructuredTokenConstraint::compile(
            &StructuredGenerationConstraint::Choice {
                choices: vec!["Juliet".to_owned(), "Romeo".to_owned()],
            },
            vec![
                "Juliet".to_owned(),
                "Romeo".to_owned(),
                "Mercutio".to_owned(),
            ],
            vec![99],
            vec![vec![0], vec![1]],
        );
        let biases = constraint.logit_bias_values();
        assert_eq!(biases[0], 0.0);
        assert_eq!(biases[1], 0.0);
        assert_eq!(biases[2], f32::NEG_INFINITY);
    }

    #[test]
    fn should_complete_choice_after_its_token_sequence() {
        let mut constraint = StructuredTokenConstraint::compile(
            &StructuredGenerationConstraint::Choice {
                choices: vec!["Juliet".to_owned()],
            },
            vec!["Jul".to_owned(), "iet".to_owned(), "Romeo".to_owned()],
            vec![99],
            vec![vec![0, 1]],
        );
        assert_eq!(constraint.logit_bias_values()[0], 0.0);
        assert_eq!(constraint.logit_bias_values()[1], f32::NEG_INFINITY);
        constraint.accept_visible_token(0);
        assert_eq!(constraint.logit_bias_values()[1], 0.0);
        constraint.accept_visible_token(1);
        assert!(constraint.is_complete());
        assert_eq!(constraint.logit_bias_values()[0], f32::NEG_INFINITY);
        assert_eq!(constraint.logit_bias_values()[1], f32::NEG_INFINITY);
    }

    #[test]
    fn should_allow_json_object_prefixes_and_reject_prose() {
        let vocabulary_pieces = vec!["{".to_owned(), "Two".to_owned(), "}".to_owned()];
        let mut constraint = StructuredTokenConstraint::compile(
            &StructuredGenerationConstraint::JsonObject,
            vocabulary_pieces,
            vec![99],
            Vec::new(),
        );
        let biases = constraint.logit_bias_values();
        assert_eq!(biases[0], 0.0);
        assert_eq!(biases[1], f32::NEG_INFINITY);
        constraint.accept_visible_token(0);
        let biases_after_open = constraint.logit_bias_values();
        assert_eq!(biases_after_open[2], 0.0);
        assert_eq!(biases_after_open[1], f32::NEG_INFINITY);
    }
}
