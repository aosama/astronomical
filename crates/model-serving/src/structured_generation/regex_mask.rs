//! DFA-backed token masking for `structured_outputs.regex`.
//!
//! Viability is evaluated with a start-and-end anchored DFA: a token is
//! allowed exactly when the DFA state after consuming its bytes is not dead,
//! which means some continuation can still reach a full match. Allowed token
//! sets are cached per DFA state because decode revisits far fewer states
//! than it emits tokens.

use std::collections::HashMap;
use std::sync::Arc;

use regex_automata::dfa::{Automaton, dense};
use regex_automata::util::primitives::StateID;
use regex_automata::{Input, MatchKind};

use crate::structured_generation::normalize_token_piece;

const MAXIMUM_CACHED_STATES: usize = 64;

/// One compiled regex constraint advanced one visible token at a time.
#[derive(Clone, Debug)]
pub(super) struct RegexTokenMask {
    automaton: dense::DFA<Vec<u32>>,
    current_state: StateID,
    vocabulary_pieces: Arc<Vec<String>>,
    allowed_token_ids_by_state: HashMap<u32, Arc<Vec<u32>>>,
}

impl RegexTokenMask {
    /// Compiles the caller regex into the same anchored shape the public
    /// boundary validates, so REST and worker can never disagree on syntax.
    pub(super) fn compile(
        regex_pattern: &str,
        vocabulary_pieces: Arc<Vec<String>>,
    ) -> Result<Self, String> {
        let anchored_pattern =
            astronomical_ipc_protocol::structured_regex_dfa_pattern(regex_pattern);
        // MatchKind::All explores every viable continuation, so an alternative
        // that is a strict prefix of another keeps the longer path reachable
        // instead of losing it to leftmost match priority.
        let automaton = dense::Builder::new()
            .configure(dense::Config::new().match_kind(MatchKind::All))
            .build(&anchored_pattern)
            .map_err(|build_error| format!("regex pattern failed to compile: {build_error}"))?;
        let current_state = automaton
            .start_state_forward(&Input::new(&[]))
            .map_err(|start_error| format!("regex start state unavailable: {start_error}"))?;
        Ok(Self {
            automaton,
            current_state,
            vocabulary_pieces,
            allowed_token_ids_by_state: HashMap::new(),
        })
    }

    /// Returns the allowed token identifiers for the current DFA state,
    /// computing and caching the set once per distinct state.
    pub(super) fn allowed_token_ids(&mut self) -> Arc<Vec<u32>> {
        let state_key = self.current_state.as_u32();
        if let Some(allowed_token_ids) = self.allowed_token_ids_by_state.get(&state_key) {
            return Arc::clone(allowed_token_ids);
        }
        let allowed_token_ids = self.compute_allowed_token_ids();
        // A pathological pattern could visit unbounded states; the bounded
        // cache recomputes the rare extra state instead of growing without limit.
        if self.allowed_token_ids_by_state.len() < MAXIMUM_CACHED_STATES {
            self.allowed_token_ids_by_state
                .insert(state_key, Arc::clone(&allowed_token_ids));
        }
        allowed_token_ids
    }

    fn compute_allowed_token_ids(&self) -> Arc<Vec<u32>> {
        let mut allowed_token_ids = Vec::new();
        for (token_id, token_piece) in self.vocabulary_pieces.iter().enumerate() {
            let normalized_piece = normalize_token_piece(token_piece);
            if normalized_piece.is_empty() || !self.piece_is_viable(&normalized_piece) {
                continue;
            }
            allowed_token_ids.push(u32::try_from(token_id).unwrap_or(u32::MAX));
        }
        Arc::new(allowed_token_ids)
    }

    fn piece_is_viable(&self, normalized_piece: &str) -> bool {
        let mut state = self.current_state;
        for piece_byte in normalized_piece.as_bytes() {
            state = self.automaton.next_state(state, *piece_byte);
            if self.automaton.is_dead_state(state) {
                return false;
            }
        }
        true
    }

    /// Steps the automaton forward through one accepted token piece.
    ///
    /// Only mask-allowed tokens reach this method, so the resulting state can
    /// never be dead; a defensive dead state still fails open to end of text.
    pub(super) fn accept_token_piece(&mut self, normalized_piece: &str) {
        let mut state = self.current_state;
        for piece_byte in normalized_piece.as_bytes() {
            state = self.automaton.next_state(state, *piece_byte);
        }
        self.current_state = state;
    }

    /// Whether the visible answer can end here with a complete regex match.
    pub(super) fn is_complete(&self) -> bool {
        let end_of_input_state = self.automaton.next_eoi_state(self.current_state);
        self.automaton.is_match_state(end_of_input_state)
    }
}
