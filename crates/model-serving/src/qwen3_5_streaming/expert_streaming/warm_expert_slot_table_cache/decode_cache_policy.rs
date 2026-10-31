use std::cmp::Ordering;

/// Concurrent probationary slots a layer may hold: one decode token's full
/// routed set (this family routes eight experts per token). Admitting the
/// whole set lets a table cover a token's demand the step it streams, while
/// the fully-decayed hysteresis and the per-step swap cap below keep
/// eviction-side churn bounded.
pub(super) const MAXIMUM_PROBATIONARY_EXPERT_SLOTS_PER_LAYER: usize = 8;
/// Candidate-policy slot replacements allowed per layer per decode step.
/// Beyond this cap the streamed page stays operation-local: measured churn
/// (492 evictions in 200 tokens, 4.4x decode source bytes) showed that
/// unbounded eviction-for-admission re-streams far more than it saves.
pub(super) const MAXIMUM_CANDIDATE_SLOT_REPLACEMENTS_PER_DECODE_STEP: usize = 2;

const PROBATIONARY_DECODE_STEP_WINDOW: u64 = 4;
const RECENT_REUSE_SCORE_HALF_LIFE_DECODE_STEPS: u64 = 8;
const PROBATIONARY_PROMOTION_REUSE_SCORE: u64 = 2;
const REUSE_SCORE_BUCKET_COUNT: u32 = u64::BITS;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct DecodeExpertSlotState {
    recent_reuse_score: u64,
    last_decode_access_epoch: u64,
    probationary_until_epoch: Option<u64>,
}

impl DecodeExpertSlotState {
    pub(super) fn mark_probationary(&mut self, decode_step_epoch: u64) {
        self.recent_reuse_score = 0;
        self.last_decode_access_epoch = decode_step_epoch;
        self.probationary_until_epoch =
            Some(decode_step_epoch.saturating_add(PROBATIONARY_DECODE_STEP_WINDOW));
    }

    /// A first admission backed by persisted usage history: the expert
    /// carries the promotion score immediately, so it is not evictable and
    /// cannot be churned away before its first in-session reuse.
    pub(super) fn mark_historically_hot(&mut self, decode_step_epoch: u64) {
        self.recent_reuse_score = PROBATIONARY_PROMOTION_REUSE_SCORE;
        self.last_decode_access_epoch = decode_step_epoch;
        self.probationary_until_epoch = None;
    }

    pub(super) fn expire_probationary_admission(&mut self, decode_step_epoch: u64) {
        if self
            .probationary_until_epoch
            .is_some_and(|probationary_until_epoch| decode_step_epoch > probationary_until_epoch)
        {
            self.probationary_until_epoch = None;
        }
    }

    pub(super) fn record_decode_reuse(&mut self, decode_step_epoch: u64) {
        let previous_reuse_score = self.recent_reuse_score_at(decode_step_epoch);
        let was_probationary = self.is_probationary(decode_step_epoch);
        self.probationary_until_epoch = None;
        self.last_decode_access_epoch = decode_step_epoch;
        self.recent_reuse_score = previous_reuse_score.saturating_add(if was_probationary {
            PROBATIONARY_PROMOTION_REUSE_SCORE
        } else {
            1
        });
    }

    pub(super) fn is_probationary(self, decode_step_epoch: u64) -> bool {
        self.probationary_until_epoch
            .is_some_and(|probationary_until_epoch| decode_step_epoch <= probationary_until_epoch)
    }

    /// A slot is a victim candidate only when its reuse evidence has fully
    /// decayed. A first-time miss arrives with no score of its own, so a
    /// relative margin (colibri's 25%+4 hysteresis) degenerates to an absolute
    /// one: any positive decayed score disqualifies the slot. Live-but-cold
    /// entries survive probationary churn; entries idle for one full decay
    /// cascade (eight half-lives) become eligible again.
    pub(super) fn is_evictable(self, decode_step_epoch: u64) -> bool {
        !self.is_probationary(decode_step_epoch)
            && self.recent_reuse_score_at(decode_step_epoch) == 0
    }

    pub(super) fn compare_eviction_priority(
        self,
        self_slot: usize,
        other: Self,
        other_slot: usize,
        decode_step_epoch: u64,
        self_payload_bytes_per_expert: u64,
        other_payload_bytes_per_expert: u64,
    ) -> Ordering {
        let self_payload_bytes_per_expert = self_payload_bytes_per_expert.max(1);
        let other_payload_bytes_per_expert = other_payload_bytes_per_expert.max(1);
        let self_recent_reuse_score = self.recent_reuse_score_at(decode_step_epoch);
        let other_recent_reuse_score = other.recent_reuse_score_at(decode_step_epoch);
        // Cross multiplication compares reuse-per-byte without rounding.
        let self_value_cross_product =
            u128::from(self_recent_reuse_score) * u128::from(other_payload_bytes_per_expert);
        let other_value_cross_product =
            u128::from(other_recent_reuse_score) * u128::from(self_payload_bytes_per_expert);

        self_value_cross_product
            .cmp(&other_value_cross_product)
            .then_with(|| {
                self.last_decode_access_epoch
                    .cmp(&other.last_decode_access_epoch)
            })
            .then_with(|| self_slot.cmp(&other_slot))
    }

    fn recent_reuse_score_at(self, decode_step_epoch: u64) -> u64 {
        let elapsed_half_lives = decode_step_epoch.saturating_sub(self.last_decode_access_epoch)
            / RECENT_REUSE_SCORE_HALF_LIFE_DECODE_STEPS;
        let shift = u32::try_from(elapsed_half_lives)
            .unwrap_or(REUSE_SCORE_BUCKET_COUNT)
            .min(REUSE_SCORE_BUCKET_COUNT);
        if shift == REUSE_SCORE_BUCKET_COUNT {
            0
        } else {
            self.recent_reuse_score >> shift
        }
    }
}

pub(super) fn available_probationary_slot_count(
    decode_policy_states: &[DecodeExpertSlotState],
    decode_step_epoch: u64,
) -> usize {
    let active_probationary_slot_count = decode_policy_states
        .iter()
        .filter(|decode_policy_state| decode_policy_state.is_probationary(decode_step_epoch))
        .count();
    MAXIMUM_PROBATIONARY_EXPERT_SLOTS_PER_LAYER.saturating_sub(active_probationary_slot_count)
}

#[cfg(test)]
#[path = "decode_cache_policy_tests.rs"]
mod decode_cache_policy_tests;
