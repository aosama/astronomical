//! Transient workspace evidence with its source and valid forward-token span.

/// The evidence source that supplied a transient workspace reserve.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransientWorkspaceReserveSource {
    /// The exact forward shape completed previously.
    ExactContext,
    /// A larger observation was conservatively scaled down.
    PhaseScaled,
    /// The analytic floor or an unscaled smaller observation supplied the reserve.
    BeyondObservedTokenSpan,
    /// No same-phase observation was available, so the global maximum was used.
    GlobalMaximum,
    /// Request admission before any forward shape exists: no learned evidence applies.
    ///
    /// The reserve is zero because request admission promises only its explicit
    /// temporary workspace; learned per-forward reserves appear once a concrete
    /// forward exists, so attributing bytes to an observation here would be false.
    NoLearnedReserve,
}

/// A completed forward's measured transient workspace high-water.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransientWorkspaceObservation {
    high_water_bytes: usize,
    forward_token_count: usize,
}

impl TransientWorkspaceObservation {
    /// Records workspace evidence for the number of tokens in the completed forward.
    #[must_use]
    pub const fn new(high_water_bytes: usize, forward_token_count: usize) -> Self {
        Self {
            high_water_bytes,
            forward_token_count,
        }
    }

    #[must_use]
    pub const fn high_water_bytes(self) -> usize {
        self.high_water_bytes
    }

    #[must_use]
    pub const fn forward_token_count(self) -> usize {
        self.forward_token_count
    }
}

/// A reusable transient workspace reserve whose provenance cannot be discarded.
///
/// Fields stay private so callers cannot fabricate a reserve without identifying
/// its source and the token span for which it is valid.
///
/// ```compile_fail,E0451
/// use astronomical_memory_policy::budget::{
///     TransientWorkspaceReserve, TransientWorkspaceReserveSource,
/// };
///
/// let _invalid_reserve = TransientWorkspaceReserve {
///     reserve_bytes: 1,
///     source: TransientWorkspaceReserveSource::GlobalMaximum,
///     forward_token_count: 1,
/// };
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransientWorkspaceReserve {
    reserve_bytes: usize,
    source: TransientWorkspaceReserveSource,
    forward_token_count: usize,
}

impl TransientWorkspaceReserve {
    /// Uses an observation made for this exact forward shape.
    #[must_use]
    pub const fn exact_context(observation: TransientWorkspaceObservation) -> Self {
        Self {
            reserve_bytes: observation.high_water_bytes,
            source: TransientWorkspaceReserveSource::ExactContext,
            forward_token_count: observation.forward_token_count,
        }
    }

    /// Scales an observation to no more than its positive token span.
    ///
    /// Returning `None` for a larger target makes scale-up impossible through
    /// this constructor; zero-token observations cannot bound proportional work.
    #[must_use]
    pub fn scaled_down_to(
        observation: TransientWorkspaceObservation,
        target_token_count: usize,
    ) -> Option<Self> {
        let scaled_reserve_bytes = scale_transient_to_token_count(
            observation.high_water_bytes,
            observation.forward_token_count,
            target_token_count,
        )?;
        Some(Self {
            reserve_bytes: scaled_reserve_bytes,
            source: TransientWorkspaceReserveSource::PhaseScaled,
            forward_token_count: target_token_count,
        })
    }

    /// Uses the analytic workspace floor for the requested forward token span.
    #[must_use]
    pub const fn analytic_floor(floor_bytes: usize, forward_token_count: usize) -> Self {
        Self {
            reserve_bytes: floor_bytes,
            source: TransientWorkspaceReserveSource::BeyondObservedTokenSpan,
            forward_token_count,
        }
    }

    /// Composes a request-admission reserve that carries no learned evidence.
    ///
    /// Request admission runs before any forward exists, so its only transient
    /// claim is the caller's explicit temporary workspace. The zero reserve and
    /// the `NoLearnedReserve` source keep that provenance honest instead of
    /// borrowing the global-maximum label, which promises an observation that
    /// was never made.
    #[must_use]
    pub const fn without_learned_reserve() -> Self {
        Self {
            reserve_bytes: 0,
            source: TransientWorkspaceReserveSource::NoLearnedReserve,
            forward_token_count: 0,
        }
    }

    /// Uses the largest available observation when no same-phase shape covers the request.
    #[must_use]
    pub const fn global_maximum(high_water_bytes: usize, forward_token_count: usize) -> Self {
        Self {
            reserve_bytes: high_water_bytes,
            source: TransientWorkspaceReserveSource::GlobalMaximum,
            forward_token_count,
        }
    }

    /// Keeps the larger unscaled observation when it covers fewer tokens.
    ///
    /// Smaller observations are not multiplied by a token ratio: their fixed
    /// per-forward overhead would otherwise inflate reserves for larger work.
    #[must_use]
    pub const fn max_with_smaller_unscaled_observation(
        mut self,
        observation: TransientWorkspaceObservation,
    ) -> Self {
        if observation.forward_token_count > 0
            && observation.forward_token_count < self.forward_token_count
            && observation.high_water_bytes > self.reserve_bytes
        {
            self.reserve_bytes = observation.high_water_bytes;
        }
        self
    }

    #[must_use]
    pub const fn reserve_bytes(self) -> usize {
        self.reserve_bytes
    }

    #[must_use]
    pub const fn source(self) -> TransientWorkspaceReserveSource {
        self.source
    }

    /// Returns the forward token count this reserve represents.
    #[must_use]
    pub const fn forward_token_count(self) -> usize {
        self.forward_token_count
    }
}

/// Scales an observation to an equal or smaller forward using ceiling division.
#[must_use]
pub(crate) fn scale_transient_to_token_count(
    observed_high_water_bytes: usize,
    observed_forward_token_count: usize,
    target_forward_token_count: usize,
) -> Option<usize> {
    if observed_forward_token_count == 0
        || target_forward_token_count == 0
        || target_forward_token_count > observed_forward_token_count
    {
        return None;
    }
    let observed_bytes_u128 =
        u128::from(u64::try_from(observed_high_water_bytes).unwrap_or(u64::MAX));
    let target_tokens_u128 =
        u128::from(u64::try_from(target_forward_token_count).unwrap_or(u64::MAX));
    let observed_tokens_u128 =
        u128::from(u64::try_from(observed_forward_token_count).unwrap_or(u64::MAX));
    let scaled_bytes = (observed_bytes_u128 * target_tokens_u128).div_ceil(observed_tokens_u128);
    usize::try_from(scaled_bytes).ok()
}

#[cfg(test)]
#[path = "transient_workspace_reserve_tests.rs"]
mod transient_workspace_reserve_tests;
