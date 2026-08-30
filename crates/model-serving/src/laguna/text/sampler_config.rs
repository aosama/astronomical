/// Canonical artifact or request sampling policy for Laguna generation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LagunaSamplerConfig {
    uses_sampling: bool,
    temperature_thousandths: u16,
    top_p_thousandths: u16,
    min_p_thousandths: u16,
    top_k: Option<u16>,
    repetition_penalty_thousandths: u16,
    maximum_new_tokens: Option<u32>,
    seed: Option<u64>,
}

impl LagunaSamplerConfig {
    /// Truncation used when `generation_config.json` omits `top_k`.
    ///
    /// Poolside's Laguna eval and published Transformers recipe uses `top_k = 20`.
    /// MLX affine packages often drop that field; sampling still applies this default
    /// rather than drawing from the full vocabulary.
    pub const DEFAULT_SAMPLING_TOP_K: u16 = 20;

    #[allow(clippy::too_many_arguments)]
    pub(super) const fn new(
        uses_sampling: bool,
        temperature_thousandths: u16,
        top_p_thousandths: u16,
        min_p_thousandths: u16,
        top_k: Option<u16>,
        repetition_penalty_thousandths: u16,
        maximum_new_tokens: Option<u32>,
        seed: Option<u64>,
    ) -> Self {
        Self {
            uses_sampling,
            temperature_thousandths,
            top_p_thousandths,
            min_p_thousandths,
            top_k,
            repetition_penalty_thousandths,
            maximum_new_tokens,
            seed,
        }
    }

    /// Returns whether random sampling is enabled by the effective policy.
    #[must_use]
    pub const fn uses_sampling(&self) -> bool {
        self.uses_sampling
    }

    /// Returns temperature in exact protocol thousandths.
    #[must_use]
    pub const fn temperature_thousandths(&self) -> u16 {
        self.temperature_thousandths
    }

    /// Returns nucleus probability in exact protocol thousandths.
    #[must_use]
    pub const fn top_p_thousandths(&self) -> u16 {
        self.top_p_thousandths
    }

    /// Returns minimum retained-token probability in thousandths.
    #[must_use]
    pub const fn min_p_thousandths(&self) -> u16 {
        self.min_p_thousandths
    }

    /// Returns an artifact-specific top-k policy when one was declared.
    #[must_use]
    pub const fn top_k(&self) -> Option<u16> {
        self.top_k
    }

    /// Returns the top-k that GPU sampling will execute.
    #[must_use]
    pub const fn sampling_top_k(&self) -> u16 {
        match self.top_k {
            Some(top_k) => top_k,
            None => Self::DEFAULT_SAMPLING_TOP_K,
        }
    }

    /// Returns repetition penalty in thousandths, where 1000 is neutral.
    #[must_use]
    pub const fn repetition_penalty_thousandths(&self) -> u16 {
        self.repetition_penalty_thousandths
    }

    /// Returns the artifact output-token ceiling when one was declared.
    #[must_use]
    pub const fn maximum_new_tokens(&self) -> Option<u32> {
        self.maximum_new_tokens
    }

    /// Returns the request-local deterministic seed.
    #[must_use]
    pub const fn seed(&self) -> Option<u64> {
        self.seed
    }

    pub(super) const fn with_request_overrides(
        &self,
        temperature_thousandths: Option<u16>,
        top_p_thousandths: Option<u16>,
        seed: Option<u64>,
    ) -> Self {
        let effective_temperature = match temperature_thousandths {
            Some(value) => value,
            None => self.temperature_thousandths,
        };
        let has_request_sampling_override =
            temperature_thousandths.is_some() || top_p_thousandths.is_some();
        Self {
            uses_sampling: effective_temperature > 0
                && (self.uses_sampling || has_request_sampling_override),
            temperature_thousandths: effective_temperature,
            top_p_thousandths: match top_p_thousandths {
                Some(value) => value,
                None => self.top_p_thousandths,
            },
            min_p_thousandths: self.min_p_thousandths,
            top_k: self.top_k,
            repetition_penalty_thousandths: self.repetition_penalty_thousandths,
            maximum_new_tokens: self.maximum_new_tokens,
            seed,
        }
    }
}
