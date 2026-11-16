//! Pure admission decisions for speculative decoding sessions.
//!
//! Speculation buys throughput only when its assumptions hold. These rules
//! decide, from request properties alone, whether one generation session may
//! run draft-and-verify: the engine must host a draft, and the request must
//! not combine verification with semantics the token-equality contract
//! cannot preserve. Greedy and sampled decoding both preserve exactly —
//! every emitted token is the target's own argmax or tempered sample —
//! while structured generation, active repetition penalties, and visual
//! embedding injection transform or bypass the logits the verifier reads,
//! so those sessions stay classic. Everything here is values in, values
//! out, so every rule is unit testable without a graphics processor.

/// How the target picks one token after every verified row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationTokenChoice {
    /// The argmax: emitted tokens are exactly the classic greedy decode.
    HighestLogit,
    /// The keyed sample from the sampler's own distribution at these
    /// settings (top-k and top-p on the untempered logits, then the
    /// temperature): an exact draw from the classic sampled law.
    TopKTopP {
        temperature_thousandths: u16,
        top_k: u16,
        top_p_thousandths: u16,
    },
}

/// Properties of one generation request that speculation admission reads.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculationSessionRequestProperties {
    /// The request's sampling strategy when token-equality verification can
    /// reproduce it: greedy argmax, or top-k top-p sampling at the request's
    /// temperature. `None` means the strategy transforms logits in ways the
    /// verification forward does not apply (structured generation and
    /// repetition penalties raise their own flags below).
    pub sampling_strategy: Option<VerificationTokenChoice>,
    /// Structured generation applies per-token constraints; the drafted
    /// block would bypass them, so such sessions fall back to classic decode.
    pub structured_generation_is_active: bool,
    /// A forced thinking transition replaces the conditioned successor with
    /// a chosen token; the classic path handles those boundaries exactly.
    pub forced_thinking_transition_is_pending: bool,
    /// An active repetition penalty transforms the logits before the argmax
    /// the classic path emits; the greedy block contract compares raw
    /// argmax only, so penalized sessions stay classic.
    pub repetition_penalty_is_active: bool,
    /// Visual embedding injection changes the input path mid-request.
    pub visual_embeddings_are_active: bool,
}

/// Whether one generation session may run speculative decoding.
#[must_use]
pub fn should_admit_speculation_for_session(
    request_properties: &SpeculationSessionRequestProperties,
) -> bool {
    request_properties.sampling_strategy.is_some()
        && !request_properties.structured_generation_is_active
        && !request_properties.forced_thinking_transition_is_pending
        && !request_properties.visual_embeddings_are_active
        && !request_properties.repetition_penalty_is_active
}

/// Maps one request sampling strategy onto the verification token choice
/// that reproduces it. Greedy maps to per-position argmax; top-k top-p
/// sampling maps to one tempered sample per position through the same
/// parameters. Every strategy the engine supports today is reproducible,
/// so this returns `None` only for future strategies.
#[must_use]
pub fn verification_token_choice_for_strategy(
    sampling_strategy: &crate::Qwen3_5SamplingStrategy,
) -> Option<VerificationTokenChoice> {
    match sampling_strategy {
        crate::Qwen3_5SamplingStrategy::HighestLogit => Some(VerificationTokenChoice::HighestLogit),
        crate::Qwen3_5SamplingStrategy::TopKTopP {
            temperature_thousandths,
            top_k,
            top_p_thousandths,
            ..
        } => Some(VerificationTokenChoice::TopKTopP {
            temperature_thousandths: *temperature_thousandths,
            top_k: *top_k,
            top_p_thousandths: *top_p_thousandths,
        }),
    }
}

/// Whether an admitted session still drafts once its prompt prefix has been
/// restored from the `astro_ssd_prompt_cache`. The drafter conditions on
/// hidden rows the target taps while prefill computes the prompt's suffix; a
/// prompt restored to its last token leaves no row to tap, so that session
/// decodes classically.
#[must_use]
pub fn drafting_survives_prompt_restore(
    speculative_session_admitted: bool,
    restored_prompt_token_count: usize,
    prompt_token_count: usize,
) -> bool {
    speculative_session_admitted && restored_prompt_token_count < prompt_token_count
}
