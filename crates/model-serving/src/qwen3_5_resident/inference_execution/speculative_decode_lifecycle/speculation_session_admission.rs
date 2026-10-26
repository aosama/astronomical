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

/// The request sampling strategies token-equality verification supports,
/// carrying the parameters the verifier needs to reproduce the classic
/// per-token choice.
pub use super::verification_decisions::VerificationTokenChoice;

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

/// The pairing contract between one draft configuration and one target
/// model geometry. All four rules must hold or the draft cannot run beside
/// this target: the taps must land inside the target's layers, the trunk
/// widths must match the conditioning projection, and the shared vocabulary
/// must match exactly (the draft ships no embedding or head of its own).
#[must_use]
pub fn draft_pairs_with_target(
    draft_target_layer_count: i32,
    draft_target_layer_ids: &[i32],
    draft_vocabulary_size: i32,
    draft_conditioning_input_width: i32,
    target_layer_count: usize,
    target_hidden_size: usize,
    target_vocabulary_size: u32,
) -> bool {
    let target_layer_count_i32 = i32::try_from(target_layer_count).unwrap_or(i32::MAX);
    if draft_target_layer_count != target_layer_count_i32 {
        return false;
    }
    let mut previous_tap_layer_id = -1_i32;
    for &tap_layer_id in draft_target_layer_ids {
        if tap_layer_id < 0 || tap_layer_id >= target_layer_count_i32 {
            return false;
        }
        if tap_layer_id <= previous_tap_layer_id {
            return false;
        }
        previous_tap_layer_id = tap_layer_id;
    }
    if draft_vocabulary_size != i32::try_from(target_vocabulary_size).unwrap_or(i32::MIN) {
        return false;
    }
    let expected_conditioning_width =
        i32::try_from(target_hidden_size).unwrap_or(i32::MAX) * draft_target_layer_ids.len() as i32;
    draft_conditioning_input_width == expected_conditioning_width
}
