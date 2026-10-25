//! Pure admission decisions for speculative decoding sessions.
//!
//! Speculation buys throughput only when its assumptions hold. These rules
//! decide, from request properties alone, whether one generation session may
//! run draft-and-verify: the engine must host a draft, and the request must
//! not combine verification with semantics the greedy block contract cannot
//! preserve. Everything here is values in, values out, so every rule is unit
//! testable without a graphics processor.

/// The sampling strategies the greedy verification contract supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculationCompatibleSampling {
    /// Argmax decoding: the acceptance rule compares the target's argmax
    /// against each proposal, which is exactly the greedy contract.
    HighestLogit,
}

/// Properties of one generation request that speculation admission reads.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculationSessionRequestProperties {
    pub sampling_strategy: Option<SpeculationCompatibleSampling>,
    /// Structured generation applies per-token constraints; the drafted
    /// block would bypass them, so such sessions fall back to classic decode.
    pub structured_generation_is_active: bool,
    /// A forced thinking transition replaces the conditioned successor with
    /// a chosen token; the classic path handles those boundaries exactly.
    pub forced_thinking_transition_is_pending: bool,
    /// Visual embedding injection changes the input path mid-request.
    pub visual_embeddings_are_active: bool,
}

/// Whether one generation session may run speculative decoding.
#[must_use]
pub fn should_admit_speculation_for_session(
    request_properties: &SpeculationSessionRequestProperties,
) -> bool {
    request_properties.sampling_strategy == Some(SpeculationCompatibleSampling::HighestLogit)
        && !request_properties.structured_generation_is_active
        && !request_properties.forced_thinking_transition_is_pending
        && !request_properties.visual_embeddings_are_active
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
