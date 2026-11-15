use astronomical_memory_policy::ledger::{LedgerPhase, MemoryGovernor, ModelMemoryShape};

use super::{ForwardFit, ForwardShape};
use crate::{ResidentForwardAdmissionDecision, StreamingEngineRequirement};

const LIMIT_BYTES: u64 = 10_000;
/// The ledger keeps a fiftieth of the limit as its margin.
const MARGIN_BYTES: u64 = 200;
const EVERY_WEIGHT_BYTES: u64 = 6_000;
const PROMPT_CHUNK_TOKEN_COUNT: usize = 2_048;
const PROMPT_CHUNK_WORKING_MEMORY_BYTES: u64 = 1_000;
const REPLY_TOKEN_WORKING_MEMORY_BYTES: u64 = 100;
/// The most stored state a prompt chunk fits beside.
const LARGEST_STORED_STATE_BESIDE_A_CHUNK_BYTES: u64 =
    LIMIT_BYTES - EVERY_WEIGHT_BYTES - PROMPT_CHUNK_WORKING_MEMORY_BYTES - MARGIN_BYTES;

fn governor_with_both_phases_measured() -> MemoryGovernor {
    let mut memory_governor = MemoryGovernor::new(
        LIMIT_BYTES,
        ModelMemoryShape {
            core_weight_bytes: EVERY_WEIGHT_BYTES,
            streamable_weight_bytes: 0,
            reading_prompt_stream_slot_bytes: 0,
            writing_reply_stream_slot_bytes: 0,
        },
    )
    .expect("the limit is not zero");
    let _ = memory_governor.record_probe(
        ForwardShape::prompt_chunk(PROMPT_CHUNK_TOKEN_COUNT).phase(),
        PROMPT_CHUNK_WORKING_MEMORY_BYTES,
    );
    let _ =
        memory_governor.record_probe(LedgerPhase::WritingReply, REPLY_TOKEN_WORKING_MEMORY_BYTES);
    memory_governor
}

fn decision_for(
    forward_shape: ForwardShape,
    stored_state_after_forward_bytes: u64,
    temporary_workspace_bytes: u64,
) -> ResidentForwardAdmissionDecision {
    ForwardFit::ask(
        &governor_with_both_phases_measured(),
        forward_shape,
        stored_state_after_forward_bytes,
        temporary_workspace_bytes,
    )
    .decision()
}

#[test]
fn should_admit_a_prompt_chunk_that_lands_exactly_on_the_limit() {
    let forward_fit = ForwardFit::ask(
        &governor_with_both_phases_measured(),
        ForwardShape::prompt_chunk(PROMPT_CHUNK_TOKEN_COUNT),
        LARGEST_STORED_STATE_BESIDE_A_CHUNK_BYTES,
        0,
    );
    assert_eq!(
        forward_fit.decision(),
        ResidentForwardAdmissionDecision::Admit
    );
    assert_eq!(forward_fit.set_aside_at_peak_bytes(), LIMIT_BYTES);
}

#[test]
fn should_name_the_chunk_when_its_working_memory_is_one_byte_over() {
    assert_eq!(
        decision_for(
            ForwardShape::prompt_chunk(PROMPT_CHUNK_TOKEN_COUNT),
            LARGEST_STORED_STATE_BESIDE_A_CHUNK_BYTES + 1,
            0,
        ),
        ResidentForwardAdmissionDecision::RequireStreamingEngine {
            cause: StreamingEngineRequirement::TransientPeakShortfallAtPrefillChunk {
                shortfall_bytes: 1,
                prefill_chunk_token_count: PROMPT_CHUNK_TOKEN_COUNT,
            },
        }
    );
}

#[test]
fn should_name_the_stored_state_when_it_does_not_fit_beside_the_weights() {
    let largest_stored_state_bytes = LIMIT_BYTES - EVERY_WEIGHT_BYTES - MARGIN_BYTES;
    assert_eq!(
        decision_for(
            ForwardShape::prompt_chunk(PROMPT_CHUNK_TOKEN_COUNT),
            largest_stored_state_bytes + 1,
            0,
        ),
        ResidentForwardAdmissionDecision::RequireStreamingEngine {
            cause: StreamingEngineRequirement::StableShortfall { shortfall_bytes: 1 },
        }
    );
}

#[test]
fn should_count_temporary_workspace_against_the_peak_and_not_against_what_stays() {
    let stored_state_bytes = LARGEST_STORED_STATE_BESIDE_A_CHUNK_BYTES - 800;
    assert_eq!(
        decision_for(
            ForwardShape::prompt_chunk(PROMPT_CHUNK_TOKEN_COUNT),
            stored_state_bytes,
            800,
        ),
        ResidentForwardAdmissionDecision::Admit
    );
    assert_eq!(
        decision_for(
            ForwardShape::prompt_chunk(PROMPT_CHUNK_TOKEN_COUNT),
            stored_state_bytes,
            900,
        ),
        ResidentForwardAdmissionDecision::RequireStreamingEngine {
            cause: StreamingEngineRequirement::TransientPeakShortfallAtPrefillChunk {
                shortfall_bytes: 100,
                prefill_chunk_token_count: PROMPT_CHUNK_TOKEN_COUNT,
            },
        }
    );
}

#[test]
fn should_name_no_chunk_when_a_reply_step_is_over() {
    let largest_stored_state_beside_a_reply_step_bytes =
        LIMIT_BYTES - EVERY_WEIGHT_BYTES - REPLY_TOKEN_WORKING_MEMORY_BYTES - MARGIN_BYTES;
    assert_eq!(
        decision_for(
            ForwardShape::reply_token(),
            largest_stored_state_beside_a_reply_step_bytes,
            0,
        ),
        ResidentForwardAdmissionDecision::Admit
    );
    assert_eq!(
        decision_for(
            ForwardShape::reply_token(),
            largest_stored_state_beside_a_reply_step_bytes + 1,
            0,
        ),
        ResidentForwardAdmissionDecision::RequireStreamingEngine {
            cause: StreamingEngineRequirement::TransientPeakShortfallWithoutPrefillChunk {
                shortfall_bytes: 1,
            },
        }
    );
}

#[test]
fn should_admit_a_chunk_larger_than_any_measured_when_what_stays_fits() {
    // Nothing has measured a chunk this large, so there is no working-memory
    // figure to hold against it. It runs, and is measured when it has.
    let largest_stored_state_bytes = LIMIT_BYTES - EVERY_WEIGHT_BYTES - MARGIN_BYTES;
    assert_eq!(
        decision_for(
            ForwardShape::prompt_chunk(2 * PROMPT_CHUNK_TOKEN_COUNT),
            largest_stored_state_bytes,
            0,
        ),
        ResidentForwardAdmissionDecision::Admit
    );
}

#[test]
fn should_say_in_the_reason_how_far_over_the_limit_the_chunk_is() {
    let forward_fit = ForwardFit::ask(
        &governor_with_both_phases_measured(),
        ForwardShape::prompt_chunk(PROMPT_CHUNK_TOKEN_COUNT),
        LARGEST_STORED_STATE_BESIDE_A_CHUNK_BYTES + 7,
        0,
    );
    let rejection = forward_fit
        .decision()
        .rejection()
        .expect("seven bytes over is a rejection");
    let reason = forward_fit.rejection_reason(rejection);
    assert!(reason.contains("2048 tokens"), "{reason}");
    assert!(reason.contains("7 bytes more"), "{reason}");
    assert!(reason.contains("10000 byte limit"), "{reason}");
}

#[test]
fn should_charge_a_one_token_prompt_chunk_what_a_reply_step_needs() {
    // Beside this much stored state a whole chunk's working memory does not
    // fit, but one token's does.
    let stored_state_bytes = LARGEST_STORED_STATE_BESIDE_A_CHUNK_BYTES + 1;
    assert_eq!(
        decision_for(ForwardShape::prompt_chunk(1), stored_state_bytes, 0),
        ResidentForwardAdmissionDecision::Admit
    );
    assert_ne!(
        decision_for(ForwardShape::prompt_chunk(2), stored_state_bytes, 0),
        ResidentForwardAdmissionDecision::Admit
    );
}

#[test]
fn should_still_name_the_prompt_chunk_when_its_one_token_does_not_fit() {
    let largest_stored_state_beside_a_reply_step_bytes =
        LIMIT_BYTES - EVERY_WEIGHT_BYTES - REPLY_TOKEN_WORKING_MEMORY_BYTES - MARGIN_BYTES;
    assert_eq!(
        decision_for(
            ForwardShape::prompt_chunk(1),
            largest_stored_state_beside_a_reply_step_bytes + 1,
            0,
        ),
        ResidentForwardAdmissionDecision::RequireStreamingEngine {
            cause: StreamingEngineRequirement::TransientPeakShortfallAtPrefillChunk {
                shortfall_bytes: 1,
                prefill_chunk_token_count: 1,
            },
        }
    );
}
