use astronomical_ipc_protocol::NeuralEnginePrefillStatus;

use super::render_neural_engine_prefill;

fn preparing(
    completed_program_count: u32,
    estimated_remaining_ms: Option<u64>,
) -> NeuralEnginePrefillStatus {
    NeuralEnginePrefillStatus::Preparing {
        completed_program_count,
        total_program_count: 56,
        elapsed_ms: 1_000,
        estimated_remaining_ms,
        is_waiting_to_measure: false,
    }
}

#[test]
fn should_say_nothing_when_the_setting_is_off() {
    assert_eq!(
        render_neural_engine_prefill(&NeuralEnginePrefillStatus::Off),
        None
    );
}

#[test]
fn should_show_no_estimate_before_the_first_program_finishes() {
    assert_eq!(
        render_neural_engine_prefill(&preparing(0, None)).as_deref(),
        Some("preparing, 0/56 programs; prefill on the GPU until ready")
    );
}

#[test]
fn should_round_a_short_estimate_up_to_whole_seconds() {
    assert_eq!(
        render_neural_engine_prefill(&preparing(55, Some(100))).as_deref(),
        Some("preparing, 55/56 programs; prefill on the GPU until ready (about 1 s left)")
    );
}

#[test]
fn should_show_a_long_estimate_in_minutes_and_seconds() {
    assert_eq!(
        render_neural_engine_prefill(&preparing(8, Some(160_000))).as_deref(),
        Some("preparing, 8/56 programs; prefill on the GPU until ready (about 2 min 40 s left)")
    );
}

#[test]
fn should_not_claim_time_left_when_the_estimate_reaches_zero() {
    assert_eq!(
        render_neural_engine_prefill(&preparing(56, Some(0))).as_deref(),
        Some("preparing, 56/56 programs; prefill on the GPU until ready")
    );
}

#[test]
fn should_say_when_the_programs_are_engaged() {
    assert_eq!(
        render_neural_engine_prefill(&NeuralEnginePrefillStatus::Active {
            layer_program_count: 32,
            neural_engine_tokens_per_chunk: 960,
            chunk_token_count: 2048,
            measured_memory_bytes: 1,
            measured_speed_percent: None,
        })
        .as_deref(),
        Some("engaged, 32 layer programs; comparing with the GPU alone during long prompts")
    );
}

#[test]
fn should_give_the_reason_when_the_programs_are_unavailable() {
    assert_eq!(
        render_neural_engine_prefill(&NeuralEnginePrefillStatus::Unavailable {
            reason: "the chunk is too small".to_owned(),
        })
        .as_deref(),
        Some("unavailable, prefill stays on the GPU: the chunk is too small")
    );
}

#[test]
fn should_say_when_built_programs_are_waiting_to_be_measured() {
    let waiting = NeuralEnginePrefillStatus::Preparing {
        completed_program_count: 56,
        total_program_count: 56,
        elapsed_ms: 1_000,
        estimated_remaining_ms: None,
        is_waiting_to_measure: true,
    };
    assert_eq!(
        render_neural_engine_prefill(&waiting).as_deref(),
        Some(
            "56 programs built, measuring their memory at the next pause in requests; prefill on the GPU until then"
        )
    );
}

#[test]
fn should_show_the_measured_speed_once_it_is_known() {
    let engaged = NeuralEnginePrefillStatus::Active {
        layer_program_count: 32,
        neural_engine_tokens_per_chunk: 960,
        chunk_token_count: 2_048,
        measured_memory_bytes: 1,
        measured_speed_percent: Some(118),
    };
    assert_eq!(
        render_neural_engine_prefill(&engaged).as_deref(),
        Some("engaged, 32 layer programs, measured 1.18x the GPU alone")
    );
}
