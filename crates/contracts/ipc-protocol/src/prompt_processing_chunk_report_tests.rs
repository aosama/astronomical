use super::{PromptProcessingChunkReason, PromptProcessingChunkReport};

#[test]
fn should_carry_the_chunk_and_its_reason_in_snake_case() {
    let report = PromptProcessingChunkReport {
        chunk_tokens: 4_096,
        reason: PromptProcessingChunkReason::ExpertModelOnM5OrNewer,
    };

    let report_json = serde_json::to_value(report).expect("serializes");

    assert_eq!(
        report_json,
        serde_json::json!({"chunk_tokens": 4096, "reason": "expert_model_on_m5_or_newer"})
    );
    assert_eq!(
        serde_json::from_value::<PromptProcessingChunkReport>(report_json).expect("parses"),
        report
    );
}

#[test]
fn should_refuse_a_reason_it_does_not_know() {
    let unknown_reason =
        serde_json::json!({"chunk_tokens": 2048, "reason": "because_it_felt_right"});

    assert!(serde_json::from_value::<PromptProcessingChunkReport>(unknown_reason).is_err());
}

#[test]
fn should_refuse_a_report_with_an_unknown_field() {
    let extra_field =
        serde_json::json!({"chunk_tokens": 2048, "reason": "configured", "chip": "M5"});

    assert!(serde_json::from_value::<PromptProcessingChunkReport>(extra_field).is_err());
}
