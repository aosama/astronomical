use astronomical_ipc_protocol::NeuralEnginePrefillStatus;

use super::{ProgramSettlement, program_settlement_of, status_after_verdict};
use crate::qwen3_5_resident::neural_engine_chunk_comparison::{
    ChunkComparisonVerdict, not_worthwhile_reason,
};

fn active_status() -> NeuralEnginePrefillStatus {
    NeuralEnginePrefillStatus::Active {
        layer_program_count: 32,
        neural_engine_tokens_per_chunk: 960,
        chunk_token_count: 2_048,
        measured_memory_bytes: 3_000,
        measured_speed_percent: None,
    }
}

#[test]
fn should_add_the_gain_to_an_active_status_when_worthwhile() {
    let settled_status = status_after_verdict(
        ChunkComparisonVerdict::Worthwhile { gain: 1.18 },
        &active_status(),
    );
    let NeuralEnginePrefillStatus::Active {
        layer_program_count,
        measured_memory_bytes,
        measured_speed_percent,
        ..
    } = settled_status
    else {
        panic!("the programs stay active: {settled_status:?}");
    };
    assert_eq!(layer_program_count, 32);
    assert_eq!(measured_memory_bytes, 3_000);
    assert_eq!(measured_speed_percent, Some(118));
}

#[test]
fn should_report_unavailable_with_the_gain_when_slower() {
    let settled_status = status_after_verdict(
        ChunkComparisonVerdict::NotWorthwhile { gain: 0.91 },
        &active_status(),
    );
    assert_eq!(
        settled_status,
        NeuralEnginePrefillStatus::Unavailable {
            reason: not_worthwhile_reason(0.91)
        }
    );
    assert!(not_worthwhile_reason(0.91).contains("0.91 times"));
}

#[test]
fn should_leave_another_status_alone_when_worthwhile() {
    assert_eq!(
        status_after_verdict(
            ChunkComparisonVerdict::Worthwhile { gain: 1.1 },
            &NeuralEnginePrefillStatus::Off
        ),
        NeuralEnginePrefillStatus::Off
    );
}

#[test]
fn should_release_failed_programs_ahead_of_any_verdict() {
    let verdict = ChunkComparisonVerdict::Worthwhile { gain: 1.2 };
    assert_eq!(
        program_settlement_of(true, false, Some(verdict)),
        Some(ProgramSettlement::ReleaseAfterFailure)
    );
    assert_eq!(
        program_settlement_of(true, true, None),
        Some(ProgramSettlement::ReleaseAfterFailure)
    );
}

#[test]
fn should_report_a_verdict_once() {
    let verdict = ChunkComparisonVerdict::NotWorthwhile { gain: 0.9 };
    assert_eq!(
        program_settlement_of(false, false, Some(verdict)),
        Some(ProgramSettlement::ReportVerdict(verdict))
    );
    assert_eq!(program_settlement_of(false, true, Some(verdict)), None);
}

#[test]
fn should_have_nothing_to_do_before_a_verdict() {
    assert_eq!(program_settlement_of(false, false, None), None);
}
