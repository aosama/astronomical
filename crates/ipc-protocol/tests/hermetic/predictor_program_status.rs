//! Predictor status percentages round to one decimal and stay quiet at zero.

use astronomical_ipc_protocol::PredictorProgramStatus;

#[test]
fn should_round_top_k_accuracy_to_one_decimal_as_tenths() {
    let predictor_program_status =
        PredictorProgramStatus::from_cpu_counts(false, 7_663, 31_040, 0, 0);
    assert_eq!(predictor_program_status.top_k_accuracy_tenths, 247);
    assert!((predictor_program_status.top_k_accuracy_percent() - 24.7).abs() < 0.000_1);
    assert_eq!(predictor_program_status.pages_avoided_tenths, 0);
}

#[test]
fn should_publish_zero_when_no_evaluations_exist() {
    let predictor_program_status = PredictorProgramStatus::from_cpu_counts(true, 0, 0, 0, 0);
    assert_eq!(predictor_program_status.top_k_accuracy_tenths, 0);
    assert!(predictor_program_status.training_active);
}
