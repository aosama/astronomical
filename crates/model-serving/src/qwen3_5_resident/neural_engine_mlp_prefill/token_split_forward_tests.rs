use astronomical_neural_engine::core_ml::CoreMlProgramError;
use astronomical_neural_engine::mlx_hand_off::MlxHandOffError;

use super::TokenSplitForwardError;

#[test]
fn should_blame_the_programs_for_a_program_failure() {
    let program_error = TokenSplitForwardError::Program(CoreMlProgramError::PredictionStart);
    assert!(program_error.is_about_the_programs());
    let missing_program = TokenSplitForwardError::MissingLayerProgram { layer_index: 3 };
    assert!(missing_program.is_about_the_programs());
}

fn mlx_error() -> mlx_c_rust::MlxCError {
    mlx_c_rust::MlxCError {
        operation: "evaluate",
        description: "out of memory".to_owned(),
    }
}

#[test]
fn should_not_blame_the_programs_for_an_mlx_error_however_it_arrives() {
    assert!(!TokenSplitForwardError::Mlx(mlx_error()).is_about_the_programs());
    let through_the_hand_off = MlxHandOffError::Mlx(mlx_error());
    assert!(!TokenSplitForwardError::HandOff(through_the_hand_off).is_about_the_programs());
}

#[test]
fn should_blame_the_programs_for_a_hand_off_they_caused() {
    let padded_surface = TokenSplitForwardError::HandOff(MlxHandOffError::PaddedOutputSurface);
    assert!(padded_surface.is_about_the_programs());
    let wrapped_program = TokenSplitForwardError::HandOff(MlxHandOffError::Program(
        CoreMlProgramError::PredictionStart,
    ));
    assert!(wrapped_program.is_about_the_programs());
}
