use crate::EnginePlacement;

use super::{Qwen3_5SelectedEngine, streaming_only_generation_choice};

#[test]
fn should_stream_a_streaming_only_generation_that_would_fit_resident() {
    let choice = streaming_only_generation_choice(
        (
            Qwen3_5SelectedEngine::Resident,
            EnginePlacement::CompleteResidencyFits,
        ),
        true,
    );
    assert_eq!(choice.0, Qwen3_5SelectedEngine::Streaming);
    assert_eq!(
        choice.1,
        EnginePlacement::CompleteResidencyDoesNotFit { shortfall_bytes: 0 }
    );
}

#[test]
fn should_keep_the_resident_choice_for_other_generations() {
    let choice = streaming_only_generation_choice(
        (
            Qwen3_5SelectedEngine::Resident,
            EnginePlacement::CompleteResidencyFits,
        ),
        false,
    );
    assert_eq!(choice.0, Qwen3_5SelectedEngine::Resident);
    assert_eq!(choice.1, EnginePlacement::CompleteResidencyFits);
}

#[test]
fn should_keep_a_streaming_choice_and_its_shortfall_unchanged() {
    let streaming_choice = (
        Qwen3_5SelectedEngine::Streaming,
        EnginePlacement::CompleteResidencyDoesNotFit {
            shortfall_bytes: 1_073_741_824,
        },
    );
    assert_eq!(
        streaming_only_generation_choice(streaming_choice, true),
        streaming_choice
    );
}
