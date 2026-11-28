use crate::EnginePlacement;

use super::{EngineStart, engine_start_choice};

/// A model that fits whole keeps its placement; only a generation that
/// always streamed keeps the streaming chunk size.
#[test]
fn should_report_a_model_that_fits_whole_as_fitting_whatever_its_chunk_size() {
    for (keeps_the_streaming_chunk_size, holds_whole_model) in [(false, true), (true, false)] {
        let (selected_engine, engine_placement) = engine_start_choice(
            EnginePlacement::CompleteResidencyFits,
            keeps_the_streaming_chunk_size,
            true,
        );

        assert_eq!(
            selected_engine,
            EngineStart {
                holds_whole_model,
                has_routed_experts: true
            }
        );
        assert_eq!(engine_placement, EnginePlacement::CompleteResidencyFits);
    }
}

#[test]
fn should_keep_a_models_shortfall_when_it_does_not_fit() {
    let short_by_a_gigabyte = EnginePlacement::CompleteResidencyDoesNotFit {
        shortfall_bytes: 1_073_741_824,
    };
    for keeps_the_streaming_chunk_size in [false, true] {
        assert_eq!(
            engine_start_choice(short_by_a_gigabyte, keeps_the_streaming_chunk_size, false),
            (
                EngineStart {
                    holds_whole_model: false,
                    has_routed_experts: false
                },
                short_by_a_gigabyte
            )
        );
    }
}
