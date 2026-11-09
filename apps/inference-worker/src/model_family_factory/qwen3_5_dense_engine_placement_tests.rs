//! Boundaries of the dense startup placement: weights under, at, one byte
//! over, and far over the effective MLX limit.

use astronomical_model_serving::EnginePlacement;

use super::qwen3_5_dense_engine_placement::place_dense_weights_under_ceiling;

/// The Ornith 1.5 9B MLX 6-bit text weights, as their headers total them.
const NINE_BILLION_SIX_BIT_TEXT_WEIGHT_BYTES: u64 = 7_276_225_536;
/// The effective limit a 48 GB Mac granted the Development instance.
const DEVELOPMENT_INSTANCE_LIMIT_BYTES: u64 = 39_195_874_112;

#[test]
fn should_keep_a_dense_model_resident_when_its_weights_fit_under_the_limit() {
    let engine_placement = place_dense_weights_under_ceiling(
        NINE_BILLION_SIX_BIT_TEXT_WEIGHT_BYTES,
        DEVELOPMENT_INSTANCE_LIMIT_BYTES,
    )
    .expect("dense placement should decide");

    assert_eq!(engine_placement, EnginePlacement::CompleteResidencyFits);
    assert!(engine_placement.runs_resident_engine());
}

#[test]
fn should_keep_a_dense_model_resident_when_its_weights_equal_the_limit() {
    let engine_placement = place_dense_weights_under_ceiling(
        NINE_BILLION_SIX_BIT_TEXT_WEIGHT_BYTES,
        NINE_BILLION_SIX_BIT_TEXT_WEIGHT_BYTES,
    )
    .expect("dense placement should decide");

    assert!(engine_placement.runs_resident_engine());
}

#[test]
fn should_stream_a_dense_model_whose_weights_exceed_the_limit_by_one_byte() {
    let engine_placement = place_dense_weights_under_ceiling(
        NINE_BILLION_SIX_BIT_TEXT_WEIGHT_BYTES,
        NINE_BILLION_SIX_BIT_TEXT_WEIGHT_BYTES - 1,
    )
    .expect("dense placement should decide");

    assert_eq!(
        engine_placement,
        EnginePlacement::CompleteResidencyDoesNotFit { shortfall_bytes: 1 }
    );
    assert!(!engine_placement.runs_resident_engine());
}

#[test]
fn should_report_the_whole_shortfall_under_a_small_user_limit() {
    let three_gibibyte_limit_bytes = 3 * 1024 * 1024 * 1024;

    let engine_placement = place_dense_weights_under_ceiling(
        NINE_BILLION_SIX_BIT_TEXT_WEIGHT_BYTES,
        three_gibibyte_limit_bytes,
    )
    .expect("dense placement should decide");

    assert_eq!(
        engine_placement,
        EnginePlacement::CompleteResidencyDoesNotFit {
            shortfall_bytes: NINE_BILLION_SIX_BIT_TEXT_WEIGHT_BYTES - three_gibibyte_limit_bytes
        }
    );
}

#[test]
fn should_stream_any_dense_weights_under_a_zero_limit() {
    let engine_placement =
        place_dense_weights_under_ceiling(1, 0).expect("dense placement should decide");

    assert_eq!(
        engine_placement,
        EnginePlacement::CompleteResidencyDoesNotFit { shortfall_bytes: 1 }
    );
}

#[test]
fn should_keep_empty_dense_weights_resident_under_a_zero_limit() {
    let engine_placement =
        place_dense_weights_under_ceiling(0, 0).expect("dense placement should decide");

    assert!(engine_placement.runs_resident_engine());
}
