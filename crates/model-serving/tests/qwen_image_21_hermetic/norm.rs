//! Hermetic tests for the Qwen-Image-2.1 zero-centered RMSNorm, compared against the diffusers
//! reference reimplementation in `norm_fixture.rs`.

use astronomical_model_serving::zero_center_rms_norm;

use super::norm_fixture::{
    NORM_DIM, NORM_EPS, NORM_INPUT, NORM_WEIGHT, ORACLE_NORM_OUT, ORACLE_NORM_ZERO_WEIGHT,
};

/// Slack for the f32 output: both sides compute in f64 and cast, so the gap is sub-ULP.
const NORM_TOLERANCE: f32 = 1e-5;

#[test]
fn should_match_the_diffusers_zero_centered_rms_norm() {
    let computed = zero_center_rms_norm(&NORM_INPUT, &NORM_WEIGHT, NORM_EPS);
    assert_eq!(
        computed.len(),
        NORM_DIM,
        "output width must equal input width"
    );
    for (channel, (&value, &oracle)) in computed.iter().zip(ORACLE_NORM_OUT.iter()).enumerate() {
        let diff = (value - oracle).abs();
        assert!(
            diff <= NORM_TOLERANCE,
            "channel {channel}: computed={value:?} oracle={oracle:?} (diff {diff})"
        );
    }
}

#[test]
fn should_use_unit_scale_when_the_stored_weight_is_zero() {
    // A zero stored weight must act as an effective scale of exactly 1 (weight + 1), NOT 0. This is
    // the whole point of zero-centering: a freshly-initialized norm is the identity up to the RMS
    // rescale. A missing `+ 1` would zero the output here.
    let zero_weight = [0.0f32; NORM_DIM];
    let computed = zero_center_rms_norm(&NORM_INPUT, &zero_weight, NORM_EPS);
    for (channel, (&value, &oracle)) in computed
        .iter()
        .zip(ORACLE_NORM_ZERO_WEIGHT.iter())
        .enumerate()
    {
        let diff = (value - oracle).abs();
        assert!(
            diff <= NORM_TOLERANCE,
            "zero-weight channel {channel}: computed={value:?} oracle={oracle:?} (diff {diff})"
        );
    }

    // With unit scale the output is unit-RMS (up to the eps floor): mean(out^2) ~= 1.
    let mean_square = computed.iter().map(|&v| v * v).sum::<f32>() / NORM_DIM as f32;
    assert!(
        (mean_square - 1.0).abs() <= 1e-4,
        "zero-weight output must be unit RMS, got mean-square {mean_square}"
    );
}

#[test]
fn should_scale_each_channel_by_its_own_weight_plus_one() {
    // The effective per-channel scale is (weight + 1); two channels whose weights differ by delta
    // must have outputs differing by that same factor on the shared normalized value.
    let computed = zero_center_rms_norm(&NORM_INPUT, &NORM_WEIGHT, NORM_EPS);
    let zero_weight = [0.0f32; NORM_DIM];
    let baseline = zero_center_rms_norm(&NORM_INPUT, &zero_weight, NORM_EPS);
    for channel in 0..NORM_DIM {
        let expected = baseline[channel] * (NORM_WEIGHT[channel] + 1.0);
        let diff = (computed[channel] - expected).abs();
        assert!(
            diff <= NORM_TOLERANCE,
            "channel {channel} must scale by weight+1: computed={:?} expected={expected:?}",
            computed[channel]
        );
    }
}
