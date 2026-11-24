//! Unit tests for the Qwen-Image-2.1 flow-matching scheduler (dynamic shift, sigma/timestep
//! schedule with the terminal stretch, and Euler step), compared against the diffusers reference
//! reimplementation in `scheduler_fixture.rs` (generated from the reviewed artifact's scheduler
//! config: 256/8192/0.5/0.9, `shift_terminal = 0.02`).

use super::*;

/// The reviewed artifact's scheduler constants are what the oracle fixture was generated from.
const REVIEWED_PARAMS: FlowMatchSchedulerParams = FlowMatchSchedulerParams::reviewed_artifact();

/// Slack for the f64 shift: both sides evaluate the same linear formula, so the gap is rounding-only.
const SHIFT_TOLERANCE: f64 = 1e-12;

// ---- Dynamic shift (calculate_shift) ---------------------------------------------------------

#[test]
fn should_match_the_diffusers_dynamic_shift() {
    assert_eq!(SHIFT_IMAGE_SEQ_LENS.len(), ORACLE_SHIFT_MU.len());
    for (index, &image_seq_len) in SHIFT_IMAGE_SEQ_LENS.iter().enumerate() {
        let computed = default_shift(image_seq_len);
        let oracle = ORACLE_SHIFT_MU[index];
        let diff = (computed - oracle).abs();
        assert!(
            diff <= SHIFT_TOLERANCE,
            "image_seq_len {image_seq_len}: computed={computed:?} oracle={oracle:?} (diff {diff})"
        );
    }
}

#[test]
fn should_anchor_the_shift_at_the_base_and_max_sequence_lengths() {
    let base_mu = default_shift(REVIEWED_PARAMS.base_seq_len);
    let max_mu = default_shift(REVIEWED_PARAMS.max_seq_len);
    assert!(
        (base_mu - REVIEWED_PARAMS.base_shift).abs() <= SHIFT_TOLERANCE,
        "mu at base_seq_len must equal base_shift, got {base_mu}"
    );
    assert!(
        (max_mu - REVIEWED_PARAMS.max_shift).abs() <= SHIFT_TOLERANCE,
        "mu at max_seq_len must equal max_shift, got {max_mu}"
    );
}

#[test]
fn should_support_custom_scheduler_constants() {
    let computed = calculate_shift(1500.0, 100.0, 2000.0, 0.2, 0.9);
    let diff = (computed - CUSTOM_MU).abs();
    assert!(
        diff <= SHIFT_TOLERANCE,
        "custom-constant shift: computed={computed:?} oracle={CUSTOM_MU:?} (diff {diff})"
    );
}

#[test]
fn should_increase_monotonically_with_sequence_length() {
    let mut previous = default_shift(0.0);
    for image_seq_len in [256.0, 512.0, 1024.0, 2048.0, 4096.0, 6889.0, 8192.0] {
        let current = default_shift(image_seq_len);
        assert!(
            current > previous,
            "mu must increase: {image_seq_len} gave {current} after {previous}"
        );
        previous = current;
    }
}

// ---- Sigma / timestep schedule ---------------------------------------------------------------

/// The schedule and Euler step follow the reference's exact f32 op order, so the results are
/// bit-identical; a mismatch means an op or an operand diverged from the reference.
fn assert_f32_exact(label: &str, computed: &[f32], oracle: &[f32]) {
    assert_eq!(computed.len(), oracle.len(), "{label}: length mismatch");
    for (index, (&got, &want)) in computed.iter().zip(oracle.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{label}[{index}]: computed={got:?} oracle={want:?}"
        );
    }
}

fn oracle_sigmas(steps: usize) -> &'static [f32] {
    match steps {
        6 => ORACLE_SIGMAS_6.as_slice(),
        8 => ORACLE_SIGMAS_8.as_slice(),
        40 => ORACLE_SIGMAS_40.as_slice(),
        _ => panic!("no oracle fixture for {steps} steps"),
    }
}

fn oracle_timesteps(steps: usize) -> &'static [f32] {
    match steps {
        6 => ORACLE_TIMESTEPS_6.as_slice(),
        8 => ORACLE_TIMESTEPS_8.as_slice(),
        40 => ORACLE_TIMESTEPS_40.as_slice(),
        _ => panic!("no oracle fixture for {steps} steps"),
    }
}

#[test]
fn should_build_the_diffusers_sigma_schedule() {
    for (case, (&steps, (&seq_len, &mu))) in SCHEDULE_STEPS
        .iter()
        .zip(SCHEDULE_SEQ_LENS.iter().zip(SCHEDULE_MU.iter()))
        .enumerate()
    {
        let schedule = build_schedule(steps, mu, &REVIEWED_PARAMS);
        // The schedule carries N + 1 sigmas: per-step sigmas plus the terminal 0.0.
        assert_eq!(
            schedule.sigmas.len(),
            steps + 1,
            "case {case}: sigmas must have N + 1 entries"
        );
        assert_f32_exact(
            &format!("sigmas[{seq_len}]"),
            &schedule.sigmas,
            oracle_sigmas(steps),
        );
    }
}

#[test]
fn should_match_the_diffusers_timestep_schedule() {
    for (&steps, &mu) in SCHEDULE_STEPS.iter().zip(SCHEDULE_MU.iter()) {
        let schedule = build_schedule(steps, mu, &REVIEWED_PARAMS);
        assert_eq!(schedule.timesteps.len(), steps);
        assert_f32_exact(
            &format!("timesteps[{steps}]"),
            &schedule.timesteps,
            oracle_timesteps(steps),
        );
    }
}

#[test]
fn should_stretch_the_schedule_to_the_reviewed_terminal_sigma() {
    // The reviewed config sets shift_terminal=0.02, so the last per-step sigma must land at
    // 0.02 (before the appended 0.0), not at the raw shifted 1/N value.
    for (&steps, &mu) in SCHEDULE_STEPS.iter().zip(SCHEDULE_MU.iter()) {
        let schedule = build_schedule(steps, mu, &REVIEWED_PARAMS);
        let pre_terminal = schedule.sigmas[steps - 1];
        let diff = (pre_terminal - REVIEWED_PARAMS.shift_terminal as f32).abs();
        assert!(
            diff <= 1e-5,
            "steps {steps}: pre-terminal sigma {pre_terminal} must equal shift_terminal 0.02 (diff {diff})"
        );
    }
}

#[test]
fn should_skip_the_terminal_stretch_when_shift_terminal_is_zero() {
    // With shift_terminal disabled (the reference's truthiness check), the schedule ends at the
    // raw dynamically-shifted 1/N value, which for the smallest fixture mu is well above 0.02.
    let mut params = FlowMatchSchedulerParams::reviewed_artifact();
    params.shift_terminal = 0.0;
    let schedule = build_schedule(SCHEDULE_STEPS[0], SCHEDULE_MU[0], &params);
    let pre_terminal = schedule.sigmas[SCHEDULE_STEPS[0] - 1];
    assert!(
        pre_terminal > 0.1,
        "without the stretch the pre-terminal sigma must stay at the shifted 1/N value, got {pre_terminal}"
    );
}

#[test]
fn should_span_the_noise_range_from_full_to_terminal_zero() {
    // The schedule must start at sigma == 1 (full noise), decrease monotonically, and append a
    // terminal 0.0 so the final Euler step lands on the clean image.
    for (&steps, &mu) in SCHEDULE_STEPS.iter().zip(SCHEDULE_MU.iter()) {
        let schedule = build_schedule(steps, mu, &REVIEWED_PARAMS);
        assert_eq!(
            schedule.sigmas[0], 1.0,
            "first sigma must be full noise (1.0)"
        );
        assert_eq!(
            *schedule.sigmas.last().unwrap(),
            0.0,
            "terminal sigma must be 0.0"
        );
        for index in 0..steps {
            assert!(
                schedule.sigmas[index] > schedule.sigmas[index + 1],
                "sigmas must strictly decrease at step {index}: {:?}",
                schedule.sigmas
            );
        }
    }
}

// ---- Transformer time input -------------------------------------------------------------------

#[test]
fn should_hand_the_transformer_normalized_sigmas_decreasing_from_full_noise() {
    // The denoise loop reads its per-step time input through this accessor. It has to be the sigma
    // column, not the training-timestep column: handing the DiT sigma x 1000 renders noise while
    // every structural check on the latent grid still passes, so pin the convention here.
    for (&steps, &mu) in SCHEDULE_STEPS.iter().zip(SCHEDULE_MU.iter()) {
        let schedule = build_schedule(steps, mu, &REVIEWED_PARAMS);
        let transformer_sigmas = schedule.transformer_sigmas();
        assert_eq!(
            transformer_sigmas.len(),
            steps,
            "one time input per denoising step"
        );
        for step in 0..steps {
            assert_eq!(
                transformer_sigmas[step].to_bits(),
                schedule.sigmas[step].to_bits(),
                "steps {steps}: step {step} must pair the transformer with its own Euler bound"
            );
            assert!(
                (transformer_sigmas[step] - schedule.timesteps[step] / NUM_TRAIN_TIMESTEPS).abs()
                    <= f32::EPSILON,
                "steps {steps}: step {step} time input {} must be the sigma, not the training timestep {}",
                transformer_sigmas[step],
                schedule.timesteps[step]
            );
        }
        assert_eq!(
            transformer_sigmas[0], 1.0,
            "steps {steps}: denoising starts at full noise"
        );
        assert!(
            transformer_sigmas[steps - 1] < 0.05,
            "steps {steps}: the last step must run near the terminal sigma, got {}",
            transformer_sigmas[steps - 1]
        );
    }
}

// ---- Euler step ------------------------------------------------------------------------------

#[test]
fn should_match_a_single_euler_step() {
    let schedule = build_schedule(6, SCHEDULE_MU[0], &REVIEWED_PARAMS);
    // Reconstruct the oracle's per-step update at index 2 from its inputs and check the whole vector.
    let sigma = schedule.sigmas[2];
    let sigma_next = schedule.sigmas[3];
    let prev = euler_step(&EULER_SAMPLE0, &EULER_MODEL_OUT, sigma, sigma_next);
    let dt = sigma_next - sigma;
    for (index, (&got, &sample)) in prev.iter().zip(EULER_SAMPLE0.iter()).enumerate() {
        let want = sample + dt * EULER_MODEL_OUT[index];
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "euler step channel {index}: computed={got:?} want={want:?}"
        );
    }
}

#[test]
fn should_match_the_diffusers_euler_denoise_loop() {
    // Drive the full N=6 schedule with a fixed synthetic model output and compare the final latent
    // to the reference loop, which is the accumulation the runtime will reproduce with real DiT output.
    let schedule = build_schedule(6, SCHEDULE_MU[0], &REVIEWED_PARAMS);
    let mut sample = EULER_SAMPLE0.to_vec();
    for step in 0..EULER_STEPS {
        sample = euler_step(
            &sample,
            &EULER_MODEL_OUT,
            schedule.sigmas[step],
            schedule.sigmas[step + 1],
        );
    }
    assert_f32_exact("euler_final", &sample, ORACLE_EULER_FINAL.as_slice());
}

// Fixed reference values for the Qwen-Image-2.1 flow-matching Euler schedule and step, derived from
// the diffusers FlowMatchEulerDiscreteScheduler driven by the reviewed artifact scheduler config
// (256/8192/0.5/0.9, shift_terminal 0.02). Do not edit by hand; they are cross-checked data.
//
// Pure oracle data consumed by the test module via `use`.
pub const SCHEDULE_STEPS: [usize; 3] = [6usize, 8usize, 40usize];
pub const SCHEDULE_SEQ_LENS: [f64; 3] = [1.024e3_f64, 4.096e3_f64, 6.889e3_f64];
pub const SCHEDULE_MU: [f64; 3] = [
    5.387_096_774_193_548e-1_f64,
    6.935_483_870_967_742e-1_f64,
    8.343_245_967_741_936e-1_f64,
];
pub const SHIFT_IMAGE_SEQ_LENS: [f64; 7] = [
    2.56e2_f64,
    5.76e2_f64,
    1.024e3_f64,
    2.048e3_f64,
    4.096e3_f64,
    6.889e3_f64,
    8.192e3_f64,
];
pub const ORACLE_SHIFT_MU: [f64; 7] = [
    5e-1_f64,
    5.161_290_322_580_645e-1_f64,
    5.387_096_774_193_548e-1_f64,
    5.903_225_806_451_613e-1_f64,
    6.935_483_870_967_742e-1_f64,
    8.343_245_967_741_936e-1_f64,
    9e-1_f64,
];
pub const CUSTOM_MU: f64 = 7.157_894_736_842_105e-1_f64;
pub const ORACLE_SIGMAS_6: [f32; 7] = [
    1_f32,
    8.624_822e-1_f32,
    7.027_944e-1_f32,
    5.151_055e-1_f32,
    2.913_426_2e-1_f32,
    1.999_998e-2_f32,
    0.000000000e+00f32,
];
pub const ORACLE_TIMESTEPS_6: [f32; 6] = [
    1e3_f32,
    8.624_822e2_f32,
    7.027_944e2_f32,
    5.151_054_7e2_f32,
    2.913_426_2e2_f32,
    1.999_998e1_f32,
];
pub const ORACLE_SIGMAS_8: [f32; 9] = [
    1_f32,
    9.160_239_7e-1_f32,
    8.200_457_7e-1_f32,
    7.092_945_6e-1_f32,
    5.800_748_5e-1_f32,
    4.273_469_4e-1_f32,
    2.440_539e-1_f32,
    1.999_998e-2_f32,
    0.000000000e+00f32,
];
pub const ORACLE_TIMESTEPS_8: [f32; 8] = [
    1e3_f32,
    9.160_24e2_f32,
    8.200_458e2_f32,
    7.092_945_6e2_f32,
    5.800_748e2_f32,
    4.273_469_5e2_f32,
    2.440_539e2_f32,
    1.999_998e1_f32,
];
pub const ORACLE_SIGMAS_40: [f32; 41] = [
    1_f32,
    9.885_731e-1_f32,
    9.768_134e-1_f32,
    9.647_063_6e-1_f32,
    9.522_360_6e-1_f32,
    9.393_863e-1_f32,
    9.261_392_4e-1_f32,
    9.124_761e-1_f32,
    8.983_772_4e-1_f32,
    8.838_212e-1_f32,
    8.687_855e-1_f32,
    8.532_461_5e-1_f32,
    8.371_771_6e-1_f32,
    8.205_511e-1_f32,
    8.033_384_7e-1_f32,
    7.855_079e-1_f32,
    7.670_252e-1_f32,
    7.478_539e-1_f32,
    7.279_551e-1_f32,
    7.072_862_4e-1_f32,
    6.858_021e-1_f32,
    6.634_531_6e-1_f32,
    6.401_863e-1_f32,
    6.159_438e-1_f32,
    5.906_628e-1_f32,
    5.642_751_5e-1_f32,
    5.367_068e-1_f32,
    5.078_764_6e-1_f32,
    4.776_955_8e-1_f32,
    4.460_668e-1_f32,
    4.128_834_6e-1_f32,
    3.780_281_5e-1_f32,
    3.413_712_4e-1_f32,
    3.027_692_4e-1_f32,
    2.620_631_5e-1_f32,
    2.190_760_4e-1_f32,
    1.736_108e-1_f32,
    1.254_466_8e-1_f32,
    7.433_605e-2_f32,
    2.000_004e-2_f32,
    0.000000000e+00f32,
];
pub const ORACLE_TIMESTEPS_40: [f32; 40] = [
    1e3_f32,
    9.885_730_6e2_f32,
    9.768_133_5e2_f32,
    9.647_063_6e2_f32,
    9.522_361e2_f32,
    9.393_863e2_f32,
    9.261_392e2_f32,
    9.124_761_4e2_f32,
    8.983_772_6e2_f32,
    8.838_211_7e2_f32,
    8.687_855e2_f32,
    8.532_461_5e2_f32,
    8.371_772e2_f32,
    8.205_511e2_f32,
    8.033_384_4e2_f32,
    7.855_079_3e2_f32,
    7.670_251_5e2_f32,
    7.478_539e2_f32,
    7.279_551e2_f32,
    7.072_862_5e2_f32,
    6.858_021e2_f32,
    6.634_532e2_f32,
    6.401_863e2_f32,
    6.159_438e2_f32,
    5.906_628e2_f32,
    5.642_751_5e2_f32,
    5.367_068e2_f32,
    5.078_764_6e2_f32,
    4.776_956e2_f32,
    4.460_668e2_f32,
    4.128_834_5e2_f32,
    3.780_281_7e2_f32,
    3.413_712_5e2_f32,
    3.027_692_6e2_f32,
    2.620_631_4e2_f32,
    2.190_760_3e2_f32,
    1.736_108_1e2_f32,
    1.254_466_8e2_f32,
    7.433_605e1_f32,
    2.000_004e1_f32,
];
pub const EULER_SAMPLE0: [f32; 8] = [
    5e-1_f32,
    -2.5e-1_f32,
    1_f32,
    0.000000000e+00f32,
    -1.5_f32,
    7.5e-1_f32,
    2_f32,
    -5e-1_f32,
];
pub const EULER_MODEL_OUT: [f32; 8] = [
    1_f32,
    -1_f32,
    5e-1_f32,
    -5e-1_f32,
    2_f32,
    0.000000000e+00f32,
    -2.5e-1_f32,
    1.25e-1_f32,
];
pub const EULER_STEPS: usize = 6;
pub const ORACLE_EULER_FINAL: [f32; 8] = [
    -5e-1_f32,
    7.5e-1_f32,
    5e-1_f32,
    5e-1_f32,
    -3.5_f32,
    7.5e-1_f32,
    2.25_f32,
    -6.250_000_6e-1_f32,
];
