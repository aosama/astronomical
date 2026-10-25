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
pub const SCHEDULE_SEQ_LENS: [f64; 3] = [
    1.02400000000000000e+03f64,
    4.09600000000000000e+03f64,
    6.88900000000000000e+03f64,
];
pub const SCHEDULE_MU: [f64; 3] = [
    5.38709677419354827e-01f64,
    6.93548387096774244e-01f64,
    8.34324596774193572e-01f64,
];
pub const SHIFT_IMAGE_SEQ_LENS: [f64; 7] = [
    2.56000000000000000e+02f64,
    5.76000000000000000e+02f64,
    1.02400000000000000e+03f64,
    2.04800000000000000e+03f64,
    4.09600000000000000e+03f64,
    6.88900000000000000e+03f64,
    8.19200000000000000e+03f64,
];
pub const ORACLE_SHIFT_MU: [f64; 7] = [
    5.00000000000000000e-01f64,
    5.16129032258064502e-01f64,
    5.38709677419354827e-01f64,
    5.90322580645161299e-01f64,
    6.93548387096774244e-01f64,
    8.34324596774193572e-01f64,
    9.00000000000000022e-01f64,
];
pub const CUSTOM_MU: f64 = 7.15789473684210531e-01f64;
pub const ORACLE_SIGMAS_6: [f32; 7] = [
    1.000000000e+00f32,
    8.624821901e-01f32,
    7.027943730e-01f32,
    5.151054859e-01f32,
    2.913426161e-01f32,
    1.999998093e-02f32,
    0.000000000e+00f32,
];
pub const ORACLE_TIMESTEPS_6: [f32; 6] = [
    1.000000000e+03f32,
    8.624821777e+02f32,
    7.027943726e+02f32,
    5.151054688e+02f32,
    2.913426208e+02f32,
    1.999998093e+01f32,
];
pub const ORACLE_SIGMAS_8: [f32; 9] = [
    1.000000000e+00f32,
    9.160239697e-01f32,
    8.200457692e-01f32,
    7.092945576e-01f32,
    5.800748467e-01f32,
    4.273469448e-01f32,
    2.440539002e-01f32,
    1.999998093e-02f32,
    0.000000000e+00f32,
];
pub const ORACLE_TIMESTEPS_8: [f32; 8] = [
    1.000000000e+03f32,
    9.160239868e+02f32,
    8.200457764e+02f32,
    7.092945557e+02f32,
    5.800748291e+02f32,
    4.273469543e+02f32,
    2.440538940e+02f32,
    1.999998093e+01f32,
];
pub const ORACLE_SIGMAS_40: [f32; 41] = [
    1.000000000e+00f32,
    9.885730743e-01f32,
    9.768133759e-01f32,
    9.647063613e-01f32,
    9.522360563e-01f32,
    9.393863082e-01f32,
    9.261392355e-01f32,
    9.124761224e-01f32,
    8.983772397e-01f32,
    8.838211894e-01f32,
    8.687855005e-01f32,
    8.532461524e-01f32,
    8.371771574e-01f32,
    8.205510974e-01f32,
    8.033384681e-01f32,
    7.855079174e-01f32,
    7.670251727e-01f32,
    7.478538752e-01f32,
    7.279551029e-01f32,
    7.072862387e-01f32,
    6.858021021e-01f32,
    6.634531617e-01f32,
    6.401863098e-01f32,
    6.159437895e-01f32,
    5.906627774e-01f32,
    5.642751455e-01f32,
    5.367068052e-01f32,
    5.078764558e-01f32,
    4.776955843e-01f32,
    4.460667968e-01f32,
    4.128834605e-01f32,
    3.780281544e-01f32,
    3.413712382e-01f32,
    3.027692437e-01f32,
    2.620631456e-01f32,
    2.190760374e-01f32,
    1.736108065e-01f32,
    1.254466772e-01f32,
    7.433605194e-02f32,
    2.000004053e-02f32,
    0.000000000e+00f32,
];
pub const ORACLE_TIMESTEPS_40: [f32; 40] = [
    1.000000000e+03f32,
    9.885730591e+02f32,
    9.768133545e+02f32,
    9.647063599e+02f32,
    9.522360840e+02f32,
    9.393862915e+02f32,
    9.261392212e+02f32,
    9.124761353e+02f32,
    8.983772583e+02f32,
    8.838211670e+02f32,
    8.687855225e+02f32,
    8.532461548e+02f32,
    8.371771851e+02f32,
    8.205510864e+02f32,
    8.033384399e+02f32,
    7.855079346e+02f32,
    7.670251465e+02f32,
    7.478538818e+02f32,
    7.279550781e+02f32,
    7.072862549e+02f32,
    6.858021240e+02f32,
    6.634531860e+02f32,
    6.401862793e+02f32,
    6.159437866e+02f32,
    5.906627808e+02f32,
    5.642751465e+02f32,
    5.367067871e+02f32,
    5.078764648e+02f32,
    4.776955872e+02f32,
    4.460668030e+02f32,
    4.128834534e+02f32,
    3.780281677e+02f32,
    3.413712463e+02f32,
    3.027692566e+02f32,
    2.620631409e+02f32,
    2.190760345e+02f32,
    1.736108093e+02f32,
    1.254466782e+02f32,
    7.433605194e+01f32,
    2.000004005e+01f32,
];
pub const EULER_SAMPLE0: [f32; 8] = [
    5.000000000e-01f32,
    -2.500000000e-01f32,
    1.000000000e+00f32,
    0.000000000e+00f32,
    -1.500000000e+00f32,
    7.500000000e-01f32,
    2.000000000e+00f32,
    -5.000000000e-01f32,
];
pub const EULER_MODEL_OUT: [f32; 8] = [
    1.000000000e+00f32,
    -1.000000000e+00f32,
    5.000000000e-01f32,
    -5.000000000e-01f32,
    2.000000000e+00f32,
    0.000000000e+00f32,
    -2.500000000e-01f32,
    1.250000000e-01f32,
];
pub const EULER_STEPS: usize = 6;
pub const ORACLE_EULER_FINAL: [f32; 8] = [
    -5.000000000e-01f32,
    7.500000000e-01f32,
    5.000000000e-01f32,
    5.000000000e-01f32,
    -3.500000000e+00f32,
    7.500000000e-01f32,
    2.250000000e+00f32,
    -6.250000596e-01f32,
];
