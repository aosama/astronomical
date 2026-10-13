//! Hermetic tests for the Qwen-Image-2.1 flow-matching scheduler (dynamic shift, sigma/timestep
//! schedule with the terminal stretch, and Euler step), compared against the diffusers reference
//! reimplementation in `scheduler_fixture.rs` (generated from the reviewed artifact's scheduler
//! config: 256/8192/0.5/0.9, `shift_terminal = 0.02`).

use astronomical_model_serving::{
    FlowMatchSchedulerParams, NUM_TRAIN_TIMESTEPS, build_schedule, calculate_shift, default_shift,
    euler_step,
};

use super::scheduler_fixture::{
    CUSTOM_MU, EULER_MODEL_OUT, EULER_SAMPLE0, EULER_STEPS, ORACLE_EULER_FINAL, ORACLE_SHIFT_MU,
    ORACLE_SIGMAS_6, ORACLE_SIGMAS_8, ORACLE_SIGMAS_40, ORACLE_TIMESTEPS_6, ORACLE_TIMESTEPS_8,
    ORACLE_TIMESTEPS_40, SCHEDULE_MU, SCHEDULE_SEQ_LENS, SCHEDULE_STEPS, SHIFT_IMAGE_SEQ_LENS,
};

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
