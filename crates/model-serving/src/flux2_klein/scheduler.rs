//! Deterministic CPU-FP64 shifted flow-matching Euler schedule.

use thiserror::Error;

use super::Flux2KleinOfficialProfile;

const TEN_STEP_SLOPE: f64 = 8.738_095_24e-5;
const TEN_STEP_INTERCEPT: f64 = 1.898_333_33;
const TWO_HUNDRED_STEP_SLOPE: f64 = 0.000_169_27;
const TWO_HUNDRED_STEP_INTERCEPT: f64 = 0.456_666_66;
const LARGE_SEQUENCE_THRESHOLD: f64 = 4_300.0;
const TRAINING_TIMESTEP_COUNT: f64 = 1_000.0;

/// Invalid geometry supplied before MLX scalar arrays are created.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum Flux2KleinFlowSchedulerError {
    #[error("flow schedule dimensions must be positive multiples of 16")]
    InvalidDimensions,
    #[error("flow schedule image sequence length overflowed")]
    SequenceLengthOverflow,
}

/// One explicit deterministic Euler transition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Flux2KleinFlowStep {
    timestep: f64,
    sigma: f64,
    next_sigma: f64,
    delta_sigma: f64,
}

impl Flux2KleinFlowStep {
    pub const fn timestep(&self) -> f64 {
        self.timestep
    }
    pub const fn sigma(&self) -> f64 {
        self.sigma
    }
    pub const fn next_sigma(&self) -> f64 {
        self.next_sigma
    }
    pub const fn delta_sigma(&self) -> f64 {
        self.delta_sigma
    }
}

/// Complete scalar schedule retained for one four-call transformer execution.
#[derive(Clone, Debug, PartialEq)]
pub struct Flux2KleinFlowSchedule {
    image_sequence_length: u64,
    initial_sigma: f64,
    steps: [Flux2KleinFlowStep; 4],
}

impl Flux2KleinFlowSchedule {
    pub const fn image_sequence_length(&self) -> u64 {
        self.image_sequence_length
    }
    pub const fn initial_sigma(&self) -> f64 {
        self.initial_sigma
    }
    pub const fn steps(&self) -> &[Flux2KleinFlowStep; 4] {
        &self.steps
    }
}

/// Stateless scheduler entry point used before graph construction.
pub struct Flux2KleinFlowScheduler;

impl Flux2KleinFlowScheduler {
    pub fn schedule(
        width_pixels: u32,
        height_pixels: u32,
    ) -> Result<Flux2KleinFlowSchedule, Flux2KleinFlowSchedulerError> {
        if width_pixels == 0
            || height_pixels == 0
            || !width_pixels.is_multiple_of(16)
            || !height_pixels.is_multiple_of(16)
        {
            return Err(Flux2KleinFlowSchedulerError::InvalidDimensions);
        }
        let image_sequence_length = u64::from(width_pixels / 16)
            .checked_mul(u64::from(height_pixels / 16))
            .ok_or(Flux2KleinFlowSchedulerError::SequenceLengthOverflow)?;
        let shift = empirical_shift(
            image_sequence_length as f64,
            Flux2KleinOfficialProfile::inference_step_count() as f64,
        );
        let mut sigmas = [0.0_f64; 5];
        for (step_index, sigma) in sigmas[..Flux2KleinOfficialProfile::inference_step_count()]
            .iter_mut()
            .enumerate()
        {
            let unshifted_sigma =
                1.0 - step_index as f64 / Flux2KleinOfficialProfile::inference_step_count() as f64;
            *sigma = exponential_time_shift(shift, unshifted_sigma);
        }
        sigmas[4] = 0.0;
        let steps = std::array::from_fn(|step_index| {
            let sigma = sigmas[step_index];
            let next_sigma = sigmas[step_index + 1];
            Flux2KleinFlowStep {
                timestep: sigma * TRAINING_TIMESTEP_COUNT,
                sigma,
                next_sigma,
                delta_sigma: next_sigma - sigma,
            }
        });
        Ok(Flux2KleinFlowSchedule {
            image_sequence_length,
            initial_sigma: sigmas[0],
            steps,
        })
    }
}

fn empirical_shift(image_sequence_length: f64, inference_step_count: f64) -> f64 {
    let ten_step_shift = TEN_STEP_SLOPE * image_sequence_length + TEN_STEP_INTERCEPT;
    let two_hundred_step_shift =
        TWO_HUNDRED_STEP_SLOPE * image_sequence_length + TWO_HUNDRED_STEP_INTERCEPT;
    if image_sequence_length > LARGE_SEQUENCE_THRESHOLD {
        return two_hundred_step_shift;
    }
    let step_slope = (two_hundred_step_shift - ten_step_shift) / 190.0;
    let step_intercept = two_hundred_step_shift - 200.0 * step_slope;
    step_slope * inference_step_count + step_intercept
}

fn exponential_time_shift(shift: f64, timestep: f64) -> f64 {
    let exponential_shift = shift.exp();
    exponential_shift / (exponential_shift + (1.0 / timestep - 1.0))
}
