//! The excitation Kokoro's vocoder shapes into a voice: a sum of the pitch's
//! first nine harmonics where the speech is voiced, noise where it is not
//! (HiFi-GAN NSF's `SourceModuleHnNSF` and `SineGen`).
//!
//! Kokoro computes the phase per sample, but every value it needs is
//! constant within a 300-sample block of the pitch curve: it averages the
//! per-sample phase step back down to the curve rate, accumulates it, and
//! linearly interpolates back up. This port does the same at the curve rate
//! directly, which gives the same numbers without building 300 times more
//! of them. The random initial phase Kokoro adds to sample 0 never reaches
//! the averaged values, so it is left out.

use std::f32::consts::PI;

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{MlxArray, MlxDtype};

use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::mlx_layers::linear::Linear;
use crate::speech_models::mlx_layers::scalar_arithmetic::add_scalar;
use crate::speech_models::speech_model_error::SpeechModelError;

pub(crate) const SAMPLE_RATE_HERTZ: f32 = 24_000.0;
/// Audio samples per pitch-curve step: the generator's 10 x 6 upsampling
/// times the 5-sample hop.
pub(crate) const SAMPLES_PER_CURVE_STEP: usize = 300;
const HARMONIC_COUNT: usize = 9;
const SINE_AMPLITUDE: f32 = 0.1;
const VOICED_NOISE_DEVIATION: f32 = 0.003;
const UNVOICED_NOISE_DEVIATION: f32 = SINE_AMPLITUDE / 3.0;
/// Pitch at or below this many hertz counts as unvoiced.
const VOICED_PITCH_THRESHOLD_HERTZ: f32 = 10.0;

pub(crate) struct HarmonicSource {
    harmonic_merge: Linear,
}

impl HarmonicSource {
    pub(crate) fn load(checkpoint: &CheckpointTensors<'_>) -> Result<Self, SpeechModelError> {
        Ok(Self {
            harmonic_merge: Linear::load(checkpoint, "decoder.generator.m_source.l_linear")?,
        })
    }

    /// `[curve steps, 1]` pitch in hertz to `[curve steps * 300]` excitation.
    /// Without a `noise_key` the excitation is the voiced harmonics alone:
    /// no two runtimes draw the same random numbers, so comparing against
    /// the PyTorch reference needs the noise left out on both sides.
    pub(crate) fn excitation(
        &self,
        runtime: &MlxRuntime,
        pitch_curve: &MlxArray,
        noise_key: Option<&MlxArray>,
    ) -> Result<MlxArray, SpeechModelError> {
        let sample_count = pitch_curve.shape()[0] as usize * SAMPLES_PER_CURVE_STEP;
        let sines = harmonic_sines(runtime, pitch_curve)?;
        let voiced_mask = voiced_mask(runtime, pitch_curve)?;
        let voiced_sines = runtime.multiply(&sines, &voiced_mask)?;
        let harmonic_waves = match noise_key {
            Some(noise_key) => runtime.add(
                &voiced_sines,
                &excitation_noise(runtime, &voiced_mask, sample_count, noise_key)?,
            )?,
            None => voiced_sines,
        };
        let merged = runtime.tanh(&self.harmonic_merge.forward(runtime, &harmonic_waves)?)?;
        Ok(runtime.reshape(&merged, &[sample_count as i32])?)
    }
}

/// `[samples, 9]`: the first nine harmonics of the pitch, at amplitude 0.1.
fn harmonic_sines(
    runtime: &MlxRuntime,
    pitch_curve: &MlxArray,
) -> Result<MlxArray, SpeechModelError> {
    let curve_length = pitch_curve.shape()[0] as usize;
    let harmonic_numbers: Vec<f32> = (1..=HARMONIC_COUNT)
        .map(|harmonic| harmonic as f32)
        .collect();
    let harmonics = runtime.array_from_f32(&harmonic_numbers, &[1, HARMONIC_COUNT as i32])?;
    let cycles_per_sample = runtime.multiply_scalar(
        &runtime.multiply(pitch_curve, &harmonics)?,
        1.0 / SAMPLE_RATE_HERTZ,
    )?;
    let wrapped_cycles =
        runtime.subtract(&cycles_per_sample, &runtime.floor(&cycles_per_sample)?)?;
    let accumulated_cycles = runtime.cumsum(&wrapped_cycles, 0, false, true)?;
    let block_phase = runtime.multiply_scalar(
        &accumulated_cycles,
        2.0 * PI * SAMPLES_PER_CURVE_STEP as f32,
    )?;
    let sample_phase = upsample_linearly(runtime, &block_phase, curve_length)?;
    Ok(runtime.multiply_scalar(&runtime.sin(&sample_phase)?, SINE_AMPLITUDE)?)
}

/// `[samples, 1]`: 1 where the pitch is above the voiced threshold, else 0.
fn voiced_mask(runtime: &MlxRuntime, pitch_curve: &MlxArray) -> Result<MlxArray, SpeechModelError> {
    let sample_pitch = runtime.repeat_axis(pitch_curve, SAMPLES_PER_CURVE_STEP as i32, 0)?;
    let threshold = runtime.array_from_f32(&[VOICED_PITCH_THRESHOLD_HERTZ], &[])?;
    Ok(runtime.astype(
        &runtime.greater(&sample_pitch, &threshold)?,
        MlxDtype::Float32,
    )?)
}

/// Gaussian noise, quiet (0.003) under voiced speech and louder (0.1 / 3)
/// under unvoiced speech, so fricatives get their hiss.
fn excitation_noise(
    runtime: &MlxRuntime,
    voiced_mask: &MlxArray,
    sample_count: usize,
    noise_key: &MlxArray,
) -> Result<MlxArray, SpeechModelError> {
    // `voiced * 0.003 + (1 - voiced) * 0.1 / 3`, written as one affine map.
    let voiced_offset = runtime.multiply_scalar(
        voiced_mask,
        VOICED_NOISE_DEVIATION - UNVOICED_NOISE_DEVIATION,
    )?;
    let noise_deviation = add_scalar(runtime, &voiced_offset, UNVOICED_NOISE_DEVIATION)?;
    let standard_noise = runtime.random_normal(
        &[sample_count as i32, HARMONIC_COUNT as i32],
        MlxDtype::Float32,
        0.0,
        1.0,
        noise_key,
    )?;
    Ok(runtime.multiply(&standard_noise, &noise_deviation)?)
}

/// `F.interpolate(mode="linear", align_corners=False)` by 300 along the curve.
fn upsample_linearly(
    runtime: &MlxRuntime,
    block_values: &MlxArray,
    curve_length: usize,
) -> Result<MlxArray, SpeechModelError> {
    let upsampling_plan = linear_upsampling_plan(curve_length);
    let sample_count = upsampling_plan.lower_steps.len() as i32;
    let lower_step_indices =
        runtime.array_from_i32(&upsampling_plan.lower_steps, &[sample_count])?;
    let upper_step_indices =
        runtime.array_from_i32(&upsampling_plan.upper_steps, &[sample_count])?;
    let lower = runtime.take_axis(block_values, &lower_step_indices, 0)?;
    let upper = runtime.take_axis(block_values, &upper_step_indices, 0)?;
    let upper_weights =
        runtime.array_from_f32(&upsampling_plan.upper_weights, &[sample_count, 1])?;
    let difference = runtime.subtract(&upper, &lower)?;
    Ok(runtime.add(&lower, &runtime.multiply(&difference, &upper_weights)?)?)
}

/// For each output sample: the two curve steps it lies between and how far
/// toward the second.
pub(crate) struct LinearUpsamplingPlan {
    pub(crate) lower_steps: Vec<i32>,
    pub(crate) upper_steps: Vec<i32>,
    pub(crate) upper_weights: Vec<f32>,
}

/// PyTorch's half-pixel source positions, clamped at 0 and at the last step.
pub(crate) fn linear_upsampling_plan(curve_length: usize) -> LinearUpsamplingPlan {
    let sample_count = curve_length * SAMPLES_PER_CURVE_STEP;
    let mut upsampling_plan = LinearUpsamplingPlan {
        lower_steps: Vec::with_capacity(sample_count),
        upper_steps: Vec::with_capacity(sample_count),
        upper_weights: Vec::with_capacity(sample_count),
    };
    let last_step = curve_length.saturating_sub(1);
    for sample_index in 0..sample_count {
        let source_position =
            ((sample_index as f64 + 0.5) / SAMPLES_PER_CURVE_STEP as f64 - 0.5).max(0.0);
        let lower_step = (source_position.floor() as usize).min(last_step);
        let upper_step = (lower_step + 1).min(last_step);
        upsampling_plan.lower_steps.push(lower_step as i32);
        upsampling_plan.upper_steps.push(upper_step as i32);
        upsampling_plan
            .upper_weights
            .push((source_position - lower_step as f64) as f32);
    }
    upsampling_plan
}

#[cfg(test)]
#[path = "harmonic_source_tests.rs"]
mod tests;
