//! The tiny short-time Fourier transform Kokoro's vocoder works in: a 20-point
//! periodic Hann window with a hop of 5 samples, matching `torch.stft` and
//! `torch.istft` with `center=True`.
//!
//! With only 11 frequency bins, both directions are ordinary convolutions
//! with a fixed Fourier basis: analysis is a strided convolution, synthesis a
//! strided transposed convolution (overlap-add). That keeps the whole
//! vocoder on the graphics processor with no FFT round trip, and the
//! overlap-add normalisation is computed exactly on the CPU once per length.

use std::f32::consts::PI;

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use crate::speech_models::speech_model_error::SpeechModelError;

pub(crate) const FOURIER_SIZE: usize = 20;
pub(crate) const HOP_SIZE: usize = 5;
pub(crate) const FREQUENCY_BIN_COUNT: usize = FOURIER_SIZE / 2 + 1;
const CENTER_PADDING: usize = FOURIER_SIZE / 2;

pub(crate) struct ShortTimeFourier {
    /// `[2 * bins, size, 1]`: cosine rows, then negative sine rows, windowed.
    analysis_kernel: MlxArray,
    /// `[1, size, 2 * bins]`: the inverse real transform of each bin, windowed.
    synthesis_kernel: MlxArray,
    /// 1 for the DC and Nyquist bins, whose imaginary part is always zero.
    edge_bin_mask: MlxArray,
}

/// `scipy.signal.get_window("hann", n, fftbins=True)`, which Kokoro uses.
fn periodic_hann_window() -> [f32; FOURIER_SIZE] {
    let mut window = [0.0; FOURIER_SIZE];
    for (sample_index, weight) in window.iter_mut().enumerate() {
        *weight = 0.5 - 0.5 * (2.0 * PI * sample_index as f32 / FOURIER_SIZE as f32).cos();
    }
    window
}

/// The windowed Fourier basis, row-major: analysis `[2 * bins, size]`
/// (cosine rows, then negative sine rows) and synthesis `[size, 2 * bins]`
/// (each bin's inverse real transform, scaled by `1 / size`).
struct FourierBasis {
    analysis_values: Vec<f32>,
    synthesis_values: Vec<f32>,
}

fn fourier_basis() -> FourierBasis {
    let window = periodic_hann_window();
    let mut fourier_basis = FourierBasis {
        analysis_values: vec![0.0f32; 2 * FREQUENCY_BIN_COUNT * FOURIER_SIZE],
        synthesis_values: vec![0.0f32; FOURIER_SIZE * 2 * FREQUENCY_BIN_COUNT],
    };
    for bin_index in 0..FREQUENCY_BIN_COUNT {
        // A one-sided spectrum counts every bin but DC and Nyquist twice.
        let is_edge_bin = bin_index == 0 || bin_index == FREQUENCY_BIN_COUNT - 1;
        let fold_weight = if is_edge_bin { 1.0 } else { 2.0 };
        for (sample_index, window_weight) in window.iter().enumerate() {
            let angle = 2.0 * PI * (bin_index * sample_index) as f32 / FOURIER_SIZE as f32;
            let analysis_row = bin_index * FOURIER_SIZE + sample_index;
            let analysis_sine_row = (FREQUENCY_BIN_COUNT + bin_index) * FOURIER_SIZE + sample_index;
            fourier_basis.analysis_values[analysis_row] = window_weight * angle.cos();
            fourier_basis.analysis_values[analysis_sine_row] = -window_weight * angle.sin();
            let synthesis_scale = fold_weight * window_weight / FOURIER_SIZE as f32;
            let synthesis_row = sample_index * 2 * FREQUENCY_BIN_COUNT;
            fourier_basis.synthesis_values[synthesis_row + bin_index] =
                synthesis_scale * angle.cos();
            fourier_basis.synthesis_values[synthesis_row + FREQUENCY_BIN_COUNT + bin_index] =
                -synthesis_scale * angle.sin();
        }
    }
    fourier_basis
}

impl ShortTimeFourier {
    pub(crate) fn new(runtime: &MlxRuntime) -> Result<Self, SpeechModelError> {
        let fourier_basis = fourier_basis();
        let mut edge_mask_values = vec![0.0f32; FREQUENCY_BIN_COUNT];
        edge_mask_values[0] = 1.0;
        edge_mask_values[FREQUENCY_BIN_COUNT - 1] = 1.0;
        let bins = FREQUENCY_BIN_COUNT as i32;
        let size = FOURIER_SIZE as i32;
        Ok(Self {
            analysis_kernel: runtime
                .array_from_f32(&fourier_basis.analysis_values, &[2 * bins, size, 1])?,
            synthesis_kernel: runtime
                .array_from_f32(&fourier_basis.synthesis_values, &[1, size, 2 * bins])?,
            edge_bin_mask: runtime.array_from_f32(&edge_mask_values, &[1, bins])?,
        })
    }

    /// `[samples]` to `[samples / hop + 1, 2 * bins]`: magnitudes, then phases.
    pub(crate) fn magnitude_and_phase(
        &self,
        runtime: &MlxRuntime,
        signal: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let spectrum = self.real_and_imaginary(runtime, signal)?;
        let frame_count = spectrum.shape()[0];
        let bins = FREQUENCY_BIN_COUNT as i32;
        let real = runtime.slice(&spectrum, &[0, 0], &[frame_count, bins], &[1, 1])?;
        let imaginary = runtime.slice(&spectrum, &[0, bins], &[frame_count, 2 * bins], &[1, 1])?;
        let magnitude = runtime.sqrt(&runtime.add(
            &runtime.multiply(&real, &real)?,
            &runtime.multiply(&imaginary, &imaginary)?,
        )?)?;
        let phase = self.fft_phase(runtime, &real, &imaginary)?;
        Ok(runtime.concatenate_axis(&[&magnitude, &phase], 1)?)
    }

    /// The reflect-padded signal through the analysis kernel,
    /// `[frames, 2 * bins]`: real parts, then imaginary parts.
    fn real_and_imaginary(
        &self,
        runtime: &MlxRuntime,
        signal: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let sample_count = signal.shape()[0] as usize;
        let reflected_indices = reflect_padded_indices(sample_count);
        let index_array =
            runtime.array_from_i32(&reflected_indices, &[reflected_indices.len() as i32])?;
        let padded = runtime.take_axis(signal, &index_array, 0)?;
        let batched = runtime.reshape(&padded, &[1, reflected_indices.len() as i32, 1])?;
        Ok(runtime.squeeze_axis(
            &runtime.conv1d(&batched, &self.analysis_kernel, HOP_SIZE as i32, 0, 1, 1)?,
            0,
        )?)
    }

    /// The phase an FFT would report for each bin.
    fn fft_phase(
        &self,
        runtime: &MlxRuntime,
        real: &MlxArray,
        imaginary: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        // An FFT returns +0 for the imaginary part of the DC and Nyquist bins,
        // so a negative real part there has phase +pi. A convolution can sum
        // to -0 instead, which would give -pi; taking |imaginary| on those two
        // bins reproduces the FFT exactly.
        let interior_mask = runtime.subtract(
            &runtime.ones_like(&self.edge_bin_mask)?,
            &self.edge_bin_mask,
        )?;
        let signed_imaginary = runtime.add(
            &runtime.multiply(imaginary, &interior_mask)?,
            &runtime.multiply(&runtime.abs(imaginary)?, &self.edge_bin_mask)?,
        )?;
        Ok(runtime.arctan2(&signed_imaginary, real)?)
    }

    /// `[frames, bins]` magnitudes and phases back to `[hop * (frames - 1)]`
    /// samples, normalised by the summed squared window as `torch.istft` does.
    pub(crate) fn inverse(
        &self,
        runtime: &MlxRuntime,
        magnitude: &MlxArray,
        phase: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let real = runtime.multiply(magnitude, &runtime.cos(phase)?)?;
        let imaginary = runtime.multiply(magnitude, &runtime.sin(phase)?)?;
        let spectrum =
            runtime.expand_dims(&runtime.concatenate_axis(&[&real, &imaginary], 1)?, 0)?;
        let overlapped = runtime.conv_transpose1d(
            &spectrum,
            &self.synthesis_kernel,
            HOP_SIZE as i32,
            0,
            1,
            0,
            1,
        )?;
        let frame_count = magnitude.shape()[0] as usize;
        if frame_count == 0 {
            return Err(SpeechModelError::Input(
                "an inverse transform needs at least one frame".to_owned(),
            ));
        }
        let overlapped_length = HOP_SIZE * (frame_count - 1) + FOURIER_SIZE;
        let samples = runtime.reshape(&overlapped, &[overlapped_length as i32])?;
        let inverse_envelope = inverse_window_envelope(frame_count);
        let envelope = runtime.array_from_f32(&inverse_envelope, &[overlapped_length as i32])?;
        let normalised = runtime.multiply(&samples, &envelope)?;
        let kept_length = (HOP_SIZE * (frame_count - 1)) as i32;
        let start = CENTER_PADDING as i32;
        Ok(runtime.slice(&normalised, &[start], &[start + kept_length], &[1])?)
    }
}

/// Indices that read `signal` with `torch.stft`'s reflect padding of half a
/// window on each side (the edge sample itself is not repeated).
pub(crate) fn reflect_padded_indices(sample_count: usize) -> Vec<i32> {
    let last_index = sample_count as i64 - 1;
    (0..sample_count + 2 * CENTER_PADDING)
        .map(|padded_position| {
            let source_position = padded_position as i64 - CENTER_PADDING as i64;
            let reflected = if source_position < 0 {
                -source_position
            } else if source_position > last_index {
                2 * last_index - source_position
            } else {
                source_position
            };
            reflected.clamp(0, last_index.max(0)) as i32
        })
        .collect()
}

/// `1 / sum of squared windows` at each overlap-added sample; zero where no
/// window reaches, as `torch.istft` leaves those samples.
pub(crate) fn inverse_window_envelope(frame_count: usize) -> Vec<f32> {
    let window = periodic_hann_window();
    let overlapped_length = HOP_SIZE * (frame_count.max(1) - 1) + FOURIER_SIZE;
    let mut envelope = vec![0.0f32; overlapped_length];
    for frame_index in 0..frame_count {
        for (sample_index, weight) in window.iter().enumerate() {
            envelope[frame_index * HOP_SIZE + sample_index] += weight * weight;
        }
    }
    envelope
        .into_iter()
        .map(|summed_square| {
            if summed_square > 1e-11 {
                1.0 / summed_square
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "short_time_fourier_tests.rs"]
mod tests;
