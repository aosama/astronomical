//! Sinusoidal timestep embedding for Qwen-Image-2.1.
//!
//! Mirrors diffusers [`QwenImage21TemporalTimesteps`]: `cos` occupies the first half of the channels
//! and `sin` the second, over frequencies `exp(-log(max_period) * k / half)`. The DiT receives the
//! timestep already divided by 1000 (a sigma in `[0, 1]`) and `time_factor` scales it back, so the
//! argument reaches ~1000 where float32 precision would make `cos`/`sin` sensitive; this computes in
//! f64 and casts at the boundary. The deviation from the reference float32 pipeline is bounded ~1e-4
//! at the most sensitive component and is immaterial because the embedding feeds a learned linear
//! projection.

/// Default embedding width (`QwenImage21TimestepProjEmbeddings` uses `timestep_dim=256`).
pub const DEFAULT_TIMESTEP_DIM: usize = 256;
/// Default sinusoidal base period.
pub const DEFAULT_MAX_PERIOD: f64 = 10000.0;
/// Default timestep scale factor.
pub const DEFAULT_TIME_FACTOR: f64 = 1000.0;

/// Parameter-free sinusoidal timestep projection.
pub struct SinusoidalTimesteps {
    timestep_dim: usize,
    time_factor: f64,
    freqs: Vec<f64>,
}

impl Default for SinusoidalTimesteps {
    fn default() -> Self {
        Self::new(
            DEFAULT_TIMESTEP_DIM,
            DEFAULT_MAX_PERIOD,
            DEFAULT_TIME_FACTOR,
        )
    }
}

impl SinusoidalTimesteps {
    /// Build the projection for `timestep_dim` channels (must be positive).
    ///
    /// The frequency table is precomputed once: `freqs[k] = exp(-log(max_period) * k / half)` for
    /// `k` in `[0, half)` where `half = timestep_dim / 2`.
    #[must_use]
    pub fn new(timestep_dim: usize, max_period: f64, time_factor: f64) -> Self {
        assert!(timestep_dim > 0, "timestep_dim must be positive");
        let half = timestep_dim / 2;
        let log_max_period = max_period.ln();
        let freqs: Vec<f64> = (0..half)
            .map(|k| (-log_max_period * (k as f64) / (half as f64)).exp())
            .collect();
        Self {
            timestep_dim,
            time_factor,
            freqs,
        }
    }

    /// Embed one timestep into `timestep_dim` channels: the cosine half followed by the sine half.
    ///
    /// An odd `timestep_dim` appends a single zero channel, matching the reference padding.
    #[must_use]
    pub fn embed(&self, timestep: f64) -> Vec<f32> {
        let scaled = self.time_factor * timestep;
        let mut embedding = Vec::with_capacity(self.timestep_dim);
        for freq in &self.freqs {
            embedding.push((scaled * freq).cos() as f32);
        }
        for freq in &self.freqs {
            embedding.push((scaled * freq).sin() as f32);
        }
        if self.timestep_dim % 2 == 1 {
            embedding.push(0.0);
        }
        embedding
    }

    /// The precomputed frequency table, exposed for hermetic verification against the oracle.
    #[doc(hidden)]
    #[must_use]
    pub fn freqs_for_tests(&self) -> &[f64] {
        &self.freqs
    }
}
