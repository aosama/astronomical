//! The geometry of one dense prefill projection: `tokens × input → output`.

use thiserror::Error;

/// A prefill projection `projected[o][t] = Σᵢ weights[o][i] · activations[i][t]`.
///
/// Activations and results are channel-major (`[channels][tokens]`) because
/// that is the 1 × 1-convolution layout the Neural Engine consumes, and the
/// same layout lets the fp32 reference and MLX use one plain matrix product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgramShape {
    input_channels: usize,
    output_channels: usize,
    token_count: usize,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ProgramShapeError {
    #[error(
        "every projection dimension must be positive, got {input_channels} → {output_channels} over {token_count} tokens"
    )]
    EmptyDimension {
        input_channels: usize,
        output_channels: usize,
        token_count: usize,
    },
    #[error("projection dimension {dimension} does not fit the 32-bit size Core ML and MLX accept")]
    DimensionTooLarge { dimension: usize },
}

impl ProgramShape {
    pub fn new(
        input_channels: usize,
        output_channels: usize,
        token_count: usize,
    ) -> Result<Self, ProgramShapeError> {
        if input_channels == 0 || output_channels == 0 || token_count == 0 {
            return Err(ProgramShapeError::EmptyDimension {
                input_channels,
                output_channels,
                token_count,
            });
        }
        for dimension in [input_channels, output_channels, token_count] {
            if i32::try_from(dimension).is_err() {
                return Err(ProgramShapeError::DimensionTooLarge { dimension });
            }
        }
        Ok(Self {
            input_channels,
            output_channels,
            token_count,
        })
    }

    #[must_use]
    pub fn input_channels(&self) -> usize {
        self.input_channels
    }

    #[must_use]
    pub fn output_channels(&self) -> usize {
        self.output_channels
    }

    #[must_use]
    pub fn token_count(&self) -> usize {
        self.token_count
    }

    #[must_use]
    pub fn weight_element_count(&self) -> usize {
        self.output_channels * self.input_channels
    }

    #[must_use]
    pub fn activation_element_count(&self) -> usize {
        self.input_channels * self.token_count
    }

    #[must_use]
    pub fn projected_element_count(&self) -> usize {
        self.output_channels * self.token_count
    }

    /// Multiplies and adds counted separately, the convention every TFLOP/s
    /// figure in issue #161 uses.
    #[must_use]
    pub fn floating_point_operations(&self) -> f64 {
        2.0 * self.output_channels as f64 * self.input_channels as f64 * self.token_count as f64
    }
}

#[cfg(test)]
#[path = "program_shape_tests.rs"]
mod tests;
