//! Qwen-Image-2.1 decoder residual block: one shortcut plus two (RMS norm → SiLU → conv) stages.
//!
//! Port of `QwenImage21ResidualBlock`. The shortcut is computed from the raw input (no norm and
//! no activation) as a 1×1 convolution only when the channel counts differ; dropout is an
//! identity at inference.

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxSafetensors};

use super::QwenImage21VaeError;
use super::convolution::QwenImage21VaeSpatialConv;
use super::rms_norm::QwenImage21VaeRmsNorm;

#[derive(Debug)]
pub(super) struct QwenImage21VaeResnetBlock {
    norm1: QwenImage21VaeRmsNorm,
    conv1: QwenImage21VaeSpatialConv,
    norm2: QwenImage21VaeRmsNorm,
    conv2: QwenImage21VaeSpatialConv,
    shortcut: Option<QwenImage21VaeSpatialConv>,
}

impl QwenImage21VaeResnetBlock {
    pub(super) fn load(
        runtime: &MlxRuntime,
        tensors: &MlxSafetensors,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
    ) -> Result<Self, QwenImage21VaeError> {
        Ok(Self {
            norm1: QwenImage21VaeRmsNorm::load(
                runtime,
                tensors,
                &format!("{prefix}.norm1"),
                input_channels,
            )?,
            conv1: QwenImage21VaeSpatialConv::load(
                tensors,
                &format!("{prefix}.conv1"),
                input_channels,
                output_channels,
                3,
                1,
            )?,
            norm2: QwenImage21VaeRmsNorm::load(
                runtime,
                tensors,
                &format!("{prefix}.norm2"),
                output_channels,
            )?,
            conv2: QwenImage21VaeSpatialConv::load(
                tensors,
                &format!("{prefix}.conv2"),
                output_channels,
                output_channels,
                3,
                1,
            )?,
            shortcut: (input_channels != output_channels)
                .then(|| {
                    QwenImage21VaeSpatialConv::load(
                        tensors,
                        &format!("{prefix}.conv_shortcut"),
                        input_channels,
                        output_channels,
                        1,
                        0,
                    )
                })
                .transpose()?,
        })
    }

    /// Normalize → activate → convolve, twice, then add the shortcut.
    ///
    /// Every stage is evaluated before the next one is built. At production resolution a
    /// fully-built block graph holds several full-resolution activations at once, and that peak —
    /// not the weights — is what pushes the decode past the wired-memory ceiling, so the stages
    /// release as they go. The norms do the same inside their own `forward`.
    pub(super) fn forward(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
    ) -> Result<MlxArray, QwenImage21VaeError> {
        let normalized1 = self.norm1.forward(runtime, input)?;
        normalized1.evaluate()?;
        let activated1 = runtime.silu(&normalized1)?;
        activated1.evaluate()?;
        let convolved1 = self.conv1.forward(runtime, &activated1)?;
        convolved1.evaluate()?;
        let normalized2 = self.norm2.forward(runtime, &convolved1)?;
        normalized2.evaluate()?;
        let activated2 = runtime.silu(&normalized2)?;
        activated2.evaluate()?;
        let convolved2 = self.conv2.forward(runtime, &activated2)?;
        convolved2.evaluate()?;
        match &self.shortcut {
            Some(shortcut) => {
                let projected_residual = shortcut.forward(runtime, input)?;
                projected_residual.evaluate()?;
                let residual_sum = runtime.add(&convolved2, &projected_residual)?;
                residual_sum.evaluate()?;
                Ok(residual_sum)
            }
            None => {
                let residual_sum = runtime.add(&convolved2, input)?;
                residual_sum.evaluate()?;
                Ok(residual_sum)
            }
        }
    }
}
