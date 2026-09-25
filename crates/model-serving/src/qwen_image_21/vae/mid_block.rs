//! Qwen-Image-2.1 VAE middle block: residual → attention → residual.
//!
//! Port of `QwenImage21MidBlock` with `num_layers = 1`, the artifact configuration.

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxSafetensors};

use super::QwenImage21VaeError;
use super::attention::QwenImage21VaeAttentionBlock;
use super::resnet::QwenImage21VaeResnetBlock;

#[derive(Debug)]
pub(super) struct QwenImage21VaeMidBlock {
    resnet_before_attention: QwenImage21VaeResnetBlock,
    attention: QwenImage21VaeAttentionBlock,
    resnet_after_attention: QwenImage21VaeResnetBlock,
}

impl QwenImage21VaeMidBlock {
    pub(super) fn load(
        runtime: &MlxRuntime,
        tensors: &MlxSafetensors,
        prefix: &str,
        channels: usize,
    ) -> Result<Self, QwenImage21VaeError> {
        Ok(Self {
            resnet_before_attention: QwenImage21VaeResnetBlock::load(
                runtime,
                tensors,
                &format!("{prefix}.resnets.0"),
                channels,
                channels,
            )?,
            attention: QwenImage21VaeAttentionBlock::load(
                runtime,
                tensors,
                &format!("{prefix}.attentions.0"),
                channels,
            )?,
            resnet_after_attention: QwenImage21VaeResnetBlock::load(
                runtime,
                tensors,
                &format!("{prefix}.resnets.1"),
                channels,
                channels,
            )?,
        })
    }

    /// Stage entry points let the decode loop bound each built graph to one block.
    pub(super) fn forward_resnet_before_attention(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
    ) -> Result<MlxArray, QwenImage21VaeError> {
        self.resnet_before_attention.forward(runtime, input)
    }

    pub(super) fn forward_attention(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
    ) -> Result<MlxArray, QwenImage21VaeError> {
        self.attention.forward(runtime, input)
    }

    pub(super) fn forward_resnet_after_attention(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
    ) -> Result<MlxArray, QwenImage21VaeError> {
        self.resnet_after_attention.forward(runtime, input)
    }
}
