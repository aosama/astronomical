#[cfg(test)]
mod tests;

use crate::artifact_validation::TensorProfile;
use crate::qwen3_5_core::artifacts::tensor_spec;
use crate::qwen3_5_core::artifacts::{Qwen3_5ArtifactError, Qwen3_5Config};

pub fn append_qwen3_5_dense_mlp_tensor_profiles(
    tensor_profiles: &mut Vec<TensorProfile>,
    decoder_layer_prefix: &str,
    hidden_size: usize,
    qwen3_5_config: &Qwen3_5Config,
) -> Result<(), Qwen3_5ArtifactError> {
    tensor_spec::append_feed_forward_projection_profiles(
        tensor_profiles,
        &format!("{decoder_layer_prefix}.mlp"),
        &[],
        (
            hidden_size,
            qwen3_5_config.dense_intermediate_size() as usize,
        ),
        qwen3_5_config,
    )
}
