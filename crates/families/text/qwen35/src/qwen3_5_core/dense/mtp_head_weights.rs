//! The weights of a dense Qwen 3.5 multi-token-prediction head, and loading
//! them from the `mtp/` folder a pack ships beside its model.
//!
//! The head guesses the token after next: a token's embedding and the
//! model's final hidden state at the position before it are normalized,
//! joined and projected back to the hidden width; one full-attention layer
//! with a dense MLP refines that; a final norm hands it to the model's own
//! output head. It is small (137 MB on the 9B) and stays in memory whole.

use std::collections::BTreeSet;
use std::path::Path;

use mlx_bridge::MlxArray;
use runtime_integration::{MlxRuntime, MlxSafetensors};
use text_generation::execution_error::TextExecutionError;

use crate::qwen3_5_core::artifacts::dense_mtp_head_sidecar::{
    DENSE_MTP_HEAD_CONFIG_FILE_NAME, DENSE_MTP_HEAD_FOLDER_NAME, DENSE_MTP_HEAD_WEIGHTS_FILE_NAME,
    DenseMtpHeadSidecar, DenseMtpHeadSidecarError, MAXIMUM_DENSE_MTP_HEAD_CONFIG_BYTES,
};
use crate::qwen3_5_core::configuration::{Qwen3_5Config, Qwen3_5FeedForwardArchitecture};
use crate::qwen3_5_core::dense::mlp::Qwen3_5DenseMlpWeights;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5FullAttentionWeights;
use crate::qwen3_5_core::model_math::dense_projection::Qwen3_5DenseProjection;
use crate::qwen3_5_core::model_math::weights::bind_dense_mtp_head_weights;

/// Everything a dense head owns.
#[derive(Debug)]
pub struct DenseMtpHeadWeights {
    /// Norm of a token's embedding before the fusion projection.
    pub embedding_norm_scale: MlxArray,
    /// Norm of the model's final hidden state before the fusion projection.
    pub hidden_norm_scale: MlxArray,
    /// `[hidden] <- [embedding, hidden]`, in that order.
    pub fusion_projection: Qwen3_5DenseProjection,
    pub input_normalization_weight: MlxArray,
    pub attention: Qwen3_5FullAttentionWeights,
    pub post_attention_normalization_weight: MlxArray,
    pub mlp: Qwen3_5DenseMlpWeights,
    pub final_norm_scale: MlxArray,
    /// The opened weights file, which the lazy arrays above read from.
    pub head_file: MlxSafetensors,
    /// What the head occupies once a reply drafts with it.
    pub payload_byte_count: u64,
}

/// Why a head the pack ships could not be used. The model still serves; it
/// writes one token a forward, as without a head.
#[derive(Debug, thiserror::Error)]
pub enum DenseMtpHeadLoadError {
    #[error("the head's configuration was refused: {0}")]
    Configuration(#[from] DenseMtpHeadSidecarError),
    #[error("the head's {file_name} could not be read: {reason}")]
    Read {
        file_name: &'static str,
        reason: std::io::ErrorKind,
    },
    #[error("the head's weights file header is invalid: {description}")]
    Header { description: String },
    #[error("the head's config.json is larger than a head configuration can be")]
    ConfigurationTooLarge,
    #[error("the head's weights could not be bound: {0}")]
    Binding(#[from] TextExecutionError),
}

/// A head found beside the model and read as far as files go: its
/// configuration, and the tensor names its weights file declares.
#[derive(Debug)]
pub struct LocatedDenseMtpHead {
    pub sidecar: DenseMtpHeadSidecar,
    pub weights_path: std::path::PathBuf,
    pub shipped_tensor_names: BTreeSet<String>,
}

/// Loads the head a pack ships in its `mtp/` folder. `Ok(None)` when there
/// is no such folder, or it holds another kind of head.
pub fn load_dense_mtp_head_weights(
    runtime: &MlxRuntime,
    model_config: &Qwen3_5Config,
    model_directory: &Path,
) -> Result<Option<DenseMtpHeadWeights>, DenseMtpHeadLoadError> {
    let Some(located_head) = locate_dense_mtp_head(model_config, model_directory)? else {
        return Ok(None);
    };
    let weights_file = std::fs::File::open(&located_head.weights_path).map_err(|open_error| {
        DenseMtpHeadLoadError::Read {
            file_name: DENSE_MTP_HEAD_WEIGHTS_FILE_NAME,
            reason: open_error.kind(),
        }
    })?;
    let head_file = runtime
        .load_safetensors(weights_file, None)
        .map_err(TextExecutionError::from)?;
    let head_weights = bind_dense_mtp_head_weights(
        runtime,
        &located_head.sidecar.head_config(model_config),
        head_file,
        &located_head.shipped_tensor_names,
    )?;
    Ok(Some(head_weights))
}

/// Finds and reads the head's files without touching the GPU. `Ok(None)`
/// when the model is not a dense Qwen 3.5, when there is no
/// `mtp/config.json`, or when it describes another kind of head.
pub fn locate_dense_mtp_head(
    model_config: &Qwen3_5Config,
    model_directory: &Path,
) -> Result<Option<LocatedDenseMtpHead>, DenseMtpHeadLoadError> {
    // The head joins the final hidden state of a dense stack. A Flash-Next
    // model's state is several residual lanes and a mixture-of-experts
    // model routes its feed-forward, so neither can host this kind.
    if model_config.qwen4_exp_architecture().is_some()
        || model_config.feed_forward_architecture() != Qwen3_5FeedForwardArchitecture::Dense
    {
        return Ok(None);
    }
    let head_directory = model_directory.join(DENSE_MTP_HEAD_FOLDER_NAME);
    let config_path = head_directory.join(DENSE_MTP_HEAD_CONFIG_FILE_NAME);
    let Ok(config_metadata) = std::fs::metadata(&config_path) else {
        return Ok(None);
    };
    if !config_metadata.is_file() {
        return Ok(None);
    }
    if config_metadata.len() > MAXIMUM_DENSE_MTP_HEAD_CONFIG_BYTES {
        return Err(DenseMtpHeadLoadError::ConfigurationTooLarge);
    }
    let config_bytes =
        std::fs::read(&config_path).map_err(|read_error| DenseMtpHeadLoadError::Read {
            file_name: DENSE_MTP_HEAD_CONFIG_FILE_NAME,
            reason: read_error.kind(),
        })?;
    let Some(sidecar) = DenseMtpHeadSidecar::from_config_bytes(&config_bytes, model_config)? else {
        return Ok(None);
    };
    let weights_path = head_directory.join(DENSE_MTP_HEAD_WEIGHTS_FILE_NAME);
    let shipped_tensor_names = head_tensor_names(&weights_path)?;
    Ok(Some(LocatedDenseMtpHead {
        sidecar,
        weights_path,
        shipped_tensor_names,
    }))
}

/// Every tensor name the head's weights file declares, read from its header:
/// MLX's loaded map cannot list its own names.
fn head_tensor_names(weights_path: &Path) -> Result<BTreeSet<String>, DenseMtpHeadLoadError> {
    let weights_header = engine_core::expert_paging::parse_safetensors_header(weights_path)
        .map_err(|header_error| DenseMtpHeadLoadError::Header {
            description: header_error.to_string(),
        })?;
    let mut shipped_tensor_names = BTreeSet::new();
    for tensor_entry in weights_header.tensor_entries {
        shipped_tensor_names.insert(tensor_entry.tensor_name);
    }
    Ok(shipped_tensor_names)
}

#[cfg(test)]
#[path = "mtp_head_weights_tests.rs"]
mod mtp_head_weights_tests;
