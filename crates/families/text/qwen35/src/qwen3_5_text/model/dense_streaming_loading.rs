//! What a dense model streams from the SSD at load: its embedding rows and its
//! decoder layers, and the memory those leave out of the fixed core.

use super::{Qwen3_5FeedForwardArchitecture, TextExecutionError, ValidatedQwen3_5Artifact};
use crate::qwen3_5_core::model::disk_embedding_rows::Qwen3_5DiskEmbeddingRows;
use crate::qwen3_5_text::model::dense_layer_streamer::Qwen3_5DenseLayerStreamer;

/// A streaming dense model's core and streaming workspace.
pub(super) struct DenseStreamingGeometry {
    pub(super) model_core_payload_bytes: u64,
    pub(super) dense_streaming_workspace_bytes: u64,
}

/// Leaves a dense model's decoder layers out of its fixed core: kept layers
/// are given back under request pressure like retained experts, so only the
/// tensors that never stream are fixed. The two layer slots a streamed
/// forward reads into become the per-operation page the floor reserves once.
/// They stay out of the complete-expert-layer field, whose prefill rule
/// would reserve three of it; prefill MLX working buffers are measured instead.
pub(super) fn dense_streaming_geometry(
    bound_model_core_payload_bytes: u64,
    dense_layer_streamer: Option<&Qwen3_5DenseLayerStreamer>,
) -> DenseStreamingGeometry {
    let Some(dense_layer_streamer) = dense_layer_streamer else {
        return DenseStreamingGeometry {
            model_core_payload_bytes: bound_model_core_payload_bytes,
            dense_streaming_workspace_bytes: 0,
        };
    };
    let mut decoder_layer_payload_bytes = 0_u64;
    for layer_payload_bytes in dense_layer_streamer.layer_payload_bytes() {
        decoder_layer_payload_bytes =
            decoder_layer_payload_bytes.saturating_add(*layer_payload_bytes);
    }
    DenseStreamingGeometry {
        model_core_payload_bytes: bound_model_core_payload_bytes
            .saturating_sub(decoder_layer_payload_bytes),
        dense_streaming_workspace_bytes: dense_layer_streamer.streaming_slot_bytes(),
    }
}

/// A dense model's on-disk embedding rows and decoder layer streamer, both
/// planned from the validated headers before binding takes the sources.
pub(super) fn dense_streaming_sources(
    validated_artifact: &ValidatedQwen3_5Artifact,
) -> Result<
    (
        Option<Qwen3_5DiskEmbeddingRows>,
        Option<Qwen3_5DenseLayerStreamer>,
    ),
    TextExecutionError,
> {
    Ok((
        dense_disk_embedding_rows(validated_artifact)?,
        dense_layer_streamer(validated_artifact)?,
    ))
}

/// The decoder layer streamer of every dense model. It keeps the layers the
/// limit holds and reads the rest from the SSD; a model that fits keeps all.
fn dense_layer_streamer(
    validated_artifact: &ValidatedQwen3_5Artifact,
) -> Result<Option<Qwen3_5DenseLayerStreamer>, TextExecutionError> {
    match validated_artifact.config().feed_forward_architecture() {
        Qwen3_5FeedForwardArchitecture::Dense => {
            Qwen3_5DenseLayerStreamer::from_validated_artifact(validated_artifact).map(Some)
        }
        Qwen3_5FeedForwardArchitecture::MixtureOfExperts => Ok(None),
    }
}

/// Whether the loaded model reads embedding rows from the SSD.
pub(super) const fn embedding_residency_of(
    disk_embedding_rows: Option<&Qwen3_5DiskEmbeddingRows>,
) -> crate::qwen3_5_core::model_math::weights::EmbeddingResidency {
    match disk_embedding_rows {
        Some(_) => crate::qwen3_5_core::model_math::weights::EmbeddingResidency::RowsReadFromDisk,
        None => crate::qwen3_5_core::model_math::weights::EmbeddingResidency::Resident,
    }
}

/// Embedding rows read from the SSD for every dense model, whether or not
/// its text weights fit the limit; a sparse model keeps its table in memory,
/// which its expert budget already plans.
fn dense_disk_embedding_rows(
    validated_artifact: &ValidatedQwen3_5Artifact,
) -> Result<Option<Qwen3_5DiskEmbeddingRows>, TextExecutionError> {
    match validated_artifact.config().feed_forward_architecture() {
        Qwen3_5FeedForwardArchitecture::Dense => {
            Qwen3_5DiskEmbeddingRows::from_validated_artifact(validated_artifact)
        }
        Qwen3_5FeedForwardArchitecture::MixtureOfExperts => Ok(None),
    }
}

#[cfg(test)]
#[path = "dense_streaming_loading_tests.rs"]
mod dense_streaming_loading_tests;
