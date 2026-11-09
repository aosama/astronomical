//! Token embeddings read from the SSD, one row per distinct token.
//!
//! A dense model that does not fit its limit cannot afford its whole
//! embedding table resident: 0.83 GB for the Ornith 9B 6-bit build, of which a
//! forward touches one row per token. This reader fetches only those rows
//! through the bounded-range reader and then runs the same gather and
//! dequantization the resident lookup runs, so the embeddings are identical.
//!
//! The cost is a host read of the token ids: during decode the sampled token
//! is still a GPU array, so each forward waits for it. That is why only the
//! streaming engine uses this reader, where the wait is small next to the SSD
//! reads it already makes; a model that fits keeps the resident table.

use std::fs::File;

use astronomical_runtime_integration::{BoundedReadInterval, MlxRuntime};
use mlx_c_rust::MlxArray;

use super::embedding_row_reads::{
    EmbeddingRowSelection, EmbeddingTableLayout, embedding_row_intervals, select_embedding_rows,
};
use crate::qwen3_5_core::artifacts::ValidatedQwen3_5Artifact;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;

/// One row-major table of the embedding: its file, layout and stored form.
#[derive(Debug)]
pub(crate) struct DiskEmbeddingTable {
    pub(crate) shard_file: File,
    pub(crate) layout: EmbeddingTableLayout,
    /// The SafeTensors dtype spelling, carried into the synthetic header.
    pub(crate) stored_dtype: String,
    /// The shape of one row, after the leading row dimension.
    pub(crate) row_shape: Vec<usize>,
}

/// The embedding tables a forward reads rows from: the packed weight, plus
/// scales and biases for a quantized table.
#[derive(Debug)]
pub(crate) struct Qwen3_5DiskEmbeddingRows {
    pub(crate) weight_table: DiskEmbeddingTable,
    pub(crate) affine_tables: Option<(DiskEmbeddingTable, DiskEmbeddingTable)>,
}

const ROWS_TENSOR_NAME: &str = "embedding_rows";
const EMBEDDING_TENSOR_PREFIX: &str = "language_model.model.embed_tokens";

impl Qwen3_5DiskEmbeddingRows {
    /// Locates the embedding tables in the validated artifact and duplicates
    /// their descriptors, before the engine takes the shard sources.
    ///
    /// Answers `None` for tied embeddings: the output head multiplies by the
    /// whole table on every token, so it stays resident and gains nothing
    /// from row reads.
    pub(crate) fn from_validated_artifact(
        validated_artifact: &ValidatedQwen3_5Artifact,
    ) -> Result<Option<Self>, Qwen3_5ExecutionError> {
        if validated_artifact.config().has_tied_embeddings() {
            return Ok(None);
        }
        let weight_table =
            disk_embedding_table(validated_artifact, "weight")?.ok_or_else(|| {
                Qwen3_5ExecutionError::MissingTensor {
                    tensor_name: format!("{EMBEDDING_TENSOR_PREFIX}.weight"),
                }
            })?;
        let affine_tables = match (
            disk_embedding_table(validated_artifact, "scales")?,
            disk_embedding_table(validated_artifact, "biases")?,
        ) {
            (Some(scales_table), Some(biases_table)) => Some((scales_table, biases_table)),
            (None, None) => None,
            _ => {
                return Err(Qwen3_5ExecutionError::InvalidInput {
                    description: "the embedding table has scales without biases, or biases without scales",
                });
            }
        };
        Ok(Some(Self {
            weight_table,
            affine_tables,
        }))
    }

    /// Embeds `token_indices` (shape `[1, tokens]`) by reading their rows.
    ///
    /// `resident_form` is the lazily bound table, never evaluated here: it
    /// supplies the quantization parameters and decides which tables apply.
    pub(crate) fn embed_tokens(
        &self,
        runtime: &MlxRuntime,
        token_indices: &MlxArray,
        resident_form: &Qwen3_5AffineWeights,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        runtime.evaluate_arrays(&[token_indices])?;
        let token_ids = token_indices.to_vec_i32()?;
        let Ok(row_selection) =
            select_embedding_rows(&token_ids, self.weight_table.layout.row_count)
        else {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "a token id is outside the on-disk embedding table",
            });
        };
        let selected_row_indices = runtime.array_from_i32(
            &row_selection.selected_row_index_by_token,
            &[1, token_ids.len() as i32],
        )?;
        let packed_rows = read_table_rows(runtime, &self.weight_table, &row_selection)?;
        let selected_weights = runtime.take_axis(&packed_rows, &selected_row_indices, 0)?;
        match (resident_form, &self.affine_tables) {
            (Qwen3_5AffineWeights::NativeBfloat16 { .. }, None) => Ok(selected_weights),
            (
                Qwen3_5AffineWeights::Quantized {
                    quantization_group_size,
                    quantization_bits,
                    ..
                },
                Some((scales_table, biases_table)),
            ) => {
                let scale_rows = read_table_rows(runtime, scales_table, &row_selection)?;
                let bias_rows = read_table_rows(runtime, biases_table, &row_selection)?;
                Ok(runtime.dequantize_affine(
                    &selected_weights,
                    &runtime.take_axis(&scale_rows, &selected_row_indices, 0)?,
                    &runtime.take_axis(&bias_rows, &selected_row_indices, 0)?,
                    *quantization_group_size,
                    *quantization_bits,
                )?)
            }
            _ => Err(Qwen3_5ExecutionError::InvalidInput {
                description: "the on-disk embedding tables do not match the bound embedding form",
            }),
        }
    }
}

/// One embedding table's file, layout and stored form, or `None` when the
/// artifact has no tensor named `embed_tokens.<parameter_name>`.
fn disk_embedding_table(
    validated_artifact: &ValidatedQwen3_5Artifact,
    parameter_name: &str,
) -> Result<Option<DiskEmbeddingTable>, Qwen3_5ExecutionError> {
    let canonical_name = format!("{EMBEDDING_TENSOR_PREFIX}.{parameter_name}");
    let Some((tensor_file_range, safetensors_source)) =
        validated_artifact.tensor_file_range(&canonical_name)
    else {
        return Ok(None);
    };
    let row_count = tensor_file_range.shape.first().copied().unwrap_or(0);
    let table_byte_count = tensor_file_range
        .file_end_offset
        .saturating_sub(tensor_file_range.file_start_offset);
    if row_count == 0 || table_byte_count % row_count as u64 != 0 {
        return Err(Qwen3_5ExecutionError::InvalidTensor {
            tensor_name: canonical_name,
            description: "embedding table rows do not divide its byte range evenly",
        });
    }
    let shard_file = safetensors_source.duplicate_file().map_err(|source| {
        Qwen3_5ExecutionError::WeightSourceDescriptor {
            description: "could not duplicate the embedding shard descriptor",
            source,
        }
    })?;
    Ok(Some(DiskEmbeddingTable {
        shard_file,
        layout: EmbeddingTableLayout {
            first_row_file_offset: tensor_file_range.file_start_offset,
            row_byte_count: (table_byte_count / row_count as u64) as usize,
            row_count,
        },
        stored_dtype: tensor_file_range.dtype.to_owned(),
        row_shape: tensor_file_range.shape[1..].to_vec(),
    }))
}

/// Reads the selected rows of one table as a lazy `[rows, row_shape..]` array.
fn read_table_rows(
    runtime: &MlxRuntime,
    embedding_table: &DiskEmbeddingTable,
    row_selection: &EmbeddingRowSelection,
) -> Result<MlxArray, Qwen3_5ExecutionError> {
    let read_intervals: Vec<BoundedReadInterval> =
        embedding_row_intervals(embedding_table.layout, &row_selection.distinct_row_ids, 0);
    let rows_payload_bytes =
        (row_selection.distinct_row_ids.len() * embedding_table.layout.row_byte_count) as u64;
    let mut rows_shape = vec![row_selection.distinct_row_ids.len()];
    rows_shape.extend_from_slice(&embedding_table.row_shape);
    let synthetic_header_bytes = synthetic_rows_header(
        &embedding_table.stored_dtype,
        &rows_shape,
        rows_payload_bytes,
    );
    let duplicated_shard_file = embedding_table.shard_file.try_clone().map_err(|source| {
        Qwen3_5ExecutionError::WeightSourceDescriptor {
            description: "could not duplicate the embedding shard descriptor",
            source,
        }
    })?;
    let rows_load = runtime.load_safetensors_from_bounded_ranges(
        duplicated_shard_file,
        synthetic_header_bytes,
        read_intervals,
        rows_payload_bytes,
        None,
    )?;
    Ok(rows_load.tensor(ROWS_TENSOR_NAME)?)
}

/// A one-tensor SafeTensors header describing the selected rows as one
/// contiguous virtual payload.
fn synthetic_rows_header(
    stored_dtype: &str,
    rows_shape: &[usize],
    rows_payload_bytes: u64,
) -> Vec<u8> {
    let encoded_header = serde_json::json!({
        ROWS_TENSOR_NAME: {
            "dtype": stored_dtype,
            "shape": rows_shape,
            "data_offsets": [0, rows_payload_bytes],
        }
    })
    .to_string()
    .into_bytes();
    let mut header_bytes = Vec::with_capacity(8 + encoded_header.len());
    header_bytes.extend_from_slice(&(encoded_header.len() as u64).to_le_bytes());
    header_bytes.extend_from_slice(&encoded_header);
    header_bytes
}
