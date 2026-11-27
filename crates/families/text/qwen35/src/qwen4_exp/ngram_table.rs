//! Flash-Next's n-gram table, read from the SSD a row at a time.
//!
//! The table is 29.8 GB at 4 bits and a forward reads sixteen 160-wide rows
//! per token, so it is never held in memory and never enters the MLX
//! ceiling or the `astro_expert_cache`: a phrase lookup must not evict a
//! routed expert. Each forward reads exactly the rows its tokens hash to,
//! through the same bounded-range reader the on-disk embedding rows use,
//! and then gathers and dequantizes them in one step on the GPU, which is
//! the reference's `QuantizedEmbedding` lookup. Repeated rows are read once,
//! and the `macos_file_cache` keeps the hot ones without any memo of ours.

use mlx_bridge::MlxArray;
use runtime_integration::MlxRuntime;
use text_generation::execution_error::TextExecutionError;

use crate::qwen3_5_core::artifacts::ValidatedQwen3_5Artifact;
use crate::qwen3_5_core::artifacts::qwen4_exp_tensor_spec::NGRAM_TABLE_SHARD_STEM;
use crate::qwen3_5_core::configuration::Qwen4ExpArchitecture;
use crate::qwen3_5_core::model::disk_embedding_rows::{
    DiskEmbeddingTable, disk_row_table, read_table_rows,
};
use crate::qwen3_5_core::model::embedding_row_reads::select_embedding_rows;
use crate::qwen3_5_core::model_math::compressed_lexicon::CompressedLexicon;

use super::ngram_hashing::NgramHashTables;
use super::ngram_table_geometry::NgramTableGeometry;

/// One shard's packed rows and their scales and biases.
#[derive(Debug)]
struct NgramShard {
    packed_rows: DiskEmbeddingTable,
    scales: DiskEmbeddingTable,
    biases: DiskEmbeddingTable,
}

/// Every shard of the table, opened from the artifact before binding takes
/// its sources.
#[derive(Debug)]
pub struct NgramTableShards {
    shards: Vec<NgramShard>,
    quantization_group_size: i32,
    quantization_bits: i32,
}

impl NgramTableShards {
    pub fn open(
        validated_artifact: &ValidatedQwen3_5Artifact,
        architecture: &Qwen4ExpArchitecture,
    ) -> Result<Self, TextExecutionError> {
        let geometry = NgramTableGeometry::for_architecture(architecture);
        let layer_prefix = format!(
            "language_model.model.layers.{}",
            architecture.ngram_layer_index
        );
        let mut shards = Vec::with_capacity(geometry.shard_count());
        for shard_index in 0..geometry.shard_count() {
            let module_name = format!("{layer_prefix}.{NGRAM_TABLE_SHARD_STEM}.{shard_index}");
            shards.push(NgramShard {
                packed_rows: required_table(validated_artifact, &module_name, "weight")?,
                scales: required_table(validated_artifact, &module_name, "scales")?,
                biases: required_table(validated_artifact, &module_name, "biases")?,
            });
        }
        let first_shard_name = format!("{layer_prefix}.{NGRAM_TABLE_SHARD_STEM}.0");
        let quantization = validated_artifact
            .config()
            .quantization_profile_for_module(&first_shard_name);
        Ok(Self {
            shards,
            quantization_group_size: quantization.group_size as i32,
            quantization_bits: quantization.bits as i32,
        })
    }
}

fn required_table(
    validated_artifact: &ValidatedQwen3_5Artifact,
    module_name: &str,
    parameter_name: &str,
) -> Result<DiskEmbeddingTable, TextExecutionError> {
    let tensor_name = format!("{module_name}.{parameter_name}");
    disk_row_table(validated_artifact, tensor_name.clone())?
        .ok_or(TextExecutionError::MissingTensor { tensor_name })
}

/// The table and the hash that addresses it.
#[derive(Debug)]
pub struct NgramTable {
    shards: NgramTableShards,
    geometry: NgramTableGeometry,
    hash_tables: NgramHashTables,
}

impl NgramTable {
    /// Joins the opened shards with the hash constants the checkpoint ships,
    /// refusing constants that address a different table than the shards.
    pub fn new(
        shards: NgramTableShards,
        architecture: &Qwen4ExpArchitecture,
        hash_tables: NgramHashTables,
    ) -> Result<Self, TextExecutionError> {
        let geometry = NgramTableGeometry::for_architecture(architecture);
        if hash_tables.head_vocabulary_sizes() != geometry.head_vocabulary_sizes()
            || hash_tables.head_row_offsets() != geometry.head_row_offsets()
        {
            return Err(TextExecutionError::InvalidTensor {
                tensor_name: "ple.ple_embedding.ngram_heads_vocab_sizes".to_owned(),
                description: "the n-gram head tables do not match the configured table",
            });
        }
        Ok(Self {
            shards,
            geometry,
            hash_tables,
        })
    }

    #[must_use]
    pub const fn hash_tables(&self) -> &NgramHashTables {
        &self.hash_tables
    }

    /// The rows `row_ids` name (token-major, `heads` per token), read and
    /// dequantized as `[1, tokens, heads * row_width]`.
    pub fn gather_rows(
        &self,
        runtime: &MlxRuntime,
        row_ids: &[u64],
        token_count: i32,
    ) -> Result<MlxArray, TextExecutionError> {
        let rows_by_shard = self.local_rows_by_shard(row_ids)?;
        let mut packed_parts = Vec::new();
        let mut scale_parts = Vec::new();
        let mut bias_parts = Vec::new();
        let mut gathered_index_by_row = vec![0_i32; row_ids.len()];
        let mut shard_base = 0_i32;
        for (shard_index, requests) in rows_by_shard.iter().enumerate() {
            if requests.is_empty() {
                continue;
            }
            let shard = &self.shards.shards[shard_index];
            let local_rows = requests
                .iter()
                .map(|(_, local_row)| *local_row)
                .collect::<Vec<_>>();
            let selection = select_embedding_rows(&local_rows, shard.packed_rows.layout.row_count)?;
            packed_parts.push(read_table_rows(runtime, &shard.packed_rows, &selection)?);
            scale_parts.push(read_table_rows(runtime, &shard.scales, &selection)?);
            bias_parts.push(read_table_rows(runtime, &shard.biases, &selection)?);
            for ((request_position, _), selected_index) in
                requests.iter().zip(&selection.selected_row_index_by_token)
            {
                gathered_index_by_row[*request_position] = shard_base + selected_index;
            }
            shard_base += selection.distinct_row_ids.len() as i32;
        }
        let gathered_indices =
            runtime.array_from_i32(&gathered_index_by_row, &[row_ids.len() as i32])?;
        let packed_entries = concatenated(runtime, &packed_parts)?;
        let entry_scales = concatenated(runtime, &scale_parts)?;
        let entry_offsets = concatenated(runtime, &bias_parts)?;
        let ngram_lexicon = CompressedLexicon {
            packed_entries: &packed_entries,
            entry_scales: &entry_scales,
            entry_offsets: &entry_offsets,
            bits_per_number: self.shards.quantization_bits,
            numbers_per_group: self.shards.quantization_group_size,
        };
        let ngram_representations =
            ngram_lexicon.learned_representations_of(runtime, &gathered_indices)?;
        Ok(runtime.reshape(&ngram_representations, &[1, token_count, -1])?)
    }

    /// `(position in row_ids, row inside the shard)` grouped by shard.
    fn local_rows_by_shard(
        &self,
        row_ids: &[u64],
    ) -> Result<Vec<Vec<(usize, i32)>>, TextExecutionError> {
        let mut rows_by_shard = vec![Vec::new(); self.geometry.shard_count()];
        for (request_position, row_id) in row_ids.iter().enumerate() {
            let (shard_index, local_row) = self.geometry.shard_and_local_row(*row_id).ok_or(
                TextExecutionError::InvalidInput {
                    description: "an n-gram hash addressed a row past the table",
                },
            )?;
            rows_by_shard[shard_index].push((request_position, local_row as i32));
        }
        Ok(rows_by_shard)
    }
}

fn concatenated(runtime: &MlxRuntime, parts: &[MlxArray]) -> Result<MlxArray, TextExecutionError> {
    let part_references = parts.iter().collect::<Vec<_>>();
    Ok(runtime.concatenate_axis(&part_references, 0)?)
}

/// The n-gram table's shards for a Flash-Next artifact, opened before
/// binding takes the artifact's sources; `None` for Qwen 3.5.
pub fn open_ngram_table_shards(
    validated_artifact: &ValidatedQwen3_5Artifact,
) -> Result<Option<NgramTableShards>, TextExecutionError> {
    validated_artifact
        .config()
        .qwen4_exp_architecture()
        .map(|architecture| NgramTableShards::open(validated_artifact, &architecture))
        .transpose()
}

/// Joins the opened shards with the hash constants the bound n-gram layer
/// carries.
pub fn ngram_table_from_bound_weights(
    shards: Option<NgramTableShards>,
    config: &crate::qwen3_5_core::configuration::Qwen3_5Config,
    weights: &crate::qwen3_5_core::model_math::weights::Qwen3_5Weights,
) -> Result<Option<NgramTable>, TextExecutionError> {
    let (Some(shards), Some(architecture)) = (shards, config.qwen4_exp_architecture()) else {
        return Ok(None);
    };
    let ngram_layer = weights
        .decoder_layer_weights
        .get(architecture.ngram_layer_index as usize)
        .and_then(|layer_weights| layer_weights.flash_next())
        .and_then(|flash_next| flash_next.ngram_layer.as_ref())
        .ok_or(TextExecutionError::MissingTensor {
            tensor_name: "ple.ple_embedding.layer_multipliers".to_owned(),
        })?;
    let hash_tables = NgramHashTables::new(
        host_values(&ngram_layer.hash_multipliers)?,
        host_values(&ngram_layer.head_vocabulary_sizes)?,
        host_values(&ngram_layer.head_row_offsets)?,
        architecture.ngram_heads_per_order as usize,
        architecture.ngram_end_of_sequence_token_id,
    );
    NgramTable::new(shards, &architecture, hash_tables).map(Some)
}

/// A small int64 constant table read back to the host as unsigned values.
fn host_values(constants: &MlxArray) -> Result<Vec<u64>, TextExecutionError> {
    let mut values = Vec::new();
    for value in constants.to_vec_i64()? {
        values.push(
            u64::try_from(value).map_err(|_| TextExecutionError::InvalidTensor {
                tensor_name: "ple.ple_embedding".to_owned(),
                description: "an n-gram hash constant is negative",
            })?,
        );
    }
    Ok(values)
}
