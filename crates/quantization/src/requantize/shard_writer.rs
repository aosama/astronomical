//! Buffers output tensors and flushes safetensors shards of a bounded size.
//!
//! The writer is generic over the tensor type so its naming, index, and
//! renaming rules are unit-testable without MLX; the direct-MLX driver
//! instantiates it with `MlxArray` and a flush callback that calls the
//! runtime's safetensors writer.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::artifact::json_file::write_pretty_json;
use crate::error::QuantizationError;

#[cfg(test)]
#[path = "shard_writer_tests.rs"]
mod shard_writer_tests;

/// Target shard size: OptiQ's 4e9 bytes.
pub const DEFAULT_SHARD_BYTES: u64 = 4_000_000_000;

/// Writes one shard's buffered tensors to `shard_path` with metadata
/// `format = "mlx"`, returning the bytes written.
pub trait ShardFlush<Tensor> {
    fn flush_shard(
        &mut self,
        shard_path: &Path,
        tensors: Vec<(String, Tensor)>,
    ) -> Result<u64, QuantizationError>;
}

/// The writer state.
pub struct ShardWriter<Tensor> {
    output_directory: PathBuf,
    shard_bytes_limit: u64,
    buffered: Vec<(String, Tensor)>,
    buffered_bytes: u64,
    /// Tensor key → temporary shard file name.
    weight_map: BTreeMap<String, String>,
    temporary_shard_names: Vec<String>,
    total_bytes_written: u64,
}

impl<Tensor> ShardWriter<Tensor> {
    pub fn new(output_directory: &Path, shard_bytes_limit: u64) -> Self {
        Self {
            output_directory: output_directory.to_path_buf(),
            shard_bytes_limit,
            buffered: Vec::new(),
            buffered_bytes: 0,
            weight_map: BTreeMap::new(),
            temporary_shard_names: Vec::new(),
            total_bytes_written: 0,
        }
    }

    /// Buffers one tensor and flushes when the buffer reaches the limit.
    pub fn add(
        &mut self,
        tensor_key: String,
        tensor: Tensor,
        tensor_bytes: u64,
        flush: &mut dyn ShardFlush<Tensor>,
    ) -> Result<(), QuantizationError> {
        self.buffered.push((tensor_key, tensor));
        self.buffered_bytes += tensor_bytes;
        if self.buffered_bytes >= self.shard_bytes_limit {
            self.flush(flush)?;
        }
        Ok(())
    }

    /// Writes the buffer as `model-part-NNNNN.safetensors` (five digits,
    /// zero-based) and records every key in the weight map.
    pub fn flush(&mut self, flush: &mut dyn ShardFlush<Tensor>) -> Result<(), QuantizationError> {
        if self.buffered.is_empty() {
            return Ok(());
        }
        let temporary_shard_name = format!(
            "model-part-{:05}.safetensors",
            self.temporary_shard_names.len()
        );
        let buffered_tensors = std::mem::take(&mut self.buffered);
        for (tensor_key, _) in &buffered_tensors {
            self.weight_map
                .insert(tensor_key.clone(), temporary_shard_name.clone());
        }
        let shard_path = self.output_directory.join(&temporary_shard_name);
        self.total_bytes_written += flush.flush_shard(&shard_path, buffered_tensors)?;
        self.temporary_shard_names.push(temporary_shard_name);
        self.buffered_bytes = 0;
        Ok(())
    }

    /// Flushes, renames to the conventional pattern, writes the index, and
    /// returns the shard count.
    ///
    /// Work to port (OptiQ's `ShardWriter.finalize`): one shard is renamed
    /// `model.safetensors`; `n > 1` shards become
    /// `model-00001-of-0000n.safetensors` (one-based). The index
    /// `model.safetensors.index.json` carries `{"metadata": {"total_size":
    /// total}, "weight_map": {key: final_name}}`. The index is written even
    /// for a single shard, which is what Astronomical's loader expects.
    pub fn finalize(
        &mut self,
        flush: &mut dyn ShardFlush<Tensor>,
    ) -> Result<usize, QuantizationError> {
        self.flush(flush)?;
        let shard_count = self.temporary_shard_names.len();
        let mut final_name_by_temporary_name = BTreeMap::new();
        for (temporary_index, temporary_shard_name) in self.temporary_shard_names.iter().enumerate()
        {
            let final_shard_name = Self::final_shard_name(temporary_index, shard_count);
            let rename_outcome = std::fs::rename(
                self.output_directory.join(temporary_shard_name),
                self.output_directory.join(&final_shard_name),
            );
            if let Err(rename_error) = rename_outcome {
                return Err(self.write_failure(&rename_error.to_string()));
            }
            final_name_by_temporary_name.insert(temporary_shard_name.clone(), final_shard_name);
        }
        let mut final_weight_map = BTreeMap::new();
        for (tensor_key, temporary_shard_name) in &self.weight_map {
            let final_shard_name = final_name_by_temporary_name[temporary_shard_name].clone();
            final_weight_map.insert(tensor_key.clone(), final_shard_name);
        }
        self.write_index(final_weight_map)?;
        Ok(shard_count)
    }

    /// Final shard file name for a temporary name given the total count.
    #[must_use]
    pub fn final_shard_name(temporary_index: usize, shard_count: usize) -> String {
        if shard_count == 1 {
            return SINGLE_SHARD_NAME.to_owned();
        }
        format!(
            "model-{:05}-of-{:05}.safetensors",
            temporary_index + 1,
            shard_count
        )
    }

    #[must_use]
    pub fn output_directory(&self) -> &Path {
        &self.output_directory
    }

    #[must_use]
    pub const fn shard_bytes_limit(&self) -> u64 {
        self.shard_bytes_limit
    }

    #[must_use]
    pub const fn total_bytes_written(&self) -> u64 {
        self.total_bytes_written
    }

    #[must_use]
    pub fn buffered_tensor_count(&self) -> usize {
        self.buffered.len()
    }

    #[must_use]
    pub const fn buffered_bytes(&self) -> u64 {
        self.buffered_bytes
    }

    #[must_use]
    pub fn weight_map(&self) -> &BTreeMap<String, String> {
        &self.weight_map
    }

    #[must_use]
    pub fn temporary_shard_names(&self) -> &[String] {
        &self.temporary_shard_names
    }

    /// `model.safetensors.index.json`, written even for a single shard
    /// because Astronomical's loader reads the index.
    fn write_index(
        &self,
        final_weight_map: BTreeMap<String, String>,
    ) -> Result<(), QuantizationError> {
        let shard_index = ShardIndexDocument {
            metadata: ShardIndexMetadata {
                total_size: self.total_bytes_written,
            },
            weight_map: final_weight_map,
        };
        write_pretty_json(
            &shard_index,
            &self.output_directory.join(SHARD_INDEX_FILE_NAME),
        )
    }

    fn write_failure(&self, description: &str) -> QuantizationError {
        QuantizationError::ArtifactWriteFailure {
            output_directory: self.output_directory.clone(),
            description: description.to_owned(),
        }
    }
}

/// The name a lone shard takes.
const SINGLE_SHARD_NAME: &str = "model.safetensors";

/// The shard index loaders read to find each tensor.
pub const SHARD_INDEX_FILE_NAME: &str = "model.safetensors.index.json";

/// The index document in the Hugging Face layout.
#[derive(Serialize)]
struct ShardIndexDocument {
    metadata: ShardIndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Serialize)]
struct ShardIndexMetadata {
    total_size: u64,
}
