//! Buffers output tensors and flushes safetensors shards of a bounded size.
//!
//! The writer is generic over the tensor type so its naming, index, and
//! renaming rules are unit-testable without MLX; the direct-MLX driver
//! instantiates it with `MlxArray` and a flush callback that calls the
//! runtime's safetensors writer.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::QuantizationError;

#[cfg(test)]
#[path = "shard_writer_tests.rs"]
mod shard_writer_tests;

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
        let _ = (tensor_key, tensor, tensor_bytes, flush);
        todo!("push, add bytes, flush when buffered_bytes >= limit")
    }

    /// Writes the buffer as `model-part-NNNNN.safetensors` (five digits,
    /// zero-based) and records every key in the weight map.
    pub fn flush(&mut self, flush: &mut dyn ShardFlush<Tensor>) -> Result<(), QuantizationError> {
        let _ = flush;
        todo!("no-op on an empty buffer; otherwise name, flush, record, reset")
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
        let _ = flush;
        todo!("flush, rename, write the index, return the count")
    }

    /// Final shard file name for a temporary name given the total count.
    #[must_use]
    pub fn final_shard_name(temporary_index: usize, shard_count: usize) -> String {
        let _ = (temporary_index, shard_count);
        todo!(
            "model.safetensors for one shard, else model-<index+1 five digits>-of-<count five digits>.safetensors"
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
}
