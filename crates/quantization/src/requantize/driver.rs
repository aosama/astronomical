//! The MLX-backed walk over the tensor plan.
//!
//! Ported from OptiQ's `requantize_shardwise` and `ShardWriter.flush` (`optiq/core/shardwise.py`).

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{MlxArray, MlxDtype};

use crate::encoder::RangeSearchPolicy;
use crate::encoder::mlx::quantize_best;
use crate::error::QuantizationError;
use crate::progress::{ProgressReporter, StageProgressTracker};

use super::RequantizeOutcome;
use super::bit_policy::TensorEncodingDecision;
use super::shard_writer::{ShardFlush, ShardWriter};
use super::source_tensor_reader::{SourceShardMaps, read_module_weight};
use super::tensor_plan::{RequantizeTensorPlan, TensorPlanEntry};

/// The progress stage name the CLI prints.
const REQUANTIZE_STAGE_NAME: &str = "requantize";

/// Flushes a shard through the runtime's safetensors writer with
/// `{"format": "mlx"}` metadata, the marker mlx-lm's loader checks.
pub struct MlxShardFlush<'runtime> {
    pub runtime: &'runtime MlxRuntime,
}

impl ShardFlush<MlxArray> for MlxShardFlush<'_> {
    fn flush_shard(
        &mut self,
        shard_path: &Path,
        tensors: Vec<(String, MlxArray)>,
    ) -> Result<u64, QuantizationError> {
        let mut named_arrays = Vec::with_capacity(tensors.len());
        for (tensor_key, tensor) in &tensors {
            named_arrays.push((tensor_key.as_str(), tensor));
        }
        let mut evaluated_arrays = Vec::with_capacity(tensors.len());
        for (_, tensor) in &tensors {
            evaluated_arrays.push(tensor);
        }
        self.runtime.evaluate_arrays(&evaluated_arrays)?;
        let shard_file = match File::create(shard_path) {
            Ok(shard_file) => shard_file,
            Err(create_error) => return Err(shard_write_failure(shard_path, &create_error)),
        };
        let write_outcome =
            match self
                .runtime
                .save_safetensors(shard_file, &named_arrays, &[("format", "mlx")])
            {
                Ok(write_outcome) => write_outcome,
                Err(writer_error) => return Err(shard_write_failure(shard_path, &writer_error)),
            };
        drop(evaluated_arrays);
        drop(named_arrays);
        drop(tensors);
        self.runtime.clear_allocator_cache()?;
        Ok(write_outcome.written_byte_count())
    }
}

/// Executes the plan: for each entry read the source tensor lazily (so
/// untouched tensors are never materialized), dequantize a packed or FP8
/// source, assemble fused halves and expert stacks, encode with
/// [`quantize_best`] or pass through at bfloat16 (integer tensors stay
/// as-is), and hand the result to the shard writer. Between source shards
/// the allocator cache is drained.
///
/// Peak memory is one source tensor plus the encoder's float32 working
/// copy, independent of how many tensors the source holds.
pub fn run_requantize(
    runtime: &MlxRuntime,
    source_directory: &Path,
    output_directory: &Path,
    plan: &RequantizeTensorPlan,
    range_search_policy: RangeSearchPolicy,
    shard_bytes_limit: u64,
    progress_reporter: &mut dyn ProgressReporter,
) -> Result<RequantizeOutcome, QuantizationError> {
    if let Err(create_error) = std::fs::create_dir_all(output_directory) {
        return Err(QuantizationError::ArtifactWriteFailure {
            output_directory: output_directory.to_path_buf(),
            description: create_error.to_string(),
        });
    }
    runtime.reset_peak_memory()?;
    let mut walk = RequantizeWalk {
        runtime,
        range_search_policy,
        shard_maps: SourceShardMaps::new(source_directory),
        writer: ShardWriter::new(output_directory, shard_bytes_limit),
        flush: MlxShardFlush { runtime },
        current_shard_file_name: None,
        written_quantization_by_module: BTreeMap::new(),
        quantized_tensor_count: 0,
        passed_through_tensor_count: 0,
    };
    let mut progress_tracker = StageProgressTracker::new(
        REQUANTIZE_STAGE_NAME,
        plan.tensor_count() as u64,
        0,
        StageProgressTracker::DEFAULT_REPORT_INTERVAL,
        progress_reporter,
    );
    for entry in &plan.entries {
        walk.visit(entry)?;
        let peak_bytes = runtime.memory_snapshot()?.peak_memory_bytes();
        progress_tracker.advance(&format!(
            "{} (peak {:.2} GB)",
            entry.tensor_key,
            peak_bytes as f64 / 1e9
        ));
    }
    progress_tracker.report_now("every planned tensor written");
    walk.finish()
}

/// The walk's state across the plan.
struct RequantizeWalk<'runtime> {
    runtime: &'runtime MlxRuntime,
    range_search_policy: RangeSearchPolicy,
    shard_maps: SourceShardMaps,
    writer: ShardWriter<MlxArray>,
    flush: MlxShardFlush<'runtime>,
    current_shard_file_name: Option<String>,
    written_quantization_by_module: BTreeMap<String, (u32, u32)>,
    quantized_tensor_count: usize,
    passed_through_tensor_count: usize,
}

impl RequantizeWalk<'_> {
    /// One plan entry, encoded or passed through.
    fn visit(&mut self, entry: &TensorPlanEntry) -> Result<(), QuantizationError> {
        self.enter_shard(&entry.shard_file_name)?;
        let source_weight = match &entry.module {
            Some(module) => read_module_weight(self.runtime, &mut self.shard_maps, module)?,
            None => {
                self.shard_maps
                    .tensor(self.runtime, &entry.shard_file_name, &entry.tensor_key)?
            }
        };
        match (entry.decision, &entry.module) {
            (
                TensorEncodingDecision::Encode {
                    bit_width,
                    group_size,
                },
                Some(module),
            ) => {
                let encoded = quantize_best(
                    self.runtime,
                    &source_weight,
                    bit_width,
                    group_size,
                    self.range_search_policy,
                )?;
                drop(source_weight);
                let module_base = &module.module_name;
                self.add_tensor(format!("{module_base}.weight"), encoded.packed_weight)?;
                self.add_tensor(format!("{module_base}.scales"), encoded.scales)?;
                self.add_tensor(format!("{module_base}.biases"), encoded.biases)?;
                self.written_quantization_by_module.insert(
                    module_base.clone(),
                    (bit_width.bits(), group_size.elements()),
                );
                self.quantized_tensor_count += 1;
            }
            _ => self.pass_through(entry, source_weight)?,
        }
        Ok(())
    }

    /// Writes a tensor unencoded: floats at bfloat16, integer buffers as
    /// stored (OptiQ keeps `uint32` as-is; widening an index buffer to a
    /// float would corrupt it).
    fn pass_through(
        &mut self,
        entry: &TensorPlanEntry,
        source_weight: MlxArray,
    ) -> Result<(), QuantizationError> {
        let is_float = matches!(
            source_weight.dtype(),
            MlxDtype::Float16 | MlxDtype::Float32 | MlxDtype::BFloat16 | MlxDtype::Float64
        );
        let written_weight = if is_float {
            self.runtime.astype(&source_weight, MlxDtype::BFloat16)?
        } else {
            source_weight
        };
        self.add_tensor(entry.tensor_key.clone(), written_weight)?;
        self.passed_through_tensor_count += 1;
        Ok(())
    }

    /// Evaluates before buffering so the source read and the encoder graph
    /// are released now rather than at flush time, which keeps the peak at
    /// one tensor.
    fn add_tensor(
        &mut self,
        tensor_key: String,
        tensor: MlxArray,
    ) -> Result<(), QuantizationError> {
        tensor.evaluate()?;
        let tensor_bytes = tensor.byte_count() as u64;
        self.writer
            .add(tensor_key, tensor, tensor_bytes, &mut self.flush)
    }

    /// Closes the previous source shard and drains the allocator cache when
    /// the plan moves to a new one.
    fn enter_shard(&mut self, shard_file_name: &str) -> Result<(), QuantizationError> {
        if self.current_shard_file_name.as_deref() == Some(shard_file_name) {
            return Ok(());
        }
        self.shard_maps.retain_only(shard_file_name);
        self.current_shard_file_name = Some(shard_file_name.to_owned());
        self.runtime.clear_allocator_cache()?;
        Ok(())
    }

    /// Flushes and indexes the shards and summarizes what was written.
    fn finish(mut self) -> Result<RequantizeOutcome, QuantizationError> {
        let shard_count = self.writer.finalize(&mut self.flush)?;
        let peak_mlx_memory_bytes = self.runtime.memory_snapshot()?.peak_memory_bytes() as u64;
        Ok(RequantizeOutcome {
            shard_count,
            total_output_bytes: self.writer.total_bytes_written(),
            quantized_tensor_count: self.quantized_tensor_count,
            passed_through_tensor_count: self.passed_through_tensor_count,
            written_quantization_by_module: self.written_quantization_by_module,
            peak_mlx_memory_bytes,
        })
    }
}

/// A shard that could not be created or serialized.
fn shard_write_failure(shard_path: &Path, failure: &dyn std::error::Error) -> QuantizationError {
    QuantizationError::ArtifactWriteFailure {
        output_directory: shard_path.parent().unwrap_or(shard_path).to_path_buf(),
        description: format!("{}: {failure}", shard_path.display()),
    }
}
