//! The MLX-backed walk over the tensor plan.
//!
//! Ported from OptiQ's `requantize_shardwise` and `ShardWriter.flush` (`optiq/core/shardwise.py`).

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{MlxArray, MlxDtype};

use crate::bit_width::{BitWidth, GroupSize};
use crate::encoder::RangeSearchPolicy;
use crate::encoder::mlx::quantize_best;
use crate::error::QuantizationError;
use crate::module_inventory::QuantizableModuleDescriptor;
use crate::module_inventory::output_layout::{OutputDestination, OutputValueTransform};
use crate::progress::{ProgressReporter, StageProgressTracker};

use super::RequantizeOutcome;
use super::bit_policy::TensorEncodingDecision;
use super::mtp_sidecar::mtp_layer_count;
use super::mtp_sidecar_writer::MtpSidecarWriter;
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
        vision_sidecar_tensors: Vec::new(),
        mtp_sidecar_writer: MtpSidecarWriter::new(plan.conversion_group_size.elements()),
        mtp_layer_count: mtp_layer_count_of(plan),
        total_parameter_count: 0,
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
    /// Tower tensors held until the walk ends, then written as one file.
    /// A tower is a few gigabytes at most, far below the model it rides
    /// with, so holding it does not move the single-tensor peak that matters.
    vision_sidecar_tensors: Vec<(String, MlxArray)>,
    mtp_sidecar_writer: MtpSidecarWriter,
    mtp_layer_count: usize,
    total_parameter_count: u64,
}

/// What the walk's last step wrote besides the buffered shards' contents.
struct WrittenFiles {
    shard_count: usize,
    vision_sidecar_tensor_count: usize,
    mtp_sidecar: Option<super::mtp_sidecar::MtpSidecarSummary>,
}

/// OptiQ's bfloat16 tower sidecar, written at the artifact root: mlx-lm
/// globs `model*.safetensors` and never opens it, while loaders that read
/// the shard index find the towers through it.
pub const VISION_SIDECAR_FILE_NAME: &str = "optiq_vision.safetensors";

impl RequantizeWalk<'_> {
    /// One plan entry, encoded or passed through.
    fn visit(&mut self, entry: &TensorPlanEntry) -> Result<(), QuantizationError> {
        self.enter_shard(&entry.shard_file_name)?;
        let source_weight = self.read_entry_source(entry)?;
        if entry.destination == OutputDestination::MtpSidecar {
            return self
                .mtp_sidecar_writer
                .add(self.runtime, &entry.tensor_key, source_weight);
        }
        match (entry.decision, &entry.module) {
            (
                TensorEncodingDecision::Encode {
                    bit_width,
                    group_size,
                },
                Some(module),
            ) => self.encode_module(entry, module, source_weight, bit_width, group_size),
            _ => self.pass_through(entry, source_weight),
        }
    }

    /// The entry's source as stored: a module's weight assembled from its
    /// layout (fused half, expert stack, FP8, packed), or one named tensor.
    fn read_entry_source(
        &mut self,
        entry: &TensorPlanEntry,
    ) -> Result<MlxArray, QuantizationError> {
        match &entry.module {
            Some(module) => read_module_weight(self.runtime, &mut self.shard_maps, module),
            None => self.shard_maps.tensor(
                self.runtime,
                &entry.shard_file_name,
                &entry.source_tensor_name,
            ),
        }
    }

    /// Encodes one module and writes its packed weight, scales and biases.
    fn encode_module(
        &mut self,
        entry: &TensorPlanEntry,
        module: &QuantizableModuleDescriptor,
        source_weight: MlxArray,
        bit_width: BitWidth,
        group_size: GroupSize,
    ) -> Result<(), QuantizationError> {
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
        self.total_parameter_count += element_count(&entry.shape);
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
        let written_weight = self.written_form(entry, source_weight)?;
        if entry.destination == OutputDestination::VisionSidecar {
            return Self::hold_for_vision_sidecar(
                &mut self.vision_sidecar_tensors,
                entry,
                written_weight,
            );
        }
        self.add_passed_through(entry, written_weight)
    }

    /// The tensor as it is written: laid out for the output, then cast to
    /// bfloat16 when it is a float the layout does not keep at its width.
    fn written_form(
        &self,
        entry: &TensorPlanEntry,
        source_weight: MlxArray,
    ) -> Result<MlxArray, QuantizationError> {
        let is_float = matches!(
            source_weight.dtype(),
            MlxDtype::Float16 | MlxDtype::Float32 | MlxDtype::BFloat16 | MlxDtype::Float64
        );
        let laid_out_weight = self.laid_out(&entry.value_transform, source_weight)?;
        if is_float && !entry.keeps_stored_dtype {
            return Ok(self.runtime.astype(&laid_out_weight, MlxDtype::BFloat16)?);
        }
        Ok(laid_out_weight)
    }

    /// Keeps a tower tensor until the walk writes the sidecar.
    fn hold_for_vision_sidecar(
        vision_sidecar_tensors: &mut Vec<(String, MlxArray)>,
        entry: &TensorPlanEntry,
        written_weight: MlxArray,
    ) -> Result<(), QuantizationError> {
        written_weight.evaluate()?;
        vision_sidecar_tensors.push((entry.tensor_key.clone(), written_weight));
        Ok(())
    }

    /// Buffers an unencoded tensor for the model shards and counts it.
    fn add_passed_through(
        &mut self,
        entry: &TensorPlanEntry,
        written_weight: MlxArray,
    ) -> Result<(), QuantizationError> {
        if is_leaf_module_parameter(&entry.tensor_key) {
            self.total_parameter_count += element_count(&entry.shape);
        }
        self.add_tensor(entry.tensor_key.clone(), written_weight)?;
        self.passed_through_tensor_count += 1;
        Ok(())
    }

    /// Applies the output layout's value transform in the tensor's stored
    /// dtype, before any cast, which is the order mlx-lm's sanitize and
    /// convert run in: an offset added after the cast would round twice.
    fn laid_out(
        &self,
        value_transform: &OutputValueTransform,
        source_weight: MlxArray,
    ) -> Result<MlxArray, QuantizationError> {
        match value_transform {
            OutputValueTransform::Unchanged => Ok(source_weight),
            OutputValueTransform::PermuteAxes(axes) => {
                Ok(self.runtime.transpose_axes(&source_weight, axes)?)
            }
            OutputValueTransform::AddOne => {
                let one = self.runtime.astype(
                    &self.runtime.array_from_f32(&[1.0], &[1])?,
                    source_weight.dtype(),
                )?;
                Ok(self.runtime.add(&source_weight, &one)?)
            }
        }
    }

    /// Writes the held tower tensors as `optiq_vision.safetensors` and lists
    /// them in the shard index; a source without towers writes nothing.
    fn write_vision_sidecar(&mut self) -> Result<usize, QuantizationError> {
        let vision_sidecar_tensors = std::mem::take(&mut self.vision_sidecar_tensors);
        let vision_sidecar_tensor_count = vision_sidecar_tensors.len();
        if vision_sidecar_tensor_count == 0 {
            return Ok(0);
        }
        for (tensor_key, _) in &vision_sidecar_tensors {
            self.writer
                .register_externally_stored_tensor(tensor_key.clone(), VISION_SIDECAR_FILE_NAME);
        }
        let sidecar_path = self
            .writer
            .output_directory()
            .join(VISION_SIDECAR_FILE_NAME);
        self.flush
            .flush_shard(&sidecar_path, vision_sidecar_tensors)?;
        Ok(vision_sidecar_tensor_count)
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
        let written_files = self.finalize_files()?;
        let peak_mlx_memory_bytes = self.runtime.memory_snapshot()?.peak_memory_bytes() as u64;
        Ok(RequantizeOutcome {
            shard_count: written_files.shard_count,
            total_output_bytes: self.writer.total_bytes_written(),
            quantized_tensor_count: self.quantized_tensor_count,
            passed_through_tensor_count: self.passed_through_tensor_count,
            written_quantization_by_module: self.written_quantization_by_module,
            peak_mlx_memory_bytes,
            vision_sidecar_tensor_count: written_files.vision_sidecar_tensor_count,
            mtp_sidecar: written_files.mtp_sidecar,
            total_parameter_count: self.total_parameter_count,
        })
    }

    /// Writes the tower sidecar, flushes and indexes the shards, then writes
    /// the prediction-head sidecar. The tower goes first because the shard
    /// index lists its tensors; the head is listed nowhere.
    fn finalize_files(&mut self) -> Result<WrittenFiles, QuantizationError> {
        let vision_sidecar_tensor_count = self.write_vision_sidecar()?;
        self.writer
            .set_total_parameter_count(self.total_parameter_count);
        let shard_count = self.writer.finalize(&mut self.flush)?;
        let mtp_sidecar_writer =
            std::mem::replace(&mut self.mtp_sidecar_writer, MtpSidecarWriter::new(0));
        let mtp_sidecar = mtp_sidecar_writer.finish(
            self.runtime,
            self.writer.output_directory(),
            self.mtp_layer_count,
        )?;
        Ok(WrittenFiles {
            shard_count,
            vision_sidecar_tensor_count,
            mtp_sidecar,
        })
    }
}

/// The head's distinct layers, counted from the plan's own head entries.
fn mtp_layer_count_of(plan: &RequantizeTensorPlan) -> usize {
    let mut head_tensors = Vec::new();
    for entry in &plan.entries {
        if entry.destination == OutputDestination::MtpSidecar {
            head_tensors.push(crate::module_inventory::PassThroughTensor::unchanged(
                entry.shard_file_name.clone(),
                entry.source_tensor_name.clone(),
                entry.shape.clone(),
            ));
        }
    }
    mtp_layer_count(&head_tensors)
}

/// mlx-lm's `total_parameters` sums leaf modules, whose parameters are
/// named `weight` and `bias`. An array held directly by a container module
/// (the GatedDeltaNet `A_log` and `dt_bias`) is not counted there.
fn is_leaf_module_parameter(tensor_key: &str) -> bool {
    tensor_key.ends_with(".weight") || tensor_key.ends_with(".bias")
}

/// The logical weights in a tensor of `shape`.
fn element_count(shape: &[usize]) -> u64 {
    shape.iter().product::<usize>() as u64
}

/// A shard that could not be created or serialized.
fn shard_write_failure(shard_path: &Path, failure: &dyn std::error::Error) -> QuantizationError {
    QuantizationError::ArtifactWriteFailure {
        output_directory: shard_path.parent().unwrap_or(shard_path).to_path_buf(),
        description: format!("{}: {failure}", shard_path.display()),
    }
}
