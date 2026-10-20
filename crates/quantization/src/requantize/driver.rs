//! The MLX-backed walk over the tensor plan.

use std::path::Path;

use astronomical_mlx_c_rust::MlxArray;
use astronomical_runtime_integration::MlxRuntime;

use crate::encoder::RangeSearchPolicy;
use crate::error::QuantizationError;
use crate::progress::ProgressReporter;

use super::RequantizeOutcome;
use super::shard_writer::{ShardFlush, ShardWriter};
use super::tensor_plan::RequantizeTensorPlan;

/// Flushes a shard through the runtime's safetensors writer with
/// `{"format": "mlx"}` metadata.
pub struct MlxShardFlush<'runtime> {
    pub runtime: &'runtime MlxRuntime,
}

impl ShardFlush<MlxArray> for MlxShardFlush<'_> {
    fn flush_shard(
        &mut self,
        shard_path: &Path,
        tensors: Vec<(String, MlxArray)>,
    ) -> Result<u64, QuantizationError> {
        let _ = (self.runtime, shard_path, tensors);
        todo!(
            "evaluate every tensor, open the file, runtime.save_safetensors with format=mlx, return the file size, then drain the allocator cache"
        )
    }
}

/// Executes the plan: for each entry read the source tensor (memory-mapped
/// through the runtime's safetensors loader so untouched tensors are never
/// materialized), dequantize an already-quantized source with its own
/// parameters, cast pass-through floats to bfloat16 (integer tensors stay
/// as-is), encode with [`crate::encoder::mlx::quantize_best`], hand the
/// packed weight plus `.scales` and `.biases` to the writer, record the
/// written `(bits, group_size)` for the module, advance progress with the
/// tensor key and the current peak MLX memory as the detail, and drop the
/// source array. Between shards, drain the allocator cache.
///
/// Fused and stacked expert sources (see
/// [`crate::module_inventory::ModuleWeightSource`]) are assembled here the
/// way OptiQ's `_load_bf16_weight` does: halves by slicing along axis 1,
/// stacks by gathering per-expert arrays grouped by shard so each shard is
/// mapped once, FP8 sources by `from_fp8` and block-scale multiplication.
///
/// Peak memory is one source tensor plus the encoder's float32 working copy;
/// the pipeline can size an upper bound from the plan's largest entry before
/// starting.
pub fn run_requantize(
    runtime: &MlxRuntime,
    source_directory: &Path,
    output_directory: &Path,
    plan: &RequantizeTensorPlan,
    range_search_policy: RangeSearchPolicy,
    shard_bytes_limit: u64,
    progress_reporter: &mut dyn ProgressReporter,
) -> Result<RequantizeOutcome, QuantizationError> {
    let _ = (
        runtime,
        source_directory,
        output_directory,
        plan,
        range_search_policy,
        shard_bytes_limit,
        progress_reporter,
    );
    let _ = ShardWriter::<MlxArray>::new(output_directory, shard_bytes_limit);
    todo!("walk the plan as documented above and finalize the writer")
}
