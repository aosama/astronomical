//! Orchestration: one request in, one artifact out.
//!
//! Mirrors OptiQ's `run_llm_pipeline` with the stages typed and the
//! engine-specific parts injected. The pipeline itself is pure control flow
//! so its branching (reference-mode resolution, static versus exact,
//! baseline reuse, sidecar handling) is testable with fakes in the offline
//! lane; the GPU stages are reached through the `direct-mlx` feature.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::allocation::{AllocationConstraints, AllocationResult};
use crate::bit_width::{CandidateBitWidths, GroupSize};
use crate::encoder::RangeSearchPolicy;
use crate::error::QuantizationError;
use crate::sensitivity::{SensitivityMethod, SensitivityTable};

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod pipeline_tests;

/// Which reference the exact sweep measures against.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceMode {
    /// bf16 if it fits in 70% of available memory, else the uniform baseline.
    Auto,
    Bfloat16,
    Uniform4Bit,
}

/// bf16 fits when the on-disk weight bytes are at most this fraction of
/// available memory, leaving headroom for activations and the operating
/// system.
pub const BFLOAT16_MEMORY_FIT_FRACTION: f64 = 0.70;

/// Everything a conversion needs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuantizationPipelineRequest {
    /// A local directory holding `config.json`, the shards, and the
    /// tokenizer. Hugging Face resolution is the library's job
    /// (Observatory already downloads catalog entries); this crate never
    /// touches the network.
    pub source_directory: PathBuf,
    pub output_directory: PathBuf,
    /// The base model identifier recorded in the metadata.
    pub base_model_name: String,
    pub method: PipelineMethod,
    pub reference_mode: ReferenceMode,
    pub target_bits_per_weight: f64,
    pub candidate_bits: Vec<u32>,
    pub group_size: GroupSize,
    /// Calibration sequences per probe; OptiQ's CLI default is 24, roughly
    /// four from each of the six domains.
    pub calibration_sequence_count: usize,
    pub calibration_sequence_length_tokens: usize,
    /// `None` uses the bundled mix.
    pub calibration_mix_path: Option<PathBuf>,
    pub floor_components_per_block: usize,
    pub range_search_policy_enabled: bool,
    pub range_search_maximum_bits: u32,
    pub overwrite_output: bool,
}

/// Exact sweep or structural rules.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineMethod {
    ExactKlSensitivity,
    StaticStructural,
}

/// What the pipeline produced.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizationPipelineOutcome {
    pub artifact_directory: PathBuf,
    pub method: SensitivityMethod,
    pub allocation: AllocationResult,
    pub shard_count: usize,
    pub total_output_bytes: u64,
    pub sensitivity_document_written: bool,
}

/// Resolves `Auto` from the source's bf16 footprint and available memory.
///
/// Work to port (OptiQ's `_resolve_reference_mode` and
/// `_estimate_bf16_size_bytes`): sum the shard sizes, adding the byte count
/// of every `F8_*` tensor again (an FP8 byte becomes two in bf16; sizing
/// from the files alone picked a bf16 reference for a 186 GB FP8 release
/// that needed 350 GB). bf16 fits when that total is at most 70% of
/// `available_memory_bytes`. Available memory is injected by the caller so
/// the decision is testable and never reads the host from inside the
/// library.
#[must_use]
pub fn resolve_reference_mode(
    requested: ReferenceMode,
    estimated_bfloat16_weight_bytes: u64,
    available_memory_bytes: u64,
) -> ReferenceMode {
    let _ = (
        requested,
        estimated_bfloat16_weight_bytes,
        available_memory_bytes,
    );
    todo!("Auto → Bfloat16 when bytes <= 0.70 * available, else Uniform4Bit; others pass through")
}

/// Sums shard bytes with FP8 tensors counted twice.
pub fn estimate_bfloat16_weight_bytes(source_directory: &Path) -> Result<u64, QuantizationError> {
    let _ = source_directory;
    todo!("file sizes plus the data_offsets span of every F8_* header entry")
}

/// Builds the allocation constraints from a request.
#[must_use]
pub fn allocation_constraints_for(
    request: &QuantizationPipelineRequest,
    candidate_bit_widths: CandidateBitWidths,
) -> AllocationConstraints {
    AllocationConstraints {
        target_bits_per_weight: request.target_bits_per_weight,
        candidate_bit_widths: Some(candidate_bit_widths),
        group_size: request.group_size,
        protect_boundary_blocks: true,
        boundary_block_count: 1,
        floor_components_per_block: request.floor_components_per_block,
        maximum_low_bit_block_run: 3,
    }
}

/// The range-search policy from a request.
#[must_use]
pub const fn range_search_policy_for(request: &QuantizationPipelineRequest) -> RangeSearchPolicy {
    RangeSearchPolicy {
        enabled: request.range_search_policy_enabled,
        maximum_bits: request.range_search_maximum_bits,
    }
}

/// The plan-only path: allocate from an existing sensitivity document and
/// report, without touching any tensor. This is what the CLI's
/// `quantize plan` sub-verb runs on any host.
pub fn plan_from_sensitivity_table(
    table: &SensitivityTable,
    constraints: &AllocationConstraints,
) -> Result<AllocationResult, QuantizationError> {
    crate::allocation::optimize_mixed_precision(table, constraints)
}

/// The full GPU path.
///
/// Work to port (OptiQ's `run_llm_pipeline`), stage by stage:
/// 1. Refuse an existing output directory unless `overwrite_output`.
/// 2. Inventory the source.
/// 3. `StaticStructural`: score the inventory, allocate, requantize with the
///    allocation's predicate, write the artifact files, done. No baselines.
/// 4. `ExactKlSensitivity`: resolve the reference mode from the bf16
///    estimate and the injected available memory. For `Uniform4Bit`, build
///    (or reuse, when present) `_uniform_4bit_baseline/` under the output
///    directory with the requantizer at uniform 4 bits, and load the probe
///    model from it with the bf16 source directory for streaming; for
///    `Bfloat16`, load the probe model from the source. Load the calibration
///    mix with `max(2 * sequence_count, 16)` sequences, then take the first
///    `sequence_count` for the sweep. Run the sweep with a checkpoint at
///    `<output>/sensitivity_checkpoint.json`. Drop the probe model and
///    delete the baseline directory.
/// 5. Allocate; print the summary and the overshoot note.
/// 6. Requantize the source with the predicate into `<output>/optiq_mixed/`
///    (or `static_mixed/`). Every tensor passes through the requantizer, so
///    multi-token-prediction heads and vision towers ride along in place;
///    the sidecar-reattachment step OptiQ needs after mlx-lm's convert does
///    not exist here.
/// 7. Write `config.json` with both quantization documents, the metadata
///    reconciled against what was written, and the sensitivity document
///    (never fatal).
/// 8. Copy the tokenizer and template files.
///
/// Peak memory for the sweep is one model plus one tensor's float32 working
/// copy; for the requantizer, one tensor plus its working copy.
#[cfg(feature = "direct-mlx")]
pub fn run_quantization_pipeline<ProbeModelLoader>(
    request: &QuantizationPipelineRequest,
    available_memory_bytes: u64,
    probe_model_loader: ProbeModelLoader,
    progress_reporter: &mut dyn crate::progress::ProgressReporter,
) -> Result<QuantizationPipelineOutcome, QuantizationError>
where
    ProbeModelLoader:
        FnOnce(&Path, Option<&Path>) -> Result<Box<dyn ProbeModelHandle>, QuantizationError>,
{
    let _ = (
        request,
        available_memory_bytes,
        probe_model_loader,
        progress_reporter,
    );
    todo!("sequence the stages documented above")
}

/// Type-erased probe model so the pipeline can be generic over engines
/// without a generic parameter leaking into the CLI.
pub trait ProbeModelHandle {
    fn run_sweep(
        &mut self,
        request: crate::sensitivity::exact_sweep::ExactSweepRequest<'_>,
    ) -> Result<SensitivityTable, QuantizationError>;
}
