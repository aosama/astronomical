//! The stage sequence of [`super::run_quantization_pipeline`], OptiQ's
//! `run_llm_pipeline`: inventory, score or sweep, allocate, plan,
//! requantize, write the documents, copy the tokenizer.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use astronomical_runtime_integration::MlxRuntime;

use crate::allocation::{AllocationResult, QuantizationPredicate, optimize_mixed_precision};
use crate::artifact::QuantizationConfigDocument;
use crate::bit_width::{BitWidth, CandidateBitWidths};
use crate::calibration::{
    CalibrationMix, HuggingFaceCalibrationTokenizer, mix::DEFAULT_SHUFFLE_SEED,
};
use crate::error::QuantizationError;
use crate::module_inventory::ModuleInventory;
use crate::progress::ProgressReporter;
use crate::requantize::driver::run_requantize;
use crate::requantize::{
    BitPolicy, DEFAULT_COPY_PATTERNS, DEFAULT_SHARD_BYTES, RequantizeOutcome, RequantizeTensorPlan,
    policy_from_predicate,
};
use crate::sensitivity::exact_sweep::ExactSweepRequest;
use crate::sensitivity::static_scorer::score_static_structural;
use crate::sensitivity::{CalibrationForwardSample, SensitivityMethod, SensitivityTable};

use super::artifact_files::{
    ArtifactDocumentsRequest, copy_source_files, write_artifact_documents,
};
use super::{
    PipelineMethod, ProbeModelLoader, QuantizationPipelineOutcome, QuantizationPipelineRequest,
    ReferenceMode, allocation_constraints_for, estimate_bfloat16_weight_bytes,
    range_search_policy_for, resolve_reference_mode,
};

/// Where the uniform 4-bit reference is built for a model too large to hold
/// at bf16; reused when a previous run left a complete one.
const UNIFORM_BASELINE_DIRECTORY_NAME: &str = "_uniform_4bit_baseline";

/// The sweep checkpoint, beside the artifact so a rerun resumes it.
const SENSITIVITY_CHECKPOINT_FILE_NAME: &str = "sensitivity_checkpoint.json";

/// OptiQ draws twice the requested calibration sequences (at least 16) and
/// sweeps the first ones, so the shuffle sees a wide offset pool even for a
/// short sweep.
const MINIMUM_CALIBRATION_DRAW: usize = 16;

/// Runs every stage; see [`super::run_quantization_pipeline`].
pub fn run_conversion(
    runtime: &MlxRuntime,
    request: &QuantizationPipelineRequest,
    available_memory_bytes: u64,
    probe_model_loader: &mut dyn ProbeModelLoader,
    progress_reporter: &mut dyn ProgressReporter,
) -> Result<QuantizationPipelineOutcome, QuantizationError> {
    let candidate_bit_widths = CandidateBitWidths::new(&request.candidate_bits)?;
    let artifact_directory = request
        .output_directory
        .join(artifact_directory_name(request.method));
    prepare_artifact_directory(request, &artifact_directory)?;
    let inventory =
        ModuleInventory::from_source_directory(&request.source_directory, request.group_size)?;
    let sensitivity_table = match request.method {
        PipelineMethod::StaticStructural => {
            score_static_structural(&inventory, &candidate_bit_widths)
        }
        PipelineMethod::ExactKlSensitivity => measure_exact_sensitivity(
            runtime,
            request,
            &candidate_bit_widths,
            available_memory_bytes,
            probe_model_loader,
            progress_reporter,
        )?,
    };
    let lowest_bits = candidate_bit_widths.lowest().bits();
    let constraints = allocation_constraints_for(request, candidate_bit_widths);
    let allocation = optimize_mixed_precision(&sensitivity_table, &constraints)?;
    log_allocation_summary(&allocation, request.floor_components_per_block);
    let requantize_outcome = requantize_with_allocation(
        runtime,
        request,
        &inventory,
        &allocation,
        &artifact_directory,
        progress_reporter,
    )?;
    finish_artifact(
        request,
        &artifact_directory,
        sensitivity_table,
        allocation,
        lowest_bits,
        requantize_outcome,
    )
}

/// `optiq_mixed` for a measured conversion, `static_mixed` for a structural
/// one, OptiQ's directory names.
#[must_use]
pub const fn artifact_directory_name(method: PipelineMethod) -> &'static str {
    match method {
        PipelineMethod::ExactKlSensitivity => "optiq_mixed",
        PipelineMethod::StaticStructural => "static_mixed",
    }
}

/// Refuses an existing artifact unless overwrite was requested. Only the
/// artifact directory is guarded: the output directory also holds the sweep
/// checkpoint and the baseline, which a rerun must find to resume.
fn prepare_artifact_directory(
    request: &QuantizationPipelineRequest,
    artifact_directory: &Path,
) -> Result<(), QuantizationError> {
    if artifact_directory.exists() {
        if !request.overwrite_output {
            return Err(QuantizationError::OutputDirectoryExists {
                output_directory: artifact_directory.to_path_buf(),
            });
        }
        if let Err(remove_error) = std::fs::remove_dir_all(artifact_directory) {
            return Err(artifact_write_failure(artifact_directory, &remove_error));
        }
    }
    match std::fs::create_dir_all(artifact_directory) {
        Ok(()) => Ok(()),
        Err(create_error) => Err(artifact_write_failure(artifact_directory, &create_error)),
    }
}

/// The exact sweep: resolve the reference, build or reuse the uniform
/// baseline when bf16 does not fit, load the calibration sequences and the
/// probe model, sweep with a checkpoint, then release the model and the
/// baseline.
fn measure_exact_sensitivity(
    runtime: &MlxRuntime,
    request: &QuantizationPipelineRequest,
    candidate_bit_widths: &CandidateBitWidths,
    available_memory_bytes: u64,
    probe_model_loader: &mut dyn ProbeModelLoader,
    progress_reporter: &mut dyn ProgressReporter,
) -> Result<SensitivityTable, QuantizationError> {
    let estimated_bytes = estimate_bfloat16_weight_bytes(&request.source_directory)?;
    let reference_mode = resolve_reference_mode(
        request.reference_mode,
        estimated_bytes,
        available_memory_bytes,
    );
    let calibration_samples = load_calibration_samples(request, progress_reporter)?;
    let is_uniform_reference = reference_mode == ReferenceMode::Uniform4Bit;
    let (probe_directory, streaming_source_directory, method) = if is_uniform_reference {
        let baseline_directory = ensure_uniform_baseline(runtime, request, progress_reporter)?;
        let streaming_source = Some(request.source_directory.as_path());
        (
            baseline_directory,
            streaming_source,
            SensitivityMethod::ExactUniform4BitReference,
        )
    } else {
        let probe_directory = request.source_directory.clone();
        (
            probe_directory,
            None,
            SensitivityMethod::ExactBfloat16Reference,
        )
    };
    let mut probe_model =
        probe_model_loader.load_probe_model(&probe_directory, streaming_source_directory)?;
    let checkpoint_path = request
        .output_directory
        .join(SENSITIVITY_CHECKPOINT_FILE_NAME);
    let sensitivity_table = probe_model.run_sweep(ExactSweepRequest {
        calibration_samples: &calibration_samples,
        candidate_bit_widths,
        group_size: request.group_size,
        method,
        checkpoint_path: Some(&checkpoint_path),
        progress_reporter,
    })?;
    drop(probe_model);
    if is_uniform_reference && let Err(remove_error) = std::fs::remove_dir_all(&probe_directory) {
        tracing::warn!(%remove_error, baseline = %probe_directory.display(), "the uniform baseline was not removed");
    }
    Ok(sensitivity_table)
}

/// Tokenizes the mix with the source's own tokenizer and keeps the first
/// `calibration_sequence_count` of a wider draw.
fn load_calibration_samples(
    request: &QuantizationPipelineRequest,
    progress_reporter: &mut dyn ProgressReporter,
) -> Result<Vec<CalibrationForwardSample>, QuantizationError> {
    let tokenizer = HuggingFaceCalibrationTokenizer::from_tokenizer_file(
        &request.source_directory.join("tokenizer.json"),
    )?;
    let drawn_sequence_count =
        (2 * request.calibration_sequence_count).max(MINIMUM_CALIBRATION_DRAW);
    let calibration_mix = match &request.calibration_mix_path {
        Some(mix_path) => CalibrationMix::from_jsonl_file(
            mix_path,
            &tokenizer,
            drawn_sequence_count,
            request.calibration_sequence_length_tokens,
            DEFAULT_SHUFFLE_SEED,
            progress_reporter,
        )?,
        None => CalibrationMix::bundled(
            &tokenizer,
            drawn_sequence_count,
            request.calibration_sequence_length_tokens,
            DEFAULT_SHUFFLE_SEED,
            progress_reporter,
        )?,
    };
    let swept_sequence_count = request
        .calibration_sequence_count
        .min(calibration_mix.samples().len());
    Ok(calibration_mix.samples()[..swept_sequence_count].to_vec())
}

/// Builds `_uniform_4bit_baseline/` at 4 bits everywhere quantizable, or
/// reuses a complete one; a partial one is rebuilt.
fn ensure_uniform_baseline(
    runtime: &MlxRuntime,
    request: &QuantizationPipelineRequest,
    progress_reporter: &mut dyn ProgressReporter,
) -> Result<PathBuf, QuantizationError> {
    let baseline_directory = request
        .output_directory
        .join(UNIFORM_BASELINE_DIRECTORY_NAME);
    let is_complete = baseline_directory.join("config.json").is_file()
        && baseline_directory
            .join("model.safetensors.index.json")
            .is_file();
    if is_complete {
        return Ok(baseline_directory);
    }
    if baseline_directory.exists()
        && let Err(remove_error) = std::fs::remove_dir_all(&baseline_directory)
    {
        return Err(artifact_write_failure(&baseline_directory, &remove_error));
    }
    let inventory =
        ModuleInventory::from_source_directory(&request.source_directory, request.group_size)?;
    let uniform_policy = BitPolicy::Uniform {
        bit_width: BitWidth::new(4)?,
        group_size: request.group_size,
    };
    let plan = RequantizeTensorPlan::from_inventory(&inventory, &uniform_policy);
    let outcome = run_requantize(
        runtime,
        &request.source_directory,
        &baseline_directory,
        &plan,
        range_search_policy_for(request),
        DEFAULT_SHARD_BYTES,
        progress_reporter,
    )?;
    QuantizationConfigDocument::from_written_modules(
        request.group_size.elements(),
        4,
        &outcome.written_quantization_by_module,
    )
    .write_into_config(
        &request.source_directory.join("config.json"),
        &baseline_directory.join("config.json"),
    )?;
    copy_source_files(
        &request.source_directory,
        &baseline_directory,
        DEFAULT_COPY_PATTERNS,
    )?;
    Ok(baseline_directory)
}

/// Plans and runs the requantizer with the allocation's predicate.
fn requantize_with_allocation(
    runtime: &MlxRuntime,
    request: &QuantizationPipelineRequest,
    inventory: &ModuleInventory,
    allocation: &AllocationResult,
    artifact_directory: &Path,
    progress_reporter: &mut dyn ProgressReporter,
) -> Result<RequantizeOutcome, QuantizationError> {
    let predicate = QuantizationPredicate::from_allocation(allocation, None);
    let plan = RequantizeTensorPlan::from_inventory(inventory, &policy_from_predicate(predicate));
    tracing::info!(
        tensor_count = plan.tensor_count(),
        quantized_tensor_count = plan.quantized_tensor_count(),
        estimated_output_bytes = plan.estimated_output_bytes(),
        "requantization plan ready"
    );
    run_requantize(
        runtime,
        &request.source_directory,
        artifact_directory,
        &plan,
        range_search_policy_for(request),
        DEFAULT_SHARD_BYTES,
        progress_reporter,
    )
}

/// Writes the documents, copies the tokenizer, and summarizes.
fn finish_artifact(
    request: &QuantizationPipelineRequest,
    artifact_directory: &Path,
    sensitivity_table: SensitivityTable,
    allocation: AllocationResult,
    default_bits: u32,
    requantize_outcome: RequantizeOutcome,
) -> Result<QuantizationPipelineOutcome, QuantizationError> {
    let written_quantization_by_module: &BTreeMap<String, (u32, u32)> =
        &requantize_outcome.written_quantization_by_module;
    let sensitivity_document_written = write_artifact_documents(&ArtifactDocumentsRequest {
        source_directory: &request.source_directory,
        artifact_directory,
        base_model_name: &request.base_model_name,
        method: sensitivity_table.method,
        allocation: &allocation,
        sensitivity_table: &sensitivity_table,
        candidate_bits: &request.candidate_bits,
        group_size: request.group_size,
        default_bits,
        written_quantization_by_module,
    })?;
    copy_source_files(
        &request.source_directory,
        artifact_directory,
        DEFAULT_COPY_PATTERNS,
    )?;
    Ok(QuantizationPipelineOutcome {
        artifact_directory: artifact_directory.to_path_buf(),
        method: sensitivity_table.method,
        allocation,
        shard_count: requantize_outcome.shard_count,
        total_output_bytes: requantize_outcome.total_output_bytes,
        sensitivity_document_written,
    })
}

/// The allocation's headline figures and, when the floor or the run guard
/// pushed past the target, OptiQ's overshoot note.
fn log_allocation_summary(allocation: &AllocationResult, floor_components_per_block: usize) {
    tracing::info!(
        target_bits_per_weight = allocation.target_bits_per_weight,
        achieved_bits_per_weight = allocation.achieved_bits_per_weight,
        module_count = allocation.decisions.len(),
        estimated_weight_bytes = allocation.estimated_weight_bytes,
        "allocation complete"
    );
    if let Some(overshoot_note) = allocation.overshoot_note(floor_components_per_block) {
        tracing::warn!("{overshoot_note}");
    }
}

fn artifact_write_failure(directory: &Path, failure: &std::io::Error) -> QuantizationError {
    QuantizationError::ArtifactWriteFailure {
        output_directory: directory.to_path_buf(),
        description: failure.to_string(),
    }
}
