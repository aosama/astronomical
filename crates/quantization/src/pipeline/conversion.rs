//! The stage sequence of [`super::run_quantization_pipeline`], OptiQ's
//! `run_llm_pipeline`: inventory, score or sweep, allocate, plan,
//! requantize, write the documents, copy the tokenizer.
//!
//! Ported from OptiQ's `run_llm_pipeline` (`optiq/models/llm.py`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use astronomical_runtime_integration::MlxRuntime;

use crate::allocation::{AllocationResult, QuantizationPredicate, optimize_mixed_precision};
use crate::artifact::QuantizationConfigDocument;
use crate::artifact::converted_config::normalize_converted_config;
use crate::artifact::generation_config::write_recommended_generation_config;
use crate::artifact::model_card::write_model_card;
use crate::bit_width::{BitWidth, CandidateBitWidths};
use crate::calibration::{
    CalibrationMix, HuggingFaceCalibrationTokenizer, mix::DEFAULT_SHUFFLE_SEED,
};
use crate::error::QuantizationError;
use crate::module_inventory::ModuleInventory;
use crate::progress::ProgressReporter;
use crate::requantize::driver::{VISION_SIDECAR_FILE_NAME, run_requantize};
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
use super::probe_source_view::loadable_model_directory;
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
        inventory.declares_multimodal_towers(),
        requantize_outcome,
    )
}

/// The `bits` default OptiQ records in the quantization document: its
/// pipeline passes `q_bits = 4` whatever the candidate ladder is. Every
/// quantized module carries its own entry, so the default only describes
/// modules a loader finds packed without one.
const QUANTIZATION_DOCUMENT_DEFAULT_BITS: u32 = 4;

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
///
/// An empty artifact directory is not an artifact. This function creates the
/// directory before the sweep so an unwritable output fails in seconds
/// rather than after hours of measurement, which means a sweep interrupted
/// before the requantizer started leaves an empty one behind; refusing it
/// would make every resume demand an overwrite it has nothing to overwrite.
fn prepare_artifact_directory(
    request: &QuantizationPipelineRequest,
    artifact_directory: &Path,
) -> Result<(), QuantizationError> {
    if holds_artifact_files(artifact_directory)? {
        remove_artifact_when_overwriting(request, artifact_directory)?;
    }
    match std::fs::create_dir_all(artifact_directory) {
        Ok(()) => Ok(()),
        Err(create_error) => Err(artifact_write_failure(artifact_directory, &create_error)),
    }
}

/// Removes an existing artifact, or refuses when overwrite was not requested.
fn remove_artifact_when_overwriting(
    request: &QuantizationPipelineRequest,
    artifact_directory: &Path,
) -> Result<(), QuantizationError> {
    if !request.overwrite_output {
        return Err(QuantizationError::OutputDirectoryExists {
            output_directory: artifact_directory.to_path_buf(),
        });
    }
    match std::fs::remove_dir_all(artifact_directory) {
        Ok(()) => Ok(()),
        Err(remove_error) => Err(artifact_write_failure(artifact_directory, &remove_error)),
    }
}

/// Whether the path exists and holds at least one entry. A path that exists
/// but is not a listable directory counts as holding files, so a stray file
/// of the artifact's name is refused rather than silently replaced.
fn holds_artifact_files(artifact_directory: &Path) -> Result<bool, QuantizationError> {
    if !artifact_directory.exists() {
        return Ok(false);
    }
    if !artifact_directory.is_dir() {
        return Ok(true);
    }
    match std::fs::read_dir(artifact_directory) {
        Ok(mut directory_entries) => Ok(directory_entries.next().is_some()),
        Err(read_error) => Err(artifact_write_failure(artifact_directory, &read_error)),
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
        let probe_directory = bfloat16_probe_directory(runtime, request, progress_reporter)?;
        (
            probe_directory,
            None,
            SensitivityMethod::ExactBfloat16Reference,
        )
    };
    let checkpoint_path = request
        .output_directory
        .join(SENSITIVITY_CHECKPOINT_FILE_NAME);
    let sweep_outcome = sweep_with_probe(
        probe_model_loader,
        &probe_directory,
        streaming_source_directory,
        ExactSweepRequest {
            calibration_samples: &calibration_samples,
            candidate_bit_widths,
            group_size: request.group_size,
            method,
            checkpoint_path: Some(&checkpoint_path),
            progress_reporter,
        },
    );
    let keeps_baseline_for_a_rerun = is_uniform_reference && sweep_outcome.is_err();
    remove_scratch_probe_directory(request, &probe_directory, keeps_baseline_for_a_rerun);
    sweep_outcome
}

/// The probe directory is scratch whenever it is not the source itself.
/// The indexed view and the bfloat16 copy are rebuilt by every run, so they
/// go whether or not the sweep succeeded. The uniform baseline is the
/// expensive one to build and a rerun resumes from it, so the caller keeps
/// it after a failure and it goes only once the sweep is complete.
fn remove_scratch_probe_directory(
    request: &QuantizationPipelineRequest,
    probe_directory: &Path,
    keeps_baseline_for_a_rerun: bool,
) {
    let is_scratch_probe_directory = probe_directory != request.source_directory;
    if is_scratch_probe_directory
        && !keeps_baseline_for_a_rerun
        && let Err(remove_error) = std::fs::remove_dir_all(probe_directory)
    {
        tracing::warn!(%remove_error, probe_directory = %probe_directory.display(), "the probe's scratch directory was not removed");
    }
}

/// Loads the probe model and sweeps it; the model is released on return.
fn sweep_with_probe(
    probe_model_loader: &mut dyn ProbeModelLoader,
    probe_directory: &Path,
    streaming_source_directory: Option<&Path>,
    sweep_request: ExactSweepRequest<'_>,
) -> Result<SensitivityTable, QuantizationError> {
    let mut probe_model =
        probe_model_loader.load_probe_model(probe_directory, streaming_source_directory)?;
    probe_model.run_sweep(sweep_request)
}

/// The directory the bf16 reference probe loads: the source itself (through
/// an indexed view when it has no shard index), or, when the source stores
/// modules in a layout only a converter understands (fused or per-expert
/// routed experts in an original release), a bfloat16 copy written in the
/// output layout. The copy costs the model's size on disk for the length of
/// the sweep; a bf16 reference already needs that much memory, and a model
/// too large for either takes the uniform reference instead.
fn bfloat16_probe_directory(
    runtime: &MlxRuntime,
    request: &QuantizationPipelineRequest,
    progress_reporter: &mut dyn ProgressReporter,
) -> Result<PathBuf, QuantizationError> {
    let inventory =
        ModuleInventory::from_source_directory(&request.source_directory, request.group_size)?;
    if !inventory.has_assembled_modules() {
        let (probe_directory, _) =
            loadable_model_directory(&request.source_directory, &request.output_directory)?;
        return Ok(probe_directory);
    }
    let copy_directory = request
        .output_directory
        .join(BFLOAT16_PROBE_COPY_DIRECTORY_NAME);
    if copy_directory.exists()
        && let Err(remove_error) = std::fs::remove_dir_all(&copy_directory)
    {
        return Err(artifact_write_failure(&copy_directory, &remove_error));
    }
    let plan = RequantizeTensorPlan::from_inventory(&inventory, &BitPolicy::KeepEverythingBfloat16);
    run_requantize(
        runtime,
        &request.source_directory,
        &copy_directory,
        &plan,
        range_search_policy_for(request),
        DEFAULT_SHARD_BYTES,
        progress_reporter,
    )?;
    let copied_config_path = copy_directory.join("config.json");
    if let Err(copy_error) = std::fs::copy(
        request.source_directory.join("config.json"),
        &copied_config_path,
    ) {
        return Err(artifact_write_failure(&copy_directory, &copy_error));
    }
    normalize_converted_config(&copied_config_path)?;
    copy_source_files(
        &request.source_directory,
        &copy_directory,
        DEFAULT_COPY_PATTERNS,
    )?;
    Ok(copy_directory)
}

/// Where the bfloat16 copy for the probe is written; removed after the sweep.
const BFLOAT16_PROBE_COPY_DIRECTORY_NAME: &str = "_bfloat16_probe_source";

/// Tokenizes the mix with the source's own tokenizer and keeps the first
/// `calibration_sequence_count` of a wider draw.
fn load_calibration_samples(
    request: &QuantizationPipelineRequest,
    progress_reporter: &mut dyn ProgressReporter,
) -> Result<Vec<CalibrationForwardSample>, QuantizationError> {
    let tokenizer =
        HuggingFaceCalibrationTokenizer::from_model_directory(&request.source_directory)?;
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
        QUANTIZATION_DOCUMENT_DEFAULT_BITS,
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
    declares_multimodal_towers: bool,
    requantize_outcome: RequantizeOutcome,
) -> Result<QuantizationPipelineOutcome, QuantizationError> {
    let vision_sidecar = written_vision_sidecar(&requantize_outcome);
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
        default_bits: QUANTIZATION_DOCUMENT_DEFAULT_BITS,
        vision_sidecar,
        declares_multimodal_towers,
        mtp_sidecar: requantize_outcome.mtp_sidecar.as_ref(),
        written_quantization_by_module,
    })?;
    write_companion_files(request, artifact_directory)?;
    Ok(QuantizationPipelineOutcome {
        artifact_directory: artifact_directory.to_path_buf(),
        method: sensitivity_table.method,
        allocation,
        shard_count: requantize_outcome.shard_count,
        total_output_bytes: requantize_outcome.total_output_bytes,
        sensitivity_document_written,
    })
}

/// The tower sidecar's file name and tensor count, when one was written.
fn written_vision_sidecar(requantize_outcome: &RequantizeOutcome) -> Option<(&'static str, usize)> {
    let tensor_count = requantize_outcome.vision_sidecar_tensor_count;
    (tensor_count > 0).then_some((VISION_SIDECAR_FILE_NAME, tensor_count))
}

/// The files beside the weights and documents: the tokenizer, template and
/// processor files copied from the source, the family's recommended
/// sampling, and the model card.
fn write_companion_files(
    request: &QuantizationPipelineRequest,
    artifact_directory: &Path,
) -> Result<(), QuantizationError> {
    copy_source_files(
        &request.source_directory,
        artifact_directory,
        DEFAULT_COPY_PATTERNS,
    )?;
    write_recommended_generation_config(artifact_directory, &request.base_model_name)?;
    write_model_card(artifact_directory)?;
    Ok(())
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
