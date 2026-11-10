//! `quantize convert`: the full pipeline in-process (feature
//! `quantize-convert`). The memory ceiling comes from the machine's physical
//! RAM through the model-serving memory policy, the same single source of
//! truth the worker uses, never a hardwired figure.

use std::io::Write;
use std::path::Path;
use std::process::Command;

use astronomical_model_serving::machine_ceiling_from_physical_ram_bytes;
use astronomical_model_serving::quantization_probe::load_qwen3_5_probe_model;
use astronomical_quantization::bit_width::GroupSize;
use astronomical_quantization::error::QuantizationError;
use astronomical_quantization::pipeline::{
    PipelineMethod, ProbeModelHandle, ProbeModelLoader, ProbeModelSweeper,
    QuantizationPipelineOutcome, QuantizationPipelineRequest, ReferenceMode,
    run_quantization_pipeline,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use super::{
    QuantizeConvertArguments, QuantizeDependencies, QuantizeError, QuantizeMethodArgument,
    QuantizeReferenceArgument, StandardErrorProgressReporter,
};

const PHYSICAL_MEMORY_SIZE_SYSCTL_KEY: &str = "hw.memsize";
const SYSCTL_EXECUTABLE_PATH: &str = "/usr/sbin/sysctl";

/// The repository's reproducibility rule: probes prefill in 2,048-token
/// chunks so a sweep measures the same graph a served request runs.
const PROBE_PREFILL_CHUNK_TOKENS: usize = 2_048;

/// OptiQ's `run_llm_pipeline` cuts 512-token calibration sequences; keeping
/// the length (with the numpy-exact shuffle) means a Rust sweep calibrates
/// on the same sequences as a Python sweep of the same source.
const CALIBRATION_SEQUENCE_LENGTH_TOKENS: usize = 512;

/// Builds the request, runs the pipeline with progress on standard error,
/// and prints the summary on standard output.
pub fn run_quantize_convert(
    convert_arguments: &QuantizeConvertArguments,
    dependencies: &mut QuantizeDependencies<'_>,
) -> Result<(), QuantizeError> {
    let request = pipeline_request_from(convert_arguments)?;
    let machine_ceiling_bytes =
        machine_ceiling_from_physical_ram_bytes(sample_physical_ram_bytes()?);
    let memory_limits = memory_limits_for(machine_ceiling_bytes)?;
    let runtime = match MlxRuntime::initialize(memory_limits) {
        Ok(runtime) => runtime,
        Err(runtime_error) => return Err(QuantizationError::from(runtime_error).into()),
    };
    let mut probe_model_loader = Qwen3_5ProbeModelLoader { memory_limits };
    let mut progress_reporter = StandardErrorProgressReporter {
        stderr: dependencies.stderr,
    };
    let outcome = run_quantization_pipeline(
        &runtime,
        &request,
        machine_ceiling_bytes,
        &mut probe_model_loader,
        &mut progress_reporter,
    )?;
    write_summary(&outcome, dependencies.stdout).map_err(QuantizeError::stdout_unwritable)
}

/// Loads the resident Qwen3.5 probe under the same memory limits as the
/// CLI's own runtime; the runtime is process-global, so a second
/// initialization must agree with the first.
struct Qwen3_5ProbeModelLoader {
    memory_limits: MlxMemoryLimits,
}

impl ProbeModelLoader for Qwen3_5ProbeModelLoader {
    fn load_probe_model(
        &mut self,
        probe_directory: &Path,
        streaming_source_directory: Option<&Path>,
    ) -> Result<Box<dyn ProbeModelHandle>, QuantizationError> {
        let probe_model = load_qwen3_5_probe_model(
            probe_directory,
            streaming_source_directory,
            self.memory_limits,
            PROBE_PREFILL_CHUNK_TOKENS,
        )?;
        Ok(Box::new(ProbeModelSweeper(probe_model)))
    }
}

/// The arguments as a pipeline request; the base model name defaults to
/// the source directory's own name.
fn pipeline_request_from(
    convert_arguments: &QuantizeConvertArguments,
) -> Result<QuantizationPipelineRequest, QuantizeError> {
    let base_model_name = match &convert_arguments.base_model_name {
        Some(base_model_name) => base_model_name.clone(),
        None => convert_arguments
            .source_directory
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
    };
    Ok(QuantizationPipelineRequest {
        source_directory: convert_arguments.source_directory.clone(),
        output_directory: convert_arguments.output_directory.clone(),
        base_model_name,
        method: match convert_arguments.method {
            QuantizeMethodArgument::Exact => PipelineMethod::ExactKlSensitivity,
            QuantizeMethodArgument::Static => PipelineMethod::StaticStructural,
        },
        reference_mode: match convert_arguments.reference {
            QuantizeReferenceArgument::Auto => ReferenceMode::Auto,
            QuantizeReferenceArgument::Bfloat16 => ReferenceMode::Bfloat16,
            QuantizeReferenceArgument::Uniform4Bit => ReferenceMode::Uniform4Bit,
        },
        target_bits_per_weight: convert_arguments.target_bits_per_weight,
        candidate_bits: convert_arguments.candidate_bits.clone(),
        group_size: GroupSize::new(convert_arguments.group_size)?,
        calibration_sequence_count: convert_arguments.calibration_sequence_count,
        calibration_sequence_length_tokens: CALIBRATION_SEQUENCE_LENGTH_TOKENS,
        calibration_mix_path: convert_arguments.calibration_mix_path.clone(),
        floor_components_per_block: convert_arguments.floor_components_per_block,
        range_search_policy_enabled: convert_arguments.range_search_enabled,
        range_search_maximum_bits: convert_arguments.range_search_maximum_bits,
        overwrite_output: convert_arguments.overwrite_output,
    })
}

/// `hw.memsize`, the one input the machine ceiling is computed from.
pub(super) fn sample_physical_ram_bytes() -> Result<u64, QuantizeError> {
    let sysctl_output = match Command::new(SYSCTL_EXECUTABLE_PATH)
        .arg("-n")
        .arg(PHYSICAL_MEMORY_SIZE_SYSCTL_KEY)
        .output()
    {
        Ok(sysctl_output) if sysctl_output.status.success() => sysctl_output,
        Ok(sysctl_output) => {
            return Err(physical_memory_unreadable(
                &sysctl_output.status.to_string(),
            ));
        }
        Err(spawn_error) => return Err(physical_memory_unreadable(&spawn_error.to_string())),
    };
    let physical_ram_text = String::from_utf8_lossy(&sysctl_output.stdout);
    match physical_ram_text.trim().parse::<u64>() {
        Ok(physical_ram_bytes) => Ok(physical_ram_bytes),
        Err(_) => Err(physical_memory_unreadable(
            "hw.memsize is not an unsigned integer",
        )),
    }
}

/// The ceiling as both the active limit and the allocator-cache limit, the
/// conversion being the only MLX tenant of this process.
pub(super) fn memory_limits_for(
    machine_ceiling_bytes: u64,
) -> Result<MlxMemoryLimits, QuantizeError> {
    let ceiling_bytes = usize::try_from(machine_ceiling_bytes).unwrap_or(usize::MAX);
    match MlxMemoryLimits::new(ceiling_bytes, ceiling_bytes) {
        Ok(memory_limits) => Ok(memory_limits),
        Err(limits_error) => Err(QuantizationError::from(limits_error).into()),
    }
}

fn write_summary(
    outcome: &QuantizationPipelineOutcome,
    output: &mut dyn Write,
) -> std::io::Result<()> {
    writeln!(output, "Artifact: {}", outcome.artifact_directory.display())?;
    writeln!(output, "Method: {:?}", outcome.method)?;
    writeln!(
        output,
        "Target bits per weight: {:.2}",
        outcome.allocation.target_bits_per_weight
    )?;
    writeln!(
        output,
        "Achieved bits per weight: {:.2}",
        outcome.allocation.achieved_bits_per_weight
    )?;
    writeln!(output, "Shards: {}", outcome.shard_count)?;
    writeln!(output, "Bytes written: {}", outcome.total_output_bytes)?;
    writeln!(
        output,
        "Sensitivity document written: {}",
        outcome.sensitivity_document_written
    )
}

fn physical_memory_unreadable(description: &str) -> QuantizeError {
    QuantizeError::PhysicalMemoryUnreadable {
        description: description.to_owned(),
    }
}
