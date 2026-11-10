//! `quantize evaluate`: how far a finished quant's next-token distribution
//! drifts from a reference model's (feature `quantize-convert`).
//!
//! Both models run through the engine that serves them, one after the
//! other, over the bundled calibration mix as flat text. The reference is
//! named explicitly: this verb never reaches the network to guess one.

use std::io::Write;
use std::path::{Path, PathBuf};

use astronomical_model_serving::machine_ceiling_from_physical_ram_bytes;
use astronomical_model_serving::quantization_probe::Qwen3_5KlDivergenceEngine;
use astronomical_quantization::calibration::HuggingFaceCalibrationTokenizer;
use astronomical_quantization::error::QuantizationError;
use astronomical_quantization::evaluation::{
    KlEvaluationRequest, KlEvaluationSummary, bundled_evaluation_prompts, evaluate_kl_divergence,
    tokenized_evaluation_prompts,
};
use astronomical_quantization::pipeline::probe_source_view::loadable_model_directory;
use astronomical_runtime_integration::MlxRuntime;

use super::quantize_convert::{memory_limits_for, sample_physical_ram_bytes};
use super::{
    QuantizeDependencies, QuantizeError, QuantizeEvaluateArguments, StandardErrorProgressReporter,
    write_json_line,
};

/// The repository's reproducibility rule for prompt processing.
const EVALUATION_PREFILL_CHUNK_TOKENS: usize = 2_048;

/// Tokenizes the prompts with the artifact's own tokenizer, forwards the
/// reference and then the artifact, and prints the three figures.
pub fn run_quantize_evaluate(
    evaluate_arguments: &QuantizeEvaluateArguments,
    dependencies: &mut QuantizeDependencies<'_>,
) -> Result<(), QuantizeError> {
    let tokenizer = HuggingFaceCalibrationTokenizer::from_model_directory(
        &evaluate_arguments.artifact_directory,
    )?;
    let prompts = tokenized_evaluation_prompts(
        &bundled_evaluation_prompts(evaluate_arguments.prompt_count)?,
        &tokenizer,
        evaluate_arguments.sequence_length_tokens,
    )?;
    let machine_ceiling_bytes =
        machine_ceiling_from_physical_ram_bytes(sample_physical_ram_bytes()?);
    let memory_limits = memory_limits_for(machine_ceiling_bytes)?;
    // Held for the whole evaluation, as `convert` holds its own: the runtime
    // is process-global, and the engine's model loads join it under these
    // same limits rather than racing to set different ones.
    let _evaluation_runtime = match MlxRuntime::initialize(memory_limits) {
        Ok(evaluation_runtime) => evaluation_runtime,
        Err(runtime_error) => return Err(QuantizationError::from(runtime_error).into()),
    };
    let view_parent = evaluation_scratch_directory();
    let (reference_directory, built_reference_view) =
        loadable_model_directory(&evaluate_arguments.reference_directory, &view_parent)?;
    let mut engine = Qwen3_5KlDivergenceEngine {
        memory_limits,
        prefill_chunk_tokens: EVALUATION_PREFILL_CHUNK_TOKENS,
    };
    let mut progress_reporter = StandardErrorProgressReporter {
        stderr: dependencies.stderr,
    };
    let evaluation_outcome = evaluate_kl_divergence(
        &mut engine,
        &KlEvaluationRequest {
            reference_directory: &reference_directory,
            candidate_directory: &evaluate_arguments.artifact_directory,
            prompts: &prompts,
        },
        &mut progress_reporter,
    );
    if built_reference_view {
        let _ = std::fs::remove_dir_all(&view_parent);
    }
    let summary = evaluation_outcome?;
    let render_outcome = if evaluate_arguments.render_json {
        write_json_line(&summary, dependencies.stdout)
    } else {
        write_summary(&summary, evaluate_arguments, dependencies.stdout)
    };
    render_outcome.map_err(QuantizeError::stdout_unwritable)
}

/// Where a view of a reference without a shard index, or stored as
/// symbolic links, is built for the engine to load.
fn evaluation_scratch_directory() -> PathBuf {
    std::env::temp_dir().join(format!(
        "astronomical-quantize-evaluate-{}",
        std::process::id()
    ))
}

fn write_summary(
    summary: &KlEvaluationSummary,
    evaluate_arguments: &QuantizeEvaluateArguments,
    output: &mut dyn Write,
) -> std::io::Result<()> {
    writeln!(
        output,
        "KL(reference || candidate)  mean={:.4}  median={:.4}  p95={:.4}",
        summary.mean_kl_divergence,
        summary.median_kl_divergence,
        summary.ninety_fifth_percentile_kl_divergence
    )?;
    writeln!(
        output,
        "Reference: {}",
        display_of(&evaluate_arguments.reference_directory)
    )?;
    writeln!(
        output,
        "Candidate: {}",
        display_of(&evaluate_arguments.artifact_directory)
    )?;
    writeln!(
        output,
        "Prompts: {} ({} tokens measured, at most {} per prompt)",
        summary.prompt_count, summary.token_count, evaluate_arguments.sequence_length_tokens
    )
}

fn display_of(directory: &Path) -> String {
    directory.display().to_string()
}
