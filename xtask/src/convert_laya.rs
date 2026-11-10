//! `cargo xtask convert-laya --checkpoint DIRECTORY --output DIRECTORY`
//! converts a PyTorch Laya checkpoint, such as the original
//! `convaiinnovations/laya` or a fine-tune of it, into the MLX layout
//! Astronomical serves.
//!
//! A PyTorch checkpoint is a folder holding `model.safetensors`,
//! `rl_agent_config.json`, `encoder/config.json` and a `tokenizer` folder.
//! The output is a folder holding `model.safetensors` with the linear layers
//! of the encoder and the decision head quantized to 8 bits in groups of 32,
//! one `config.json` with the temperatures, and the tokenizer files.
//!
//! The conversion is plain arithmetic on one tensor at a time, so it needs
//! no graphics processor, no Python and no model runtime. It writes nothing
//! that names a place on the machine it ran on: the training record in the
//! source config is never copied.

mod affine_quantization;
mod half_float;
mod laya_config;
mod progress;
mod safetensors_file;
mod tensor_bytes;
mod tensor_plan;

use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use affine_quantization::quantize_matrix;
use progress::ConversionProgress;
use safetensors_file::{OutputCheckpoint, PlannedTensor, SourceCheckpoint, TensorRecord};
use tensor_bytes::{decoded_weights, half_bits_bytes, half_precision_bytes};
use tensor_plan::{TensorTreatment, planned_output_tensors, tensor_treatment};

const USAGE: &str = "usage: cargo xtask convert-laya --checkpoint DIRECTORY --output DIRECTORY";
/// The weights are written under this name and renamed once complete, so an
/// interrupted conversion never leaves a file that looks like a model.
const UNFINISHED_WEIGHTS_FILE_NAME: &str = "model.safetensors.unfinished";
const TOKENIZER_FILE_NAMES: [&str; 2] = ["tokenizer.json", "tokenizer_config.json"];

/// The two folders one conversion reads from and writes to.
#[derive(Debug, PartialEq)]
pub(crate) struct ConvertLayaInvocation {
    pub(crate) checkpoint_directory: PathBuf,
    pub(crate) output_directory: PathBuf,
}

impl ConvertLayaInvocation {
    pub(crate) fn parse(arguments: &[String]) -> Result<Self, String> {
        let mut checkpoint_directory = None;
        let mut output_directory = None;
        let mut remaining_arguments = arguments.iter();
        while let Some(argument) = remaining_arguments.next() {
            let directory_slot = match argument.as_str() {
                "--checkpoint" => &mut checkpoint_directory,
                "--output" => &mut output_directory,
                unknown_argument => {
                    return Err(format!("unknown argument {unknown_argument}\n{USAGE}"));
                }
            };
            let Some(directory_text) = remaining_arguments.next() else {
                return Err(format!("{argument} needs a directory\n{USAGE}"));
            };
            if directory_slot.is_some() {
                return Err(format!("{argument} was given twice\n{USAGE}"));
            }
            *directory_slot = Some(PathBuf::from(directory_text));
        }
        match (checkpoint_directory, output_directory) {
            (Some(checkpoint_directory), Some(output_directory)) => Ok(Self {
                checkpoint_directory,
                output_directory,
            }),
            _ => Err(format!("both folders are required\n{USAGE}")),
        }
    }
}

pub fn run(arguments: &[String]) -> Result<(), String> {
    let invocation = ConvertLayaInvocation::parse(arguments)?;
    convert_checkpoint(&invocation)
}

/// Converts one checkpoint folder into one MLX model folder.
pub(crate) fn convert_checkpoint(invocation: &ConvertLayaInvocation) -> Result<(), String> {
    let checkpoint_directory = &invocation.checkpoint_directory;
    let output_directory = &invocation.output_directory;
    let checkpoint_settings = read_json_document(checkpoint_directory, "rl_agent_config.json")?;
    let encoder_config = read_json_document(checkpoint_directory, "encoder/config.json")?;
    let model_config = laya_config::mlx_laya_config(&checkpoint_settings, &encoder_config)?;
    let mut source_checkpoint =
        SourceCheckpoint::open(&checkpoint_directory.join("model.safetensors"))?;
    prepare_empty_output_directory(output_directory)?;
    println!(
        "[convert-laya] status=start tensors={}",
        source_checkpoint.tensor_records.len()
    );
    convert_weights(&mut source_checkpoint, output_directory)?;
    write_json_document(output_directory, "config.json", &model_config)?;
    copy_tokenizer_files(checkpoint_directory, output_directory)?;
    println!(
        "[convert-laya] status=success files=config.json,model.safetensors,tokenizer.json,tokenizer_config.json"
    );
    Ok(())
}

/// Refuses to write into a folder that already holds anything: converting
/// over an existing model could leave a mix of two models' files.
fn prepare_empty_output_directory(output_directory: &Path) -> Result<(), String> {
    if let Ok(mut existing_entries) = fs::read_dir(output_directory)
        && existing_entries.next().is_some()
    {
        return Err(
            "the output folder already holds files; choose a new or empty folder".to_owned(),
        );
    }
    fs::create_dir_all(output_directory)
        .map_err(|create_error| format!("could not create the output folder: {create_error}"))
}

/// Writes every converted tensor, reporting progress while it works.
fn convert_weights(
    source_checkpoint: &mut SourceCheckpoint,
    output_directory: &Path,
) -> Result<(), String> {
    let source_records = source_checkpoint.tensor_records.clone();
    let mut planned_tensors: Vec<PlannedTensor> = Vec::new();
    let mut total_source_bytes = 0_u64;
    for source_record in &source_records {
        planned_tensors.extend(planned_output_tensors(source_record)?);
        total_source_bytes += source_record.byte_count;
    }
    let unfinished_weights_path = output_directory.join(UNFINISHED_WEIGHTS_FILE_NAME);
    let weights_file = fs::File::create(&unfinished_weights_path)
        .map_err(|create_error| format!("could not create the weights file: {create_error}"))?;
    let mut output_checkpoint =
        OutputCheckpoint::begin(BufWriter::new(weights_file), planned_tensors)?;
    let mut conversion_progress = ConversionProgress::start(total_source_bytes);
    for source_record in &source_records {
        let source_bytes = source_checkpoint.read_tensor_bytes(source_record)?;
        append_converted_tensor(&mut output_checkpoint, source_record, source_bytes)?;
        conversion_progress.advance(source_record.byte_count);
    }
    output_checkpoint.finish()?;
    conversion_progress.finish();
    fs::rename(
        &unfinished_weights_path,
        output_directory.join("model.safetensors"),
    )
    .map_err(|rename_error| format!("could not finish the weights file: {rename_error}"))
}

/// Converts one source tensor and appends what it becomes.
fn append_converted_tensor<OutputSink: std::io::Write>(
    output_checkpoint: &mut OutputCheckpoint<OutputSink>,
    source_record: &TensorRecord,
    source_bytes: Vec<u8>,
) -> Result<(), String> {
    match tensor_treatment(&source_record.tensor_name, &source_record.shape) {
        TensorTreatment::Dropped => Ok(()),
        TensorTreatment::KeptAsHalfPrecision { output_name } => output_checkpoint.append_tensor(
            &output_name,
            &half_precision_bytes(source_bytes, source_record.element_type)?,
        ),
        TensorTreatment::Quantized { module_name } => {
            let weights = decoded_weights(&source_bytes, source_record.element_type)?;
            let quantized_matrix = quantize_matrix(&weights)
                .map_err(|quantization_error| format!("{module_name}: {quantization_error}"))?;
            // Four one-byte levels packed lowest first are the little-endian
            // bytes of one u32, so the levels are the packed tensor as they are.
            output_checkpoint.append_tensor(
                &format!("{module_name}.weight"),
                &quantized_matrix.quantized_levels,
            )?;
            output_checkpoint.append_tensor(
                &format!("{module_name}.scales"),
                &half_bits_bytes(&quantized_matrix.scale_half_bits),
            )?;
            output_checkpoint.append_tensor(
                &format!("{module_name}.biases"),
                &half_bits_bytes(&quantized_matrix.bias_half_bits),
            )
        }
    }
}

fn read_json_document(
    checkpoint_directory: &Path,
    relative_file_path: &str,
) -> Result<serde_json::Value, String> {
    let document_bytes = fs::read(checkpoint_directory.join(relative_file_path))
        .map_err(|read_error| format!("could not read {relative_file_path}: {read_error}"))?;
    serde_json::from_slice(&document_bytes)
        .map_err(|json_error| format!("{relative_file_path} is not JSON: {json_error}"))
}

fn write_json_document(
    output_directory: &Path,
    file_name: &str,
    document: &serde_json::Value,
) -> Result<(), String> {
    let document_text = serde_json::to_string_pretty(document)
        .map_err(|json_error| format!("could not write {file_name}: {json_error}"))?;
    fs::write(output_directory.join(file_name), document_text + "\n")
        .map_err(|write_error| format!("could not write {file_name}: {write_error}"))
}

/// Copies the tokenizer beside the weights. The settings file is rewritten
/// without anything that names a local path; the vocabulary is copied as is.
fn copy_tokenizer_files(
    checkpoint_directory: &Path,
    output_directory: &Path,
) -> Result<(), String> {
    let tokenizer_settings =
        read_json_document(checkpoint_directory, "tokenizer/tokenizer_config.json")?;
    write_json_document(
        output_directory,
        TOKENIZER_FILE_NAMES[1],
        &laya_config::without_local_paths(tokenizer_settings),
    )?;
    fs::copy(
        checkpoint_directory
            .join("tokenizer")
            .join(TOKENIZER_FILE_NAMES[0]),
        output_directory.join(TOKENIZER_FILE_NAMES[0]),
    )
    .map_err(|copy_error| format!("could not copy tokenizer.json: {copy_error}"))?;
    Ok(())
}

#[cfg(test)]
#[path = "convert_laya_tests.rs"]
mod tests;
