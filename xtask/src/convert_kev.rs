//! `cargo xtask convert-kev --adapter DIRECTORY --base DIRECTORY --output DIRECTORY`
//! turns a Kev checkpoint into a folder Astronomical serves.
//!
//! A Kev checkpoint (`jaredpalmer/kev-0.8b`, `jaredpalmer/kev-9b`) is a LoRA
//! adapter, a pointer head in a `torch.save` archive, and a tokenizer; the
//! backbone is a separate Qwen3.5 base model. The output is that base with
//! the adapter folded into its weights, the head as safetensors, Kev's
//! settings in `kev_decision.json`, and the tokenizer, so the engine loads
//! one folder and never reads a pickle.
//!
//! The base must be the exact revision the adapter was trained on: `head.pt`
//! names it, and a Hugging Face cache snapshot of any other revision is
//! refused. Merging is arithmetic on the CPU, one tensor at a time; for the
//! 9B adapter build xtask with `--release`, or the merge takes many minutes.

mod backbone_copy;
mod kev_head;
mod lora_adapter;
mod lora_merge;
#[cfg(test)]
mod test_support;

use std::fs;
use std::path::{Path, PathBuf};

use backbone_copy::{BaseShards, write_merged_backbone};
use kev_head::KevHead;
use lora_adapter::LoraAdapter;
use serde_json::{Map, Value};

use crate::model_files::local_paths::without_local_paths;
use crate::model_files::safetensors_file::{OutputCheckpoint, PlannedTensor, TensorElementType};

const USAGE: &str =
    "usage: cargo xtask convert-kev --adapter DIRECTORY --base DIRECTORY --output DIRECTORY";
/// The settings file that marks a folder as a Kev decision model.
pub(crate) const KEV_SETTINGS_FILE_NAME: &str = "kev_decision.json";
pub(crate) const KEV_HEAD_FILE_NAME: &str = "kev_head.safetensors";
/// Files copied from the base as they are: its model config, shard index,
/// licence and the processor configs its loader expects beside the weights.
const COPIED_BASE_FILE_NAMES: [&str; 5] = [
    "config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "LICENSE",
];

/// The three folders one conversion reads from and writes to.
#[derive(Debug, PartialEq)]
pub(crate) struct ConvertKevInvocation {
    pub(crate) adapter_directory: PathBuf,
    pub(crate) base_directory: PathBuf,
    pub(crate) output_directory: PathBuf,
}

impl ConvertKevInvocation {
    pub(crate) fn parse(arguments: &[String]) -> Result<Self, String> {
        let mut folder_by_flag: [(&str, Option<PathBuf>); 3] =
            [("--adapter", None), ("--base", None), ("--output", None)];
        let mut remaining_arguments = arguments.iter();
        while let Some(argument) = remaining_arguments.next() {
            let Some((_, folder_slot)) = folder_by_flag
                .iter_mut()
                .find(|(flag_name, _)| flag_name == argument)
            else {
                return Err(format!("unknown argument {argument}\n{USAGE}"));
            };
            let Some(folder_text) = remaining_arguments.next() else {
                return Err(format!("{argument} needs a directory\n{USAGE}"));
            };
            if folder_slot.is_some() {
                return Err(format!("{argument} was given twice\n{USAGE}"));
            }
            *folder_slot = Some(PathBuf::from(folder_text));
        }
        let [
            (_, Some(adapter_directory)),
            (_, Some(base_directory)),
            (_, Some(output_directory)),
        ] = folder_by_flag
        else {
            return Err(format!("all three folders are required\n{USAGE}"));
        };
        Ok(Self {
            adapter_directory,
            base_directory,
            output_directory,
        })
    }
}

pub fn run(arguments: &[String]) -> Result<(), String> {
    convert_checkpoint(&ConvertKevInvocation::parse(arguments)?)
}

/// Converts one Kev checkpoint and its base into one servable folder.
pub(crate) fn convert_checkpoint(invocation: &ConvertKevInvocation) -> Result<(), String> {
    let kev_head = KevHead::read(&invocation.adapter_directory.join("head.pt"))?;
    refuse_a_different_base_revision(
        &invocation.base_directory,
        kev_head.base_revision.as_deref(),
    )?;
    let mut base_shards = BaseShards::open(&invocation.base_directory)?;
    let mut lora_adapter =
        LoraAdapter::read(&invocation.adapter_directory, &base_shards.tensor_names())?;
    prepare_empty_output_directory(&invocation.output_directory)?;
    println!(
        "[convert-kev] status=start adapted_modules={} rank={} base_shards={}",
        lora_adapter.adapted_modules.len(),
        lora_adapter.rank,
        base_shards.shards.len()
    );
    write_merged_backbone(
        &mut base_shards,
        &mut lora_adapter,
        &invocation.output_directory,
    )?;
    write_kev_head(&kev_head, &invocation.output_directory)?;
    write_kev_settings(&kev_head, &lora_adapter, &invocation.output_directory)?;
    copy_side_files(invocation)?;
    println!("[convert-kev] status=success");
    Ok(())
}

/// A Hugging Face cache snapshot is named after its revision, so a base of
/// another revision than the adapter's is caught before any merge. A plain
/// folder carries no revision and is taken as given.
fn refuse_a_different_base_revision(
    base_directory: &Path,
    expected_revision: Option<&str>,
) -> Result<(), String> {
    let Some(expected_revision) = expected_revision else {
        return Ok(());
    };
    let Some(snapshot_name) = base_directory.file_name().and_then(|name| name.to_str()) else {
        return Ok(());
    };
    let is_revision_named =
        snapshot_name.len() == 40 && snapshot_name.bytes().all(|byte| byte.is_ascii_hexdigit());
    if is_revision_named && snapshot_name != expected_revision {
        return Err(format!(
            "the adapter was trained on base revision {expected_revision}, but the base folder is revision {snapshot_name}"
        ));
    }
    Ok(())
}

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

/// The pointer head as float32 safetensors, under the reference's names.
fn write_kev_head(kev_head: &KevHead, output_directory: &Path) -> Result<(), String> {
    let mut planned_tensors = Vec::with_capacity(kev_head.head_tensors.len());
    for head_tensor in &kev_head.head_tensors {
        planned_tensors.push(PlannedTensor {
            tensor_name: head_tensor.tensor_name.to_owned(),
            element_type: TensorElementType::Float32,
            shape: head_tensor.shape.clone(),
        });
    }
    let head_file = fs::File::create(output_directory.join(KEV_HEAD_FILE_NAME))
        .map_err(|create_error| format!("could not create the head file: {create_error}"))?;
    let mut head_checkpoint = OutputCheckpoint::begin(head_file, planned_tensors)?;
    for head_tensor in &kev_head.head_tensors {
        let mut tensor_bytes = Vec::with_capacity(head_tensor.values.len() * 4);
        for value in &head_tensor.values {
            tensor_bytes.extend_from_slice(&value.to_le_bytes());
        }
        head_checkpoint.append_tensor(head_tensor.tensor_name, &tensor_bytes)?;
    }
    head_checkpoint.finish()
}

fn write_kev_settings(
    kev_head: &KevHead,
    lora_adapter: &LoraAdapter,
    output_directory: &Path,
) -> Result<(), String> {
    let mut settings = Map::new();
    settings.insert("model_type".to_owned(), Value::from("kev"));
    settings.insert("temperature".to_owned(), Value::from(kev_head.temperature));
    settings.insert(
        "pointer_dimension".to_owned(),
        Value::from(kev_head.pointer_dimension),
    );
    settings.insert(
        "base_model".to_owned(),
        Value::from(kev_head.base_model.clone()),
    );
    settings.insert(
        "base_revision".to_owned(),
        kev_head
            .base_revision
            .clone()
            .map_or(Value::Null, Value::from),
    );
    settings.insert("adapter_rank".to_owned(), Value::from(lora_adapter.rank));
    settings.insert("adapter_alpha".to_owned(), Value::from(lora_adapter.alpha));
    let settings_text = serde_json::to_string_pretty(&Value::Object(settings))
        .map_err(|json_error| format!("could not write {KEV_SETTINGS_FILE_NAME}: {json_error}"))?;
    fs::write(
        output_directory.join(KEV_SETTINGS_FILE_NAME),
        settings_text + "\n",
    )
    .map_err(|write_error| format!("could not write {KEV_SETTINGS_FILE_NAME}: {write_error}"))
}

/// Copies the base's side files, and Kev's tokenizer (the base's, as Kev
/// ships it) with any local path removed from its settings.
fn copy_side_files(invocation: &ConvertKevInvocation) -> Result<(), String> {
    for copied_file_name in COPIED_BASE_FILE_NAMES {
        let source_path = invocation.base_directory.join(copied_file_name);
        if !source_path.is_file() {
            continue;
        }
        let document_bytes = fs::read(&source_path)
            .map_err(|read_error| format!("could not read {copied_file_name}: {read_error}"))?;
        let copied_bytes = if copied_file_name.ends_with(".json") {
            scrubbed_json_bytes(&document_bytes, copied_file_name)?
        } else {
            document_bytes
        };
        fs::write(
            invocation.output_directory.join(copied_file_name),
            copied_bytes,
        )
        .map_err(|write_error| format!("could not write {copied_file_name}: {write_error}"))?;
    }
    let tokenizer_settings_bytes =
        fs::read(invocation.adapter_directory.join("tokenizer_config.json"))
            .map_err(|read_error| format!("could not read tokenizer_config.json: {read_error}"))?;
    fs::write(
        invocation.output_directory.join("tokenizer_config.json"),
        scrubbed_json_bytes(&tokenizer_settings_bytes, "tokenizer_config.json")?,
    )
    .map_err(|write_error| format!("could not write tokenizer_config.json: {write_error}"))?;
    fs::copy(
        invocation.adapter_directory.join("tokenizer.json"),
        invocation.output_directory.join("tokenizer.json"),
    )
    .map_err(|copy_error| format!("could not copy tokenizer.json: {copy_error}"))?;
    Ok(())
}

fn scrubbed_json_bytes(document_bytes: &[u8], file_name: &str) -> Result<Vec<u8>, String> {
    let document: Value = serde_json::from_slice(document_bytes)
        .map_err(|json_error| format!("{file_name} is not JSON: {json_error}"))?;
    let scrubbed_text = serde_json::to_string_pretty(&without_local_paths(document))
        .map_err(|json_error| format!("could not write {file_name}: {json_error}"))?;
    Ok((scrubbed_text + "\n").into_bytes())
}

#[cfg(test)]
#[path = "convert_kev_tests.rs"]
mod tests;
