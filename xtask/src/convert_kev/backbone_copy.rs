//! Writes the merged backbone: every base shard, tensor for tensor, with the
//! adapted weights replaced by their merged values.
//!
//! The shards keep the base checkpoint's names, layout and split, so the
//! output is the base model with different numbers in some tensors: the
//! engine's own loader, which reads the published Hugging Face layout,
//! reads it unchanged, and the shard index can be copied as it is.

use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use super::lora_adapter::{AdaptedModule, LoraAdapter};
use super::lora_merge::{LoraMergeInputs, MergeShape, merged_weight_bytes};
use crate::model_files::conversion_progress::ConversionProgress;
use crate::model_files::safetensors_file::{
    OutputCheckpoint, PlannedTensor, SourceCheckpoint, TensorRecord,
};
use crate::model_files::tensor_values::float32_values;

const UNFINISHED_SHARD_SUFFIX: &str = ".unfinished";

/// The base checkpoint's shards, opened.
pub(crate) struct BaseShards {
    pub(crate) shard_paths: Vec<PathBuf>,
    pub(crate) shards: Vec<SourceCheckpoint>,
}

impl BaseShards {
    /// Opens every shard the base names: the shards listed in its index, or
    /// its single `model.safetensors`.
    pub(crate) fn open(base_directory: &Path) -> Result<Self, String> {
        let shard_file_names = base_shard_file_names(base_directory)?;
        let mut shard_paths = Vec::with_capacity(shard_file_names.len());
        let mut shards = Vec::with_capacity(shard_file_names.len());
        for shard_file_name in shard_file_names {
            let shard_path = base_directory.join(&shard_file_name);
            shards.push(SourceCheckpoint::open(&shard_path)?);
            shard_paths.push(shard_path);
        }
        Ok(Self {
            shard_paths,
            shards,
        })
    }

    pub(crate) fn tensor_names(&self) -> Vec<String> {
        let mut tensor_names = Vec::new();
        for shard in &self.shards {
            for tensor_record in &shard.tensor_records {
                tensor_names.push(tensor_record.tensor_name.clone());
            }
        }
        tensor_names
    }

    fn total_bytes(&self) -> u64 {
        let mut total_bytes = 0;
        for shard in &self.shards {
            for tensor_record in &shard.tensor_records {
                total_bytes += tensor_record.byte_count;
            }
        }
        total_bytes
    }
}

fn base_shard_file_names(base_directory: &Path) -> Result<Vec<String>, String> {
    let index_path = base_directory.join("model.safetensors.index.json");
    if !index_path.is_file() {
        return Ok(vec!["model.safetensors".to_owned()]);
    }
    let shard_index: serde_json::Value = serde_json::from_slice(
        &fs::read(&index_path)
            .map_err(|read_error| format!("could not read the shard index: {read_error}"))?,
    )
    .map_err(|json_error| format!("the shard index is not JSON: {json_error}"))?;
    let Some(weight_map) = shard_index
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
    else {
        return Err("the shard index has no weight_map".to_owned());
    };
    let mut shard_file_names = Vec::new();
    for shard_file_name in weight_map.values() {
        let Some(shard_file_name) = shard_file_name.as_str() else {
            return Err("the shard index names a shard that is not text".to_owned());
        };
        if shard_file_name.contains('/') || shard_file_name.starts_with('.') {
            return Err(format!(
                "the shard index names a file outside the model folder: {shard_file_name}"
            ));
        }
        if !shard_file_names
            .iter()
            .any(|known_name| known_name == shard_file_name)
        {
            shard_file_names.push(shard_file_name.to_owned());
        }
    }
    shard_file_names.sort();
    Ok(shard_file_names)
}

/// Writes every base shard into `output_directory`, merging the adapter.
pub(crate) fn write_merged_backbone(
    base_shards: &mut BaseShards,
    lora_adapter: &mut LoraAdapter,
    output_directory: &Path,
) -> Result<(), String> {
    let mut conversion_progress =
        ConversionProgress::start("convert-kev", base_shards.total_bytes());
    let mut merged_module_count = 0;
    for (shard_path, shard) in base_shards
        .shard_paths
        .iter()
        .zip(base_shards.shards.iter_mut())
    {
        let Some(shard_file_name) = shard_path.file_name() else {
            return Err("a base shard has no file name".to_owned());
        };
        let finished_path = output_directory.join(shard_file_name);
        let mut unfinished_path = finished_path.clone().into_os_string();
        unfinished_path.push(UNFINISHED_SHARD_SUFFIX);
        let output_file = fs::File::create(&unfinished_path)
            .map_err(|create_error| format!("could not create an output shard: {create_error}"))?;
        let tensor_records = shard.tensor_records.clone();
        let mut output_shard =
            OutputCheckpoint::begin(BufWriter::new(output_file), planned_like(&tensor_records))?;
        for tensor_record in &tensor_records {
            let base_bytes = shard.read_tensor_bytes(tensor_record)?;
            let written_bytes = match lora_adapter
                .adapted_modules
                .get(&tensor_record.tensor_name)
                .cloned()
            {
                Some(adapted_module) => {
                    merged_module_count += 1;
                    merged_tensor_bytes(lora_adapter, &adapted_module, tensor_record, &base_bytes)?
                }
                None => base_bytes,
            };
            output_shard.append_tensor(&tensor_record.tensor_name, &written_bytes)?;
            conversion_progress.advance(tensor_record.byte_count);
        }
        output_shard.finish()?;
        fs::rename(&unfinished_path, &finished_path)
            .map_err(|rename_error| format!("could not finish an output shard: {rename_error}"))?;
    }
    conversion_progress.finish();
    if merged_module_count != lora_adapter.adapted_modules.len() {
        return Err("not every adapted module was found in the base shards".to_owned());
    }
    Ok(())
}

fn planned_like(tensor_records: &[TensorRecord]) -> Vec<PlannedTensor> {
    let mut planned_tensors = Vec::with_capacity(tensor_records.len());
    for tensor_record in tensor_records {
        planned_tensors.push(PlannedTensor {
            tensor_name: tensor_record.tensor_name.clone(),
            element_type: tensor_record.element_type,
            shape: tensor_record.shape.clone(),
        });
    }
    planned_tensors
}

fn merged_tensor_bytes(
    lora_adapter: &mut LoraAdapter,
    adapted_module: &AdaptedModule,
    base_record: &TensorRecord,
    base_bytes: &[u8],
) -> Result<Vec<u8>, String> {
    if base_record.shape.len() != 2
        || base_record.shape[0] != adapted_module.up_projection.shape[0]
        || base_record.shape[1] != adapted_module.down_projection.shape[1]
    {
        return Err(format!(
            "the adapter's factors do not fit base tensor {}",
            base_record.tensor_name
        ));
    }
    let adapter_checkpoint = &mut lora_adapter.adapter_checkpoint;
    let down_projection = float32_values(
        &adapter_checkpoint.read_tensor_bytes(&adapted_module.down_projection)?,
        adapted_module.down_projection.element_type,
    )?;
    let up_projection = float32_values(
        &adapter_checkpoint.read_tensor_bytes(&adapted_module.up_projection)?,
        adapted_module.up_projection.element_type,
    )?;
    merged_weight_bytes(&LoraMergeInputs {
        base_weight_bytes: base_bytes,
        base_element_type: base_record.element_type,
        down_projection: &down_projection,
        up_projection: &up_projection,
        scale: lora_adapter.scale,
        merge_shape: MergeShape {
            output_count: base_record.shape[0] as usize,
            input_count: base_record.shape[1] as usize,
            rank: lora_adapter.rank as usize,
        },
    })
}
