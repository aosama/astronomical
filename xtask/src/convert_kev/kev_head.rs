//! Kev's pointer head and settings, read from its `head.pt` without
//! unpickling it.
//!
//! `head.pt` is a `torch.save` archive of one dictionary: the head's four
//! tensors under `head`, and the settings the reference reads beside them
//! (the calibration temperature, the pointer width, the base model and its
//! revision, and two training switches the serving path must check).

use std::path::Path;

use crate::model_files::bfloat16::bfloat16_bits_to_f32;
use crate::model_files::half_float::half_bits_to_f32;
use crate::model_files::stored_zip::StoredZipArchive;
use crate::model_files::torch_pickle::{
    PickledTensor, PickledValue, StorageElementType, read_pickle,
};

/// The pointer head's projections, in the reference's names: `q` reads the
/// question's decision token, `k` each option's closing token.
pub(crate) const HEAD_TENSOR_NAMES: [&str; 4] = ["q.weight", "q.bias", "k.weight", "k.bias"];

/// One head tensor as float32 values, with its shape.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HeadTensor {
    pub(crate) tensor_name: &'static str,
    pub(crate) shape: Vec<u64>,
    pub(crate) values: Vec<f32>,
}

/// Everything the converter keeps from `head.pt`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct KevHead {
    pub(crate) head_tensors: Vec<HeadTensor>,
    pub(crate) temperature: f64,
    pub(crate) pointer_dimension: u64,
    pub(crate) base_model: String,
    pub(crate) base_revision: Option<String>,
}

impl KevHead {
    pub(crate) fn read(head_archive_path: &Path) -> Result<Self, String> {
        let head_archive = StoredZipArchive::open(head_archive_path)
            .map_err(|archive_error| format!("head.pt: {archive_error}"))?;
        Self::from_archive(&head_archive)
    }

    pub(crate) fn from_archive(head_archive: &StoredZipArchive) -> Result<Self, String> {
        let archive_prefix = archive_prefix(head_archive)?;
        if let Ok(byte_order) = head_archive.entry_bytes(&format!("{archive_prefix}/byteorder"))
            && byte_order != b"little"
        {
            return Err(
                "head.pt stores its tensors big-endian, which no Mac reads natively".to_owned(),
            );
        }
        let checkpoint_settings =
            read_pickle(head_archive.entry_bytes(&format!("{archive_prefix}/data.pkl"))?)
                .map_err(|pickle_error| format!("head.pt: {pickle_error}"))?;
        refuse_unsupported_layouts(&checkpoint_settings)?;
        let Some(head_state) = checkpoint_settings.member("head") else {
            return Err("head.pt has no head".to_owned());
        };
        let mut head_tensors = Vec::with_capacity(HEAD_TENSOR_NAMES.len());
        for tensor_name in HEAD_TENSOR_NAMES {
            let Some(PickledValue::Tensor(pickled_tensor)) = head_state.member(tensor_name) else {
                return Err(format!("head.pt has no head tensor {tensor_name}"));
            };
            head_tensors.push(HeadTensor {
                tensor_name,
                shape: pickled_tensor.shape.clone(),
                values: tensor_values(head_archive, &archive_prefix, pickled_tensor)?,
            });
        }
        let pointer_dimension = head_tensors[0].shape.first().copied().unwrap_or(0);
        Ok(Self {
            head_tensors,
            temperature: number_setting(&checkpoint_settings, "temperature")?.unwrap_or(1.0),
            pointer_dimension,
            base_model: text_setting(&checkpoint_settings, "base")?
                .ok_or_else(|| "head.pt does not name its base model".to_owned())?,
            base_revision: text_setting(&checkpoint_settings, "base_revision")?,
        })
    }
}

/// `torch.save` puts every entry under one folder named after the file.
fn archive_prefix(head_archive: &StoredZipArchive) -> Result<String, String> {
    for entry_name in head_archive.entry_names() {
        if let Some(archive_prefix) = entry_name.strip_suffix("/data.pkl") {
            return Ok(archive_prefix.to_owned());
        }
    }
    Err("head.pt has no data.pkl; it is not a torch.save archive".to_owned())
}

/// The serving path reads each option and the decision token from one
/// causal row; a checkpoint trained with isolated options or extra special
/// embeddings needs a different layout, so it is refused rather than misread.
fn refuse_unsupported_layouts(checkpoint_settings: &PickledValue) -> Result<(), String> {
    for switch_name in ["option_isolation", "special_embeddings"] {
        if let Some(PickledValue::Boolean(true) | PickledValue::Integer(1)) =
            checkpoint_settings.member(switch_name)
        {
            return Err(format!(
                "the checkpoint was trained with {switch_name}, which this converter does not support"
            ));
        }
    }
    if let Some(PickledValue::Text(weights_layout)) = checkpoint_settings.member("weights")
        && weights_layout == "full"
    {
        return Err("the checkpoint holds full weights, not an adapter".to_owned());
    }
    Ok(())
}

fn number_setting(
    checkpoint_settings: &PickledValue,
    setting_name: &str,
) -> Result<Option<f64>, String> {
    match checkpoint_settings.member(setting_name) {
        None | Some(PickledValue::None) => Ok(None),
        Some(PickledValue::Float(number)) if number.is_finite() && *number > 0.0 => {
            Ok(Some(*number))
        }
        Some(PickledValue::Integer(number)) if *number > 0 => Ok(Some(*number as f64)),
        Some(_) => Err(format!("head.pt has an invalid {setting_name}")),
    }
}

fn text_setting(
    checkpoint_settings: &PickledValue,
    setting_name: &str,
) -> Result<Option<String>, String> {
    match checkpoint_settings.member(setting_name) {
        None | Some(PickledValue::None) => Ok(None),
        Some(PickledValue::Text(text)) => Ok(Some(text.clone())),
        Some(_) => Err(format!("head.pt has an invalid {setting_name}")),
    }
}

/// A tensor's values in row-major order, read from its storage entry.
fn tensor_values(
    head_archive: &StoredZipArchive,
    archive_prefix: &str,
    pickled_tensor: &PickledTensor,
) -> Result<Vec<f32>, String> {
    let element_count: u64 = pickled_tensor.shape.iter().product();
    let mut expected_stride = 1;
    for (dimension, stride) in pickled_tensor
        .shape
        .iter()
        .zip(&pickled_tensor.strides)
        .rev()
    {
        if *stride != expected_stride {
            return Err("head.pt holds a tensor that is not stored row-major".to_owned());
        }
        expected_stride *= dimension;
    }
    let storage = &pickled_tensor.storage;
    if pickled_tensor.storage_offset + element_count > storage.element_count {
        return Err("head.pt holds a tensor larger than its storage".to_owned());
    }
    let storage_bytes =
        head_archive.entry_bytes(&format!("{archive_prefix}/data/{}", storage.storage_key))?;
    let element_bytes = storage.element_type.bytes_per_element();
    if storage_bytes.len() as u64 != storage.element_count * element_bytes {
        return Err("head.pt's tensor storage is not the size its header says".to_owned());
    }
    let first_byte = (pickled_tensor.storage_offset * element_bytes) as usize;
    let tensor_bytes =
        &storage_bytes[first_byte..first_byte + (element_count * element_bytes) as usize];
    Ok(decoded_values(tensor_bytes, storage.element_type))
}

fn decoded_values(tensor_bytes: &[u8], element_type: StorageElementType) -> Vec<f32> {
    let mut values = Vec::with_capacity(tensor_bytes.len() / 2);
    if element_type == StorageElementType::Float32 {
        let (whole_singles, _) = tensor_bytes.as_chunks::<4>();
        for single_bytes in whole_singles {
            values.push(f32::from_le_bytes(*single_bytes));
        }
        return values;
    }
    let (whole_halves, _) = tensor_bytes.as_chunks::<2>();
    for sixteen_bit_bytes in whole_halves {
        let sixteen_bits = u16::from_le_bytes(*sixteen_bit_bytes);
        values.push(if element_type == StorageElementType::BFloat16 {
            bfloat16_bits_to_f32(sixteen_bits)
        } else {
            half_bits_to_f32(sixteen_bits)
        });
    }
    values
}

#[cfg(test)]
#[path = "kev_head_tests.rs"]
mod tests;
