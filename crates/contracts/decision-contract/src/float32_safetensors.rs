//! Reading and writing safetensors files whose every tensor is finite
//! 32-bit floats: the format of both files training writes beside a model's
//! original weights (the trained final layers and the add-on weights).

use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use thiserror::Error;

/// One tensor in 32-bit floats, under its name.
#[derive(Clone, Debug, PartialEq)]
pub struct TunedTensor {
    pub name: String,
    /// Empty for a scalar, as the shipped scalars are stored.
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

/// Why a file is not a set of finite 32-bit tensors.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum Float32TensorsError {
    #[error("it is not a safetensors file: {0}")]
    NotSafetensors(String),
    #[error("it holds no tensors")]
    NoTensors,
    #[error("{0} is not stored as 32-bit floats")]
    NotFloat32(String),
    #[error("{0} holds a value that is not a finite number")]
    NonFiniteValue(String),
}

/// Every tensor of the file, in name order, each checked to be finite
/// 32-bit floats.
///
/// The file's own index is a hash map, so its order changes from run to run;
/// reading in name order makes a refusal name the same tensor every time.
pub(crate) fn read_float32_tensors(
    file_bytes: &[u8],
) -> Result<Vec<TunedTensor>, Float32TensorsError> {
    let stored_tensors = match SafeTensors::deserialize(file_bytes) {
        Ok(stored_tensors) => stored_tensors,
        Err(parse_error) => {
            return Err(Float32TensorsError::NotSafetensors(parse_error.to_string()));
        }
    };
    let mut tensor_names = stored_tensors.names();
    tensor_names.sort_unstable();
    let mut tensors = Vec::with_capacity(tensor_names.len());
    for tensor_name in tensor_names {
        let tensor_view = match stored_tensors.tensor(tensor_name) {
            Ok(tensor_view) => tensor_view,
            Err(read_error) => {
                return Err(Float32TensorsError::NotSafetensors(read_error.to_string()));
            }
        };
        tensors.push(finite_float32_tensor(tensor_name, &tensor_view)?);
    }
    if tensors.is_empty() {
        return Err(Float32TensorsError::NoTensors);
    }
    Ok(tensors)
}

fn finite_float32_tensor(
    tensor_name: &str,
    tensor_view: &TensorView<'_>,
) -> Result<TunedTensor, Float32TensorsError> {
    if tensor_view.dtype() != Dtype::F32 {
        return Err(Float32TensorsError::NotFloat32(tensor_name.to_owned()));
    }
    let (whole_floats, _) = tensor_view.data().as_chunks::<4>();
    let mut values = Vec::with_capacity(whole_floats.len());
    for float_bytes in whole_floats {
        let float_value = f32::from_le_bytes(*float_bytes);
        if !float_value.is_finite() {
            return Err(Float32TensorsError::NonFiniteValue(tensor_name.to_owned()));
        }
        values.push(float_value);
    }
    Ok(TunedTensor {
        name: tensor_name.to_owned(),
        shape: tensor_view.shape().to_vec(),
        values,
    })
}

/// The file's bytes for `tensors`, every one as little-endian 32-bit floats.
///
/// # Errors
/// Returns the serializer's reason when a tensor's values do not fill its shape.
pub(crate) fn float32_tensors_bytes(tensors: &[TunedTensor]) -> Result<Vec<u8>, String> {
    let mut tensor_bytes = Vec::with_capacity(tensors.len());
    for tuned_tensor in tensors {
        let mut little_endian_bytes = Vec::with_capacity(tuned_tensor.values.len() * 4);
        for float_value in &tuned_tensor.values {
            little_endian_bytes.extend_from_slice(&float_value.to_le_bytes());
        }
        tensor_bytes.push(little_endian_bytes);
    }
    let mut tensor_views = Vec::with_capacity(tensors.len());
    for (tuned_tensor, little_endian_bytes) in tensors.iter().zip(&tensor_bytes) {
        match TensorView::new(Dtype::F32, tuned_tensor.shape.clone(), little_endian_bytes) {
            Ok(tensor_view) => tensor_views.push((tuned_tensor.name.as_str(), tensor_view)),
            Err(view_error) => return Err(format!("{}: {view_error}", tuned_tensor.name)),
        }
    }
    match safetensors::serialize(tensor_views, None) {
        Ok(file_bytes) => Ok(file_bytes),
        Err(serialize_error) => Err(serialize_error.to_string()),
    }
}
