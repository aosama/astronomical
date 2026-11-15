//! `decision_head_tuning.safetensors`: a decision model's trained final
//! layers, kept beside the original weights instead of written into them.
//!
//! The shipped heads are stored at reduced precision (Clef's in bfloat16,
//! Laya's scorer in float16). Writing trained weights back at that precision
//! rounds away the small changes training makes, a known way fine-tunes
//! silently come out worse. So the trained tensors are stored here in 32-bit
//! floats, under the names the shipped tensors have, and an engine that finds
//! this file uses them in place of the shipped ones. The original files stay
//! byte-identical.
//!
//! The file describes itself by its tensor names: each family may tune only
//! the tensors listed for it here, so a file meant for one family is refused
//! by another. Shapes are checked against the shipped tensors when an engine
//! substitutes them, because only the engine knows those.

use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use thiserror::Error;

/// The trained head of a decision model folder, beside its original weights.
pub const DECISION_HEAD_TUNING_FILE_NAME: &str = "decision_head_tuning.safetensors";
/// Clef's trained part, the largest, is 4.2 million floats (about 17 MB); a
/// file far beyond that is not one of these.
pub const MAXIMUM_HEAD_TUNING_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// The family whose head a tuning file replaces part of.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TunableHeadFamily {
    Kev,
    Laya,
    Clef,
}

const KEV_TUNABLE_TENSORS: &[&str] = &["q.weight", "q.bias", "k.weight", "k.bias"];
const LAYA_TUNABLE_TENSORS: &[&str] = &[
    "scorer.layers.0.weight",
    "scorer.layers.0.bias",
    "scorer.layers.1.weight",
    "scorer.layers.1.bias",
    "scorer.layers.3.weight",
    "scorer.layers.3.bias",
];
const CLEF_TUNABLE_TENSORS: &[&str] = &[
    "residual_scorer.0.weight",
    "residual_scorer.0.bias",
    "residual_scorer.3.weight",
    "residual_scorer.3.bias",
    "prior_logit_scale",
    "joint_logit_scale",
    "residual_gate",
];

impl TunableHeadFamily {
    /// The shipped tensors this family's training may replace.
    #[must_use]
    pub const fn tunable_tensor_names(self) -> &'static [&'static str] {
        match self {
            Self::Kev => KEV_TUNABLE_TENSORS,
            Self::Laya => LAYA_TUNABLE_TENSORS,
            Self::Clef => CLEF_TUNABLE_TENSORS,
        }
    }
}

/// One trained tensor, in 32-bit floats, under its shipped name.
#[derive(Clone, Debug, PartialEq)]
pub struct TunedTensor {
    pub name: String,
    /// Empty for a scalar, as the shipped scalars are stored.
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

/// Every trained tensor of one folder.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HeadTuning {
    pub tensors: Vec<TunedTensor>,
}

impl HeadTuning {
    /// The trained tensor with this shipped name, if training replaced it.
    #[must_use]
    pub fn tensor(&self, tensor_name: &str) -> Option<&TunedTensor> {
        // The repository's lints require `find` here over a hand-written loop.
        self.tensors
            .iter()
            .find(|tuned_tensor| tuned_tensor.name == tensor_name)
    }

    /// Whether the folder has no trained head, so it serves as shipped.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Why a tuning file cannot be used.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum HeadTuningFileError {
    #[error("it is not a safetensors file: {0}")]
    NotSafetensors(String),
    #[error("it holds no tensors")]
    NoTensors,
    #[error("{0} is not a tensor this model's training may change")]
    UntunableTensor(String),
    #[error("{0} is not stored as 32-bit floats")]
    NotFloat32(String),
    #[error("{0} holds a value that is not a finite number")]
    NonFiniteValue(String),
}

/// Reads and checks a tuning file's bytes for `family`.
pub fn parse_head_tuning(
    file_bytes: &[u8],
    family: TunableHeadFamily,
) -> Result<HeadTuning, HeadTuningFileError> {
    let stored_tensors = match SafeTensors::deserialize(file_bytes) {
        Ok(stored_tensors) => stored_tensors,
        Err(parse_error) => {
            return Err(HeadTuningFileError::NotSafetensors(parse_error.to_string()));
        }
    };
    // The file's own index is a hash map, so its order changes from run to
    // run; checking in name order makes a refusal name the same tensor
    // every time.
    let mut tensor_names = stored_tensors.names();
    tensor_names.sort_unstable();
    let mut tensors = Vec::with_capacity(tensor_names.len());
    for tensor_name in tensor_names {
        let tensor_view = match stored_tensors.tensor(tensor_name) {
            Ok(tensor_view) => tensor_view,
            Err(read_error) => {
                return Err(HeadTuningFileError::NotSafetensors(read_error.to_string()));
            }
        };
        tensors.push(checked_tensor(
            tensor_name.to_owned(),
            &tensor_view,
            family,
        )?);
    }
    if tensors.is_empty() {
        return Err(HeadTuningFileError::NoTensors);
    }
    Ok(HeadTuning { tensors })
}

fn checked_tensor(
    tensor_name: String,
    tensor_view: &TensorView<'_>,
    family: TunableHeadFamily,
) -> Result<TunedTensor, HeadTuningFileError> {
    if !family
        .tunable_tensor_names()
        .contains(&tensor_name.as_str())
    {
        return Err(HeadTuningFileError::UntunableTensor(tensor_name));
    }
    if tensor_view.dtype() != Dtype::F32 {
        return Err(HeadTuningFileError::NotFloat32(tensor_name));
    }
    let (whole_floats, _) = tensor_view.data().as_chunks::<4>();
    let mut values = Vec::with_capacity(whole_floats.len());
    for float_bytes in whole_floats {
        let float_value = f32::from_le_bytes(*float_bytes);
        if !float_value.is_finite() {
            return Err(HeadTuningFileError::NonFiniteValue(tensor_name));
        }
        values.push(float_value);
    }
    Ok(TunedTensor {
        name: tensor_name,
        shape: tensor_view.shape().to_vec(),
        values,
    })
}

/// The file's bytes for `head_tuning`, every tensor as little-endian
/// 32-bit floats.
///
/// # Errors
/// Returns the serializer's reason when a tensor's values do not fill its shape.
pub fn head_tuning_bytes(head_tuning: &HeadTuning) -> Result<Vec<u8>, String> {
    let mut tensor_bytes = Vec::with_capacity(head_tuning.tensors.len());
    for tuned_tensor in &head_tuning.tensors {
        let mut little_endian_bytes = Vec::with_capacity(tuned_tensor.values.len() * 4);
        for float_value in &tuned_tensor.values {
            little_endian_bytes.extend_from_slice(&float_value.to_le_bytes());
        }
        tensor_bytes.push(little_endian_bytes);
    }
    let mut tensor_views = Vec::with_capacity(head_tuning.tensors.len());
    for (tuned_tensor, little_endian_bytes) in head_tuning.tensors.iter().zip(&tensor_bytes) {
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

#[cfg(test)]
#[path = "head_tuning_file_tests.rs"]
mod tests;
