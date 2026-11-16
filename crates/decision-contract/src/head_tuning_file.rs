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

use thiserror::Error;

use crate::float32_safetensors::{
    Float32TensorsError, float32_tensors_bytes, read_float32_tensors,
};

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

pub use crate::float32_safetensors::TunedTensor;

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
    let tensors = read_float32_tensors(file_bytes).map_err(HeadTuningFileError::from)?;
    for tuned_tensor in &tensors {
        if !family
            .tunable_tensor_names()
            .contains(&tuned_tensor.name.as_str())
        {
            return Err(HeadTuningFileError::UntunableTensor(
                tuned_tensor.name.clone(),
            ));
        }
    }
    Ok(HeadTuning { tensors })
}

impl From<Float32TensorsError> for HeadTuningFileError {
    fn from(read_error: Float32TensorsError) -> Self {
        match read_error {
            Float32TensorsError::NotSafetensors(reason) => Self::NotSafetensors(reason),
            Float32TensorsError::NoTensors => Self::NoTensors,
            Float32TensorsError::NotFloat32(tensor_name) => Self::NotFloat32(tensor_name),
            Float32TensorsError::NonFiniteValue(tensor_name) => Self::NonFiniteValue(tensor_name),
        }
    }
}

/// The file's bytes for `head_tuning`, every tensor as little-endian
/// 32-bit floats.
///
/// # Errors
/// Returns the serializer's reason when a tensor's values do not fill its shape.
pub fn head_tuning_bytes(head_tuning: &HeadTuning) -> Result<Vec<u8>, String> {
    float32_tensors_bytes(&head_tuning.tensors)
}

#[cfg(test)]
#[path = "head_tuning_file_tests.rs"]
mod tests;
