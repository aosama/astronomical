//! A model folder's trainable part as it serves today: the family's head
//! with its shipped weights, and any earlier training laid over them.
//!
//! Training starts from exactly what the folder serves. Starting anywhere
//! else (a fresh random head, or the shipped weights ignoring an earlier
//! tuning) would make the first thing training does a change the person
//! never asked for; the parity check before training also relies on it.

use std::path::Path;

use decision_contract::head_tuning_file::{
    DECISION_HEAD_TUNING_FILE_NAME, HeadTuning, MAXIMUM_HEAD_TUNING_FILE_BYTES, TunableHeadFamily,
    parse_head_tuning,
};
use decision_contract::kev_folder_layout::KevFolderLayout;
use thiserror::Error;

use crate::clef_trainable::ClefTrainableHead;
use crate::head_parameters::ParameterLayout;
use crate::kev_trainable::KevTrainableHead;
use crate::laya_trainable::LayaTrainableHead;
use crate::safetensors_slices::{
    NamedFloatTensor, SafetensorsSliceError, read_named_float_tensors,
};
use crate::trainable_head::TrainableHead;

const LAYA_WEIGHTS_FILE_NAME: &str = "model.safetensors";
const CLEF_HEAD_FILE_NAME: &str = "joint_head.safetensors";

/// A family's trainable part and the parameters the folder serves.
pub struct ShippedHead {
    pub family: TunableHeadFamily,
    pub trainable_head: Box<dyn TrainableHead>,
    pub served_parameters: Vec<f32>,
}

/// Why the folder's trainable part could not be read.
#[derive(Debug, Error)]
pub enum ShippedHeadError {
    #[error(transparent)]
    Read(#[from] SafetensorsSliceError),
    #[error("{tensor_name} has shape {found:?}, where this head needs {expected:?}")]
    UnexpectedShape {
        tensor_name: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    #[error("the folder's earlier training cannot be read: {0}")]
    EarlierTuning(String),
}

/// Reads the trainable part of the `family` model in `model_directory`.
pub fn read_shipped_head(
    model_directory: &Path,
    family: TunableHeadFamily,
) -> Result<ShippedHead, ShippedHeadError> {
    let (trainable_head, shipped_tensors) = match family {
        TunableHeadFamily::Kev => kev_head(model_directory)?,
        TunableHeadFamily::Laya => laya_head(model_directory)?,
        TunableHeadFamily::Clef => clef_head(model_directory)?,
    };
    let layout = trainable_head.layout();
    let mut served_parameters = vec![0.0_f32; layout.parameter_count];
    for shipped_tensor in &shipped_tensors {
        place_tensor(
            layout,
            &shipped_tensor.name,
            &shipped_tensor.shape,
            &shipped_tensor.values,
            &mut served_parameters,
        )?;
    }
    for tuned_tensor in earlier_tuning(model_directory, family)?.tensors {
        place_tensor(
            layout,
            &tuned_tensor.name,
            &tuned_tensor.shape,
            &tuned_tensor.values,
            &mut served_parameters,
        )?;
    }
    Ok(ShippedHead {
        family,
        trainable_head,
        served_parameters,
    })
}

type HeadAndTensors = (Box<dyn TrainableHead>, Vec<NamedFloatTensor>);

/// The head file of the first Kev layout the folder holds one of
/// (`kev_head.safetensors`, then `head.safetensors`); the converted name when
/// neither exists, so the missing-file error names the usual one. The engine
/// validated the folder's settings before training starts.
fn kev_head_file_name(model_directory: &Path) -> &'static str {
    for layout in KevFolderLayout::ALL {
        if model_directory.join(layout.head_file_name()).is_file() {
            return layout.head_file_name();
        }
    }
    KevFolderLayout::Converted.head_file_name()
}

fn kev_head(model_directory: &Path) -> Result<HeadAndTensors, ShippedHeadError> {
    let shipped_tensors = read_named_float_tensors(
        &model_directory.join(kev_head_file_name(model_directory)),
        TunableHeadFamily::Kev.tunable_tensor_names(),
    )?;
    let query_shape = &shipped_tensors[0].shape;
    let [pointer_dimension, hidden_size] = query_shape.as_slice() else {
        return Err(unexpected_shape("q.weight", vec![0, 0], query_shape));
    };
    Ok((
        Box::new(KevTrainableHead::new(*pointer_dimension, *hidden_size)),
        shipped_tensors,
    ))
}

fn laya_head(model_directory: &Path) -> Result<HeadAndTensors, ShippedHeadError> {
    let shipped_tensors = read_named_float_tensors(
        &model_directory.join(LAYA_WEIGHTS_FILE_NAME),
        TunableHeadFamily::Laya.tunable_tensor_names(),
    )?;
    // Names are read in the family's order: the norm, then layers 1 and 3.
    let hidden_shape = &shipped_tensors[2].shape;
    let [hidden_width, marker_width] = hidden_shape.as_slice() else {
        return Err(unexpected_shape(
            "scorer.layers.1.weight",
            vec![0, 0],
            hidden_shape,
        ));
    };
    Ok((
        Box::new(LayaTrainableHead::new(*marker_width, *hidden_width)),
        shipped_tensors,
    ))
}

fn clef_head(model_directory: &Path) -> Result<HeadAndTensors, ShippedHeadError> {
    let shipped_tensors = read_named_float_tensors(
        &model_directory.join(CLEF_HEAD_FILE_NAME),
        TunableHeadFamily::Clef.tunable_tensor_names(),
    )?;
    let residual_shape = &shipped_tensors[0].shape;
    let [residual_width, feature_width] = residual_shape.as_slice() else {
        return Err(unexpected_shape(
            "residual_scorer.0.weight",
            vec![0, 0],
            residual_shape,
        ));
    };
    if feature_width % 4 != 0 {
        return Err(unexpected_shape(
            "residual_scorer.0.weight",
            vec![*residual_width, feature_width / 4 * 4],
            residual_shape,
        ));
    }
    Ok((
        Box::new(ClefTrainableHead::new(feature_width / 4, *residual_width)),
        shipped_tensors,
    ))
}

/// The folder's own earlier training, or none.
fn earlier_tuning(
    model_directory: &Path,
    family: TunableHeadFamily,
) -> Result<HeadTuning, ShippedHeadError> {
    let tuning_path = model_directory.join(DECISION_HEAD_TUNING_FILE_NAME);
    let tuning_size = match std::fs::metadata(&tuning_path) {
        Ok(tuning_metadata) => tuning_metadata.len(),
        Err(metadata_error) if metadata_error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(HeadTuning::default());
        }
        Err(metadata_error) => {
            return Err(ShippedHeadError::EarlierTuning(metadata_error.to_string()));
        }
    };
    if tuning_size > MAXIMUM_HEAD_TUNING_FILE_BYTES {
        return Err(ShippedHeadError::EarlierTuning(
            "it is too large".to_owned(),
        ));
    }
    let tuning_bytes = match std::fs::read(&tuning_path) {
        Ok(tuning_bytes) => tuning_bytes,
        Err(read_error) => return Err(ShippedHeadError::EarlierTuning(read_error.to_string())),
    };
    match parse_head_tuning(&tuning_bytes, family) {
        Ok(head_tuning) => Ok(head_tuning),
        Err(tuning_error) => Err(ShippedHeadError::EarlierTuning(tuning_error.to_string())),
    }
}

/// Copies one tensor's values into its slice of the flat parameters, after
/// checking its shape against the layout's.
fn place_tensor(
    layout: &ParameterLayout,
    tensor_name: &str,
    shape: &[usize],
    values: &[f32],
    parameters: &mut [f32],
) -> Result<(), ShippedHeadError> {
    let parameter_tensor = layout.tensor(tensor_name);
    if parameter_tensor.shape != shape || values.len() != parameter_tensor.length {
        return Err(unexpected_shape(
            tensor_name,
            parameter_tensor.shape.clone(),
            shape,
        ));
    }
    parameter_tensor
        .values_mut(parameters)
        .copy_from_slice(values);
    Ok(())
}

fn unexpected_shape(tensor_name: &str, expected: Vec<usize>, found: &[usize]) -> ShippedHeadError {
    ShippedHeadError::UnexpectedShape {
        tensor_name: tensor_name.to_owned(),
        expected,
        found: found.to_vec(),
    }
}

#[cfg(test)]
#[path = "shipped_head_tests.rs"]
mod tests;
