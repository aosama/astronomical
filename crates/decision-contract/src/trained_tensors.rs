//! A training's returned tensors as the files beside a model's originals
//! hold them: the trained final layers as a head tuning, and the add-ons of
//! a whole-model training as decision adapters.
//!
//! The worker returns both as named `TrainedTensor`s; every tool that writes
//! a trained folder, and every test that checks one, turns them into the
//! file formats the same way, here.

use thiserror::Error;

use crate::decision_adapters_file::{DecisionAdapters, LinearAdapter};
use crate::decision_training_job::TrainedTensor;
use crate::head_tuning_file::{HeadTuning, TunedTensor};

const DOWN_TENSOR_SUFFIX: &str = ".adapter_down";
const UP_TENSOR_SUFFIX: &str = ".adapter_up";

/// Add-on tensors that do not form `down` and `up` pairs of one layer.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[error("the worker returned add-on weights that do not pair up")]
pub struct UnpairedAddOns;

/// The trained final layers as the 32-bit head tuning file holds them.
#[must_use]
pub fn head_tuning_of(trained_tensors: &[TrainedTensor]) -> HeadTuning {
    let mut tuned_tensors = Vec::with_capacity(trained_tensors.len());
    for trained_tensor in trained_tensors {
        tuned_tensors.push(TunedTensor {
            name: trained_tensor.name.clone(),
            shape: trained_tensor.shape.clone(),
            values: trained_tensor.values.0.clone(),
        });
    }
    HeadTuning {
        tensors: tuned_tensors,
    }
}

/// The add-ons as the decision adapters file holds them: the worker sends
/// each layer's `down` then its `up`. `None` when there are none, as after a
/// final-layer training.
///
/// # Errors
/// Returns [`UnpairedAddOns`] when a `down` lacks its `up`, the two name
/// different layers, or their ranks differ.
pub fn decision_adapters_of(
    adapter_tensors: &[TrainedTensor],
) -> Result<Option<DecisionAdapters>, UnpairedAddOns> {
    if adapter_tensors.is_empty() {
        return Ok(None);
    }
    let mut adapters = Vec::with_capacity(adapter_tensors.len() / 2);
    for adapter_pair in adapter_tensors.chunks(2) {
        let [down_tensor, up_tensor] = adapter_pair else {
            return Err(UnpairedAddOns);
        };
        adapters.push(paired_adapter(down_tensor, up_tensor)?);
    }
    Ok(Some(DecisionAdapters { adapters }))
}

fn paired_adapter(
    down_tensor: &TrainedTensor,
    up_tensor: &TrainedTensor,
) -> Result<LinearAdapter, UnpairedAddOns> {
    let (Some(target), Some(up_target)) = (
        down_tensor.name.strip_suffix(DOWN_TENSOR_SUFFIX),
        up_tensor.name.strip_suffix(UP_TENSOR_SUFFIX),
    ) else {
        return Err(UnpairedAddOns);
    };
    let ([rank, input_width], [output_width, up_rank]) =
        (down_tensor.shape.as_slice(), up_tensor.shape.as_slice())
    else {
        return Err(UnpairedAddOns);
    };
    if target != up_target || rank != up_rank {
        return Err(UnpairedAddOns);
    }
    Ok(LinearAdapter {
        target: target.to_owned(),
        rank: *rank,
        input_width: *input_width,
        output_width: *output_width,
        down_values: down_tensor.values.0.clone(),
        up_values: up_tensor.values.0.clone(),
    })
}

#[cfg(test)]
#[path = "trained_tensors_tests.rs"]
mod tests;
