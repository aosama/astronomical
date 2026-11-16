//! The trainable parameters of a whole-model training, in the order the
//! training loop moves them: the final scorer's tensors under their shipped
//! names (as final-layer training lays them out), then each add-on's `down`
//! and `up`, layer by layer.
//!
//! Every add-on starts with `up` at zero, so the starting parameters score
//! exactly as the served model does, which the parity check proves before
//! any weight moves. `down` starts small and random, as is usual for these
//! add-ons: were it zero too, neither matrix would ever get a gradient.

use std::collections::BTreeMap;

use astronomical_decision_contract::FloatVector;
use astronomical_decision_contract::decision_training_job::TrainedTensor;
use astronomical_decision_training::head_parameters::ParameterLayout;
use mlx_bridge::{MlxArray, MlxCError};

use crate::decision_models::decision_adapters::AdapterTarget;
use crate::decision_models::training::gpu_math::{GpuMath, GpuResult};
use crate::decision_models::training::gpu_upload::upload_parameters;
use crate::linear_adapters::LinearAdapters;

/// How many numbers each add-on narrows a layer's input to. Eight is the
/// usual size for a few thousand examples: enough to change what a layer
/// reads, small enough not to memorise a few hundred questions.
pub(crate) const ADAPTER_RANK: usize = 8;

/// Where every trainable parameter of a whole-model training goes.
pub(crate) struct WholeModelLayout {
    pub(crate) scorer_layout: ParameterLayout,
    pub(crate) adapter_targets: Vec<AdapterTarget>,
    pub(crate) adapter_rank: usize,
}

/// The parts a forward pass reads in place of the shipped ones.
pub(crate) struct TrainedPartArrays {
    pub(crate) tuned_tensors: BTreeMap<String, MlxArray>,
    pub(crate) adapters: LinearAdapters,
}

impl WholeModelLayout {
    /// The scorer as served, uploaded, then every add-on at its start.
    pub(crate) fn starting_parameters(
        &self,
        math: GpuMath<'_>,
        served_scorer_values: &[f32],
    ) -> GpuResult<Vec<MlxArray>> {
        let mut parameters = upload_parameters(math, &self.scorer_layout, served_scorer_values)?;
        for (target_position, target) in self.adapter_targets.iter().enumerate() {
            let down_values = starting_down_values(
                self.adapter_rank * target.input_width,
                target.input_width,
                target_position as u64,
            );
            parameters.push(math.upload(&down_values, &[self.adapter_rank, target.input_width])?);
            parameters.push(math.runtime.zeros(
                &[target.output_width as i32, self.adapter_rank as i32],
                mlx_bridge::MlxDtype::Float32,
            )?);
        }
        Ok(parameters)
    }

    /// The scorer tensors by name and the add-ons, sharing `parameters`.
    pub(crate) fn trained_parts(
        &self,
        parameters: &[MlxArray],
    ) -> Result<TrainedPartArrays, MlxCError> {
        let scorer_count = self.scorer_layout.tensors.len();
        let mut tuned_tensors = BTreeMap::new();
        for (parameter_tensor, parameter) in self.scorer_layout.tensors.iter().zip(parameters) {
            tuned_tensors.insert(parameter_tensor.name.clone(), parameter.retain()?);
        }
        let mut target_names = Vec::with_capacity(self.adapter_targets.len());
        for target in &self.adapter_targets {
            target_names.push(target.name.clone());
        }
        let adapters =
            LinearAdapters::from_trainable_arrays(&target_names, &parameters[scorer_count..])?;
        Ok(TrainedPartArrays {
            tuned_tensors,
            adapters,
        })
    }

    /// The trained scorer tensors and the add-on tensors, from each
    /// parameter's values read back in order.
    pub(crate) fn exported_tensors(
        &self,
        parameter_values: Vec<Vec<f32>>,
    ) -> (Vec<TrainedTensor>, Vec<TrainedTensor>) {
        let scorer_count = self.scorer_layout.tensors.len();
        let mut scorer_tensors = Vec::with_capacity(scorer_count);
        let mut adapter_tensors = Vec::with_capacity(self.adapter_targets.len() * 2);
        let mut remaining_values = parameter_values.into_iter();
        for parameter_tensor in &self.scorer_layout.tensors {
            scorer_tensors.push(TrainedTensor {
                name: parameter_tensor.name.clone(),
                shape: parameter_tensor.shape.clone(),
                values: FloatVector(remaining_values.next().unwrap_or_default()),
            });
        }
        for target in &self.adapter_targets {
            adapter_tensors.push(TrainedTensor {
                name: format!("{}.adapter_down", target.name),
                shape: vec![self.adapter_rank, target.input_width],
                values: FloatVector(remaining_values.next().unwrap_or_default()),
            });
            adapter_tensors.push(TrainedTensor {
                name: format!("{}.adapter_up", target.name),
                shape: vec![target.output_width, self.adapter_rank],
                values: FloatVector(remaining_values.next().unwrap_or_default()),
            });
        }
        (scorer_tensors, adapter_tensors)
    }
}

/// `value_count` values spread evenly in `±1/√inputs`, the range a linear
/// layer of that width is usually started in, from a fixed sequence so a
/// training is reproducible.
fn starting_down_values(value_count: usize, input_width: usize, stream: u64) -> Vec<f32> {
    let bound = 1.0 / (input_width.max(1) as f64).sqrt();
    let mut sequence_state = stream
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let mut values = Vec::with_capacity(value_count);
    for _ in 0..value_count {
        sequence_state = sequence_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit_value = (sequence_state >> 40) as f64 / (1_u64 << 24) as f64;
        values.push(((unit_value * 2.0 - 1.0) * bound) as f32);
    }
    values
}

#[cfg(test)]
#[path = "whole_model_layout_tests.rs"]
mod tests;
