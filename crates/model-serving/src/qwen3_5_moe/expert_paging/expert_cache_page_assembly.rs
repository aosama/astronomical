//! Builds compact selected-expert tensors from retained one-expert pages.

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use super::expert_cache::ExpertWeightMemoryCache;
use super::expert_pager::{ExpertPagingError, PagedExpertWeights};
use crate::qwen3_5::model::decoder_layer_weights::Qwen3_5AffineWeights;

impl ExpertWeightMemoryCache {
    pub(crate) fn assemble_selected_experts(
        &self,
        runtime: &MlxRuntime,
        layer_index: usize,
        selected_expert_ids: &[usize],
    ) -> Result<PagedExpertWeights, ExpertPagingError> {
        let Some(&first_selected_expert_id) = selected_expert_ids.first() else {
            return Err(ExpertPagingError::Runtime {
                description: "cannot assemble an empty selected expert page".to_owned(),
            });
        };
        self.cached_expert(layer_index, first_selected_expert_id)
            .ok_or_else(|| missing_cached_expert_error(layer_index, first_selected_expert_id))?;
        let mut gate_projections = Vec::with_capacity(selected_expert_ids.len());
        let mut up_projections = Vec::with_capacity(selected_expert_ids.len());
        let mut down_projections = Vec::with_capacity(selected_expert_ids.len());

        for &selected_expert_id in selected_expert_ids {
            let cached_expert = self
                .cached_expert(layer_index, selected_expert_id)
                .ok_or_else(|| missing_cached_expert_error(layer_index, selected_expert_id))?;
            gate_projections.push(&cached_expert.paged_expert_weights.gate_projection);
            up_projections.push(&cached_expert.paged_expert_weights.up_projection);
            down_projections.push(&cached_expert.paged_expert_weights.down_projection);
        }

        Ok(PagedExpertWeights {
            gate_projection: concatenate_affine_projections(
                runtime,
                &gate_projections,
                "gate projection",
            )?,
            up_projection: concatenate_affine_projections(
                runtime,
                &up_projections,
                "up projection",
            )?,
            down_projection: concatenate_affine_projections(
                runtime,
                &down_projections,
                "down projection",
            )?,
            _metal_expert_pack_load_owner: None,
        })
    }
}

fn concatenate_affine_projections(
    runtime: &MlxRuntime,
    cached_projections: &[&Qwen3_5AffineWeights],
    projection_description: &str,
) -> Result<Qwen3_5AffineWeights, ExpertPagingError> {
    let Some(first_projection) = cached_projections.first() else {
        return Err(ExpertPagingError::Runtime {
            description: "cannot concatenate empty expert projections".to_owned(),
        });
    };
    match first_projection {
        Qwen3_5AffineWeights::NativeBfloat16 { .. } => {
            let native_bfloat16_weights = cached_projections
                .iter()
                .map(|projection| match projection {
                    Qwen3_5AffineWeights::NativeBfloat16 { weight } => Ok(weight),
                    Qwen3_5AffineWeights::Quantized { .. } => Err(ExpertPagingError::Runtime {
                        description: format!(
                            "{projection_description} pages use mixed storage formats"
                        ),
                    }),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Qwen3_5AffineWeights::NativeBfloat16 {
                weight: concatenate_cached_arrays(
                    runtime,
                    &native_bfloat16_weights,
                    projection_description,
                    "weights",
                )?,
            })
        }
        Qwen3_5AffineWeights::Quantized {
            quantization_bits,
            quantization_group_size,
            ..
        } => {
            let mut packed_weights = Vec::with_capacity(cached_projections.len());
            let mut quantization_scales = Vec::with_capacity(cached_projections.len());
            let mut quantization_biases = Vec::with_capacity(cached_projections.len());
            for projection in cached_projections {
                let Qwen3_5AffineWeights::Quantized {
                    packed_weight,
                    quantization_scales: scales,
                    quantization_biases: biases,
                    ..
                } = projection
                else {
                    return Err(ExpertPagingError::Runtime {
                        description: format!(
                            "{projection_description} pages use mixed storage formats"
                        ),
                    });
                };
                packed_weights.push(packed_weight);
                quantization_scales.push(scales);
                quantization_biases.push(biases);
            }
            Ok(Qwen3_5AffineWeights::Quantized {
                packed_weight: concatenate_cached_arrays(
                    runtime,
                    &packed_weights,
                    projection_description,
                    "weights",
                )?,
                quantization_scales: concatenate_cached_arrays(
                    runtime,
                    &quantization_scales,
                    projection_description,
                    "scales",
                )?,
                quantization_biases: concatenate_cached_arrays(
                    runtime,
                    &quantization_biases,
                    projection_description,
                    "biases",
                )?,
                quantization_bits: *quantization_bits,
                quantization_group_size: *quantization_group_size,
            })
        }
    }
}

fn missing_cached_expert_error(layer_index: usize, expert_id: usize) -> ExpertPagingError {
    ExpertPagingError::Runtime {
        description: format!("cached expert {expert_id} missing for layer {layer_index}"),
    }
}

fn concatenate_cached_arrays(
    runtime: &MlxRuntime,
    cached_arrays: &[&MlxArray],
    projection_description: &str,
    parameter_description: &str,
) -> Result<MlxArray, ExpertPagingError> {
    runtime
        .concatenate_axis(cached_arrays, 0)
        .map_err(|error| ExpertPagingError::Runtime {
            description: format!(
                "failed to concatenate cached {projection_description} {parameter_description}: {error}"
            ),
        })
}
