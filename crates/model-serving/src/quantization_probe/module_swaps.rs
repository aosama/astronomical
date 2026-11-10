//! Where the probe reaches into the resident model to swap one module's
//! weights and put them back: plain affine weights in the decoder layers,
//! the mixture-of-experts router, and routed-expert projections held by the
//! resident expert owner.

use astronomical_quantization::bit_width::{BitWidth, GroupSize};
use astronomical_quantization::error::QuantizationError;

use crate::qwen3_5_core::model_math::decoder_layer_weights::{
    Qwen3_5AffineWeights, Qwen3_5DecoderFeedForwardWeights,
};
use crate::qwen3_5_core::model_math::feed_forward_weights::{
    Qwen3_5MoEFeedForwardWeights, Qwen3_5MoERouterGateWeights,
};
use crate::qwen3_5_resident::experts::{
    Qwen3_5ResidentExpertLayerWeights, Qwen3_5ResidentGateUpWeights,
};

use super::super::expert_swaps::{
    PROBE_SEPARATION_REASON, SwappedOutWeights, router_as_affine_weights,
    separate_gate_and_up_projections,
};
use super::super::module_locations::{
    FeedForwardProjection, ModuleLocation, affine_weights_at_mut,
};
use super::super::simulated_quantization::mlx_step;
use super::{Qwen3_5SensitivityProbe, tied_head_key, unresolvable};

impl Qwen3_5SensitivityProbe {
    /// Where a module lives, or a failure the sweep counts as a skip.
    pub(super) fn location_of(
        &self,
        module_name: &str,
    ) -> Result<ModuleLocation, QuantizationError> {
        match self.module_locations.get(module_name).copied() {
            Some(location) => Ok(location),
            None => Err(unresolvable(
                module_name,
                "it has no location in the Qwen3.5 weights",
            )),
        }
    }

    /// The affine weights a module name resolves to, mutably: a module in
    /// the decoder layers, or a routed-expert down projection, which the
    /// resident expert owner holds as an ordinary affine weight.
    pub(super) fn affine_weights_mut(
        &mut self,
        module_name: &str,
    ) -> Result<&mut Qwen3_5AffineWeights, QuantizationError> {
        let location = self.location_of(module_name)?;
        self.affine_weights_at(module_name, location)
    }

    /// The affine weights at a resolved location.
    fn affine_weights_at(
        &mut self,
        module_name: &str,
        location: ModuleLocation,
    ) -> Result<&mut Qwen3_5AffineWeights, QuantizationError> {
        if let ModuleLocation::DecoderLayerRoutedExperts {
            decoder_layer_index,
            projection: FeedForwardProjection::Down,
        } = location
        {
            let expert_layer = self.resident_expert_layer_mut(module_name, decoder_layer_index)?;
            return Ok(&mut expert_layer.down_projection);
        }
        match affine_weights_at_mut(&mut self.model.base.weights, location) {
            Some(affine_weights) => Ok(affine_weights),
            None => Err(unresolvable(
                module_name,
                "the layer does not hold this projection as affine weights",
            )),
        }
    }

    /// One layer's resident routed-expert arrays. A model whose experts did
    /// not fit in memory pages them from disk and holds no resident owner;
    /// its experts cannot be swapped, and the sweep's guard reports it.
    pub(super) fn resident_expert_layer_mut(
        &mut self,
        module_name: &str,
        decoder_layer_index: usize,
    ) -> Result<&mut Qwen3_5ResidentExpertLayerWeights, QuantizationError> {
        let resident_layer = match self.model.resident_expert_weights.as_mut() {
            Some(resident_expert_weights) => resident_expert_weights.layer_mut(decoder_layer_index),
            None => None,
        };
        match resident_layer {
            Some(resident_layer) => Ok(resident_layer),
            None => Err(unresolvable(
                module_name,
                "its routed experts are not resident in memory",
            )),
        }
    }

    /// One layer's router and shared-expert weights.
    pub(super) fn mixture_of_experts_weights_mut(
        &mut self,
        module_name: &str,
        decoder_layer_index: usize,
    ) -> Result<&mut Qwen3_5MoEFeedForwardWeights, QuantizationError> {
        let layer_weights = self
            .model
            .base
            .weights
            .decoder_layer_weights
            .get_mut(decoder_layer_index);
        match layer_weights {
            Some(layer_weights) => match &mut layer_weights.mlp_weights {
                Qwen3_5DecoderFeedForwardWeights::MixtureOfExperts(mixture_of_experts_weights) => {
                    Ok(mixture_of_experts_weights)
                }
                Qwen3_5DecoderFeedForwardWeights::Dense(_) => Err(unresolvable(
                    module_name,
                    "the layer has a dense feed-forward, not a mixture of experts",
                )),
            },
            None => Err(unresolvable(module_name, "the model has no such layer")),
        }
    }

    /// Swaps a module held as plain affine weights, and the tied head with
    /// an embedding.
    pub(super) fn swap_affine_module(
        &mut self,
        module_name: &str,
        bit_width: BitWidth,
        group_size: GroupSize,
    ) -> Result<(), QuantizationError> {
        let current_weights = mlx_step(
            self.affine_weights_mut(module_name)?.retained_reference(),
            "retain the module",
        )?;
        let replacement =
            self.replacement_weights(module_name, &current_weights, bit_width, group_size)?;
        let previous = std::mem::replace(self.affine_weights_mut(module_name)?, replacement);
        self.swapped_out_weights
            .insert(module_name.to_owned(), SwappedOutWeights::Affine(previous));
        if self.ties_head_to(module_name) {
            let head_replacement = mlx_step(
                self.affine_weights_mut(module_name)?.retained_reference(),
                "retain the tied head",
            )?;
            let previous_head = std::mem::replace(
                &mut self.model.base.weights.language_model_head_weights,
                head_replacement,
            );
            self.swapped_out_weights.insert(
                tied_head_key(module_name),
                SwappedOutWeights::Affine(previous_head),
            );
        }
        Ok(())
    }

    /// Swaps the router for a copy at the probed width.
    pub(super) fn swap_router(
        &mut self,
        module_name: &str,
        decoder_layer_index: usize,
        bit_width: BitWidth,
        group_size: GroupSize,
    ) -> Result<(), QuantizationError> {
        let current_router = router_as_affine_weights(
            &self
                .mixture_of_experts_weights_mut(module_name, decoder_layer_index)?
                .router_projection,
        )?;
        let replacement =
            self.replacement_weights(module_name, &current_router, bit_width, group_size)?;
        let mixture_of_experts_weights =
            self.mixture_of_experts_weights_mut(module_name, decoder_layer_index)?;
        let previous = std::mem::replace(
            &mut mixture_of_experts_weights.router_projection,
            Qwen3_5MoERouterGateWeights::Affine(replacement),
        );
        self.swapped_out_weights
            .insert(module_name.to_owned(), SwappedOutWeights::Router(previous));
        Ok(())
    }

    /// Swaps a routed-expert gate or up projection for a copy at the probed
    /// width, holding the pair apart for the probe (see `expert_swaps`).
    pub(super) fn swap_expert_gate_or_up(
        &mut self,
        module_name: &str,
        decoder_layer_index: usize,
        projection: FeedForwardProjection,
        bit_width: BitWidth,
        group_size: GroupSize,
    ) -> Result<(), QuantizationError> {
        let resident_layer = match &self.model.resident_expert_weights {
            Some(resident_expert_weights) => resident_expert_weights.layer(decoder_layer_index),
            None => None,
        };
        let Some(resident_layer) = resident_layer else {
            return Err(unresolvable(
                module_name,
                "its routed experts are not resident in memory",
            ));
        };
        let (current_gate, current_up) = separate_gate_and_up_projections(
            self.model.runtime(),
            &resident_layer.gate_up_weights,
        )?;
        let (gate_projection, up_projection) = if projection == FeedForwardProjection::Gate {
            let replacement =
                self.replacement_weights(module_name, &current_gate, bit_width, group_size)?;
            (replacement, current_up)
        } else {
            let replacement =
                self.replacement_weights(module_name, &current_up, bit_width, group_size)?;
            (current_gate, replacement)
        };
        let expert_layer = self.resident_expert_layer_mut(module_name, decoder_layer_index)?;
        let previous = std::mem::replace(
            &mut expert_layer.gate_up_weights,
            Qwen3_5ResidentGateUpWeights::Separate {
                gate_projection,
                up_projection,
                incompatibility_reason: PROBE_SEPARATION_REASON,
            },
        );
        self.swapped_out_weights.insert(
            module_name.to_owned(),
            SwappedOutWeights::ExpertGateUp(previous),
        );
        Ok(())
    }

    /// Puts back what a swap took out.
    pub(super) fn restore_swapped_out(
        &mut self,
        module_name: &str,
        swapped_out: SwappedOutWeights,
    ) -> Result<(), QuantizationError> {
        let location = self.location_of(module_name)?;
        match (swapped_out, location) {
            (SwappedOutWeights::Affine(previous), _) => {
                *self.affine_weights_mut(module_name)? = previous;
            }
            (
                SwappedOutWeights::Router(previous),
                ModuleLocation::DecoderLayerRouter {
                    decoder_layer_index,
                },
            ) => {
                self.mixture_of_experts_weights_mut(module_name, decoder_layer_index)?
                    .router_projection = previous;
            }
            (
                SwappedOutWeights::ExpertGateUp(previous),
                ModuleLocation::DecoderLayerRoutedExperts {
                    decoder_layer_index,
                    ..
                },
            ) => {
                self.resident_expert_layer_mut(module_name, decoder_layer_index)?
                    .gate_up_weights = previous;
            }
            _ => {
                return Err(unresolvable(
                    module_name,
                    "its swapped-out weights do not match where it lives",
                ));
            }
        }
        Ok(())
    }
}
