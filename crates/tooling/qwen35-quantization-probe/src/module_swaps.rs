//! Where the probe reaches into the chat model to swap one module's weights
//! and put them back: plain affine weights in the decoder layers, the
//! mixture-of-experts router, and a routed-expert projection across every
//! expert of a layer the model holds complete.

use quantization::bit_width::{BitWidth, GroupSize};
use quantization::error::QuantizationError;

use qwen35::qwen3_5_core::model_math::decoder_layer_weights::{
    Qwen3_5AffineWeights, Qwen3_5DecoderFeedForwardWeights,
};
use qwen35::qwen3_5_core::model_math::feed_forward_weights::{
    Qwen3_5MoEFeedForwardWeights, Qwen3_5MoERouterGateWeights,
};

use qwen35::RoutedExpertProjection;

use super::super::expert_swaps::{SwappedOutWeights, router_as_affine_weights};
use super::super::module_locations::{
    FeedForwardProjection, ModuleLocation, affine_weights_at_mut,
};
use super::super::simulated_quantization::mlx_step;
use super::{Qwen3_5SensitivityProbe, probe_step, tied_head_key, unresolvable};

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

    /// The affine weights a module name resolves to in the decoder layers,
    /// mutably.
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
        match affine_weights_at_mut(&mut self.model.base.weights, location) {
            Some(affine_weights) => Ok(affine_weights),
            None => Err(unresolvable(
                module_name,
                "the layer does not hold this projection as affine weights",
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
            self.swap_tied_head(module_name)?;
        }
        Ok(())
    }

    /// Points the head tied to the swapped embedding at its replacement too.
    fn swap_tied_head(&mut self, module_name: &str) -> Result<(), QuantizationError> {
        let head_replacement = mlx_step(
            self.affine_weights_mut(module_name)?.retained_reference(),
            "retain the tied head",
        )?;
        let head_slot = self
            .model
            .base
            .weights
            .language_model_head_weights
            .bound_mut()
            .ok_or(unresolvable(
                module_name,
                "the tied head was converted for the register-ordered matmul",
            ))?;
        let previous_head = std::mem::replace(head_slot, head_replacement);
        self.swapped_out_weights.insert(
            tied_head_key(module_name),
            SwappedOutWeights::Affine(previous_head),
        );
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

    /// Swaps a routed-expert projection, across every expert of the layer,
    /// for a copy at the probed width. A model whose experts did not fit in
    /// memory reads them from the SSD and holds no complete layer; its
    /// experts cannot be swapped, and the sweep counts them as skipped.
    pub(super) fn swap_routed_expert_projection(
        &mut self,
        module_name: &str,
        decoder_layer_index: usize,
        projection: FeedForwardProjection,
        bit_width: BitWidth,
        group_size: GroupSize,
    ) -> Result<(), QuantizationError> {
        let routed_projection = routed_expert_projection_of(projection);
        let current_weights = self
            .model
            .routed_expert_projection(decoder_layer_index, routed_projection)
            .or(Err(unresolvable(module_name, NOT_HELD_COMPLETE)))?;
        // A codebook projection comes with its activation rotation in the
        // table; an affine replacement would run under the wrong rotation.
        if matches!(
            current_weights,
            Qwen3_5AffineWeights::Jangtq2Codebook { .. }
        ) {
            return Err(unresolvable(
                module_name,
                "codebook-packed routed experts cannot be probed",
            ));
        }
        let replacement =
            self.replacement_weights(module_name, &current_weights, bit_width, group_size)?;
        let previous = self.replace_routed_expert_projection(
            module_name,
            decoder_layer_index,
            routed_projection,
            replacement,
        )?;
        self.swapped_out_weights.insert(
            module_name.to_owned(),
            SwappedOutWeights::RoutedExpert(previous),
        );
        Ok(())
    }

    fn replace_routed_expert_projection(
        &self,
        module_name: &str,
        decoder_layer_index: usize,
        routed_projection: RoutedExpertProjection,
        replacement: Qwen3_5AffineWeights,
    ) -> Result<Qwen3_5AffineWeights, QuantizationError> {
        probe_step(self.model.replace_routed_expert_projection(
            decoder_layer_index,
            routed_projection,
            replacement,
        ))
        .or(Err(unresolvable(module_name, NOT_HELD_COMPLETE)))
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
                SwappedOutWeights::RoutedExpert(previous),
                ModuleLocation::DecoderLayerRoutedExperts {
                    decoder_layer_index,
                    projection,
                },
            ) => {
                self.replace_routed_expert_projection(
                    module_name,
                    decoder_layer_index,
                    routed_expert_projection_of(projection),
                    previous,
                )?;
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

/// Why a routed-expert projection cannot be swapped.
const NOT_HELD_COMPLETE: &str = "its routed experts are not held complete in memory";

/// The chat model's name for a routed expert's projection.
const fn routed_expert_projection_of(projection: FeedForwardProjection) -> RoutedExpertProjection {
    match projection {
        FeedForwardProjection::Gate => RoutedExpertProjection::Gate,
        FeedForwardProjection::Up => RoutedExpertProjection::Up,
        FeedForwardProjection::Down => RoutedExpertProjection::Down,
    }
}
