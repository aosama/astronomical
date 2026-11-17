//! Recomputing each decoder layer during the backward pass, for a training
//! pass too long to keep every layer's activations.
//!
//! Training batches questions into passes under a memory budget, but a
//! single text longer than the budget still forms a pass of its own. Kept
//! whole, its activations grow with its length, about 16 MB a token on
//! Clef's 9B (`cargo test-clef-whole-model-pass-memory`), and a long text
//! ran a 48 GB Mac out of memory (issue #2299). Through MLX's checkpoint
//! transform each layer keeps only its input and runs again when its
//! gradient is needed: about a third more arithmetic for a small fraction of
//! the memory. Measured on the same over-budget text, the trained weights
//! were bit-identical and the peak fell from 15.74 to 11.02 GB on Clef 9B
//! (`cargo test-clef-whole-model-layer-recomputation`).
//!
//! MLX differentiates a checkpointed function only through its explicit
//! inputs, so each layer takes its hidden states and its own add-on
//! weights as inputs, and puts exactly those add-ons in force while it
//! runs; the add-ons in force before are restored afterwards, also when the
//! layer runs again during the backward pass.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;

use mlx_bridge::{MlxArray, MlxArrayVector, MlxCError, MlxClosure};

use crate::decision_models::decision_adapters::AdapterTarget;
use crate::decision_models::forward_pass_planning::pass_exceeds_budget;
use crate::decision_models::qwen_backbone::HoldsDecisionBackbone;
use crate::linear_adapters::LinearAdapters;
use crate::qwen3_5_core::model::layer_additions::Qwen3_5LayerAdditions;
use crate::qwen3_5_core::model_math::differentiable_gated_delta::DifferentiableGatedDelta;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_resident::{DecoderLayerRecomputation, Qwen3_5ResidentModel};

/// Every decoder layer of a decision model's backbone, recomputed during
/// the backward pass.
pub(crate) struct CheckpointedDecoderLayers {
    loaded_state: Rc<dyn HoldsDecisionBackbone>,
    /// Each layer's add-on targets, by name, in forward order.
    layer_targets: Vec<Vec<String>>,
}

impl CheckpointedDecoderLayers {
    /// Recomputation for the backbone `loaded_state` holds, whose layers
    /// carry the add-ons `targets_by_layer` lists.
    pub(crate) fn new(
        loaded_state: Rc<dyn HoldsDecisionBackbone>,
        targets_by_layer: &[Vec<AdapterTarget>],
    ) -> Self {
        let mut layer_targets = Vec::with_capacity(targets_by_layer.len());
        for targets in targets_by_layer {
            let mut target_names = Vec::with_capacity(targets.len());
            for target in targets {
                target_names.push(target.name.clone());
            }
            layer_targets.push(target_names);
        }
        Self {
            loaded_state,
            layer_targets,
        }
    }

    fn checkpointed_layer(
        &self,
        hidden_states: &MlxArray,
        layer_index: usize,
        token_count: i32,
    ) -> Result<MlxArray, MlxCError> {
        let model = &self.loaded_state.decision_backbone().model;
        let Some(outer_additions) = model.layer_additions_in_force() else {
            return Err(recomputation_failure(
                "no add-ons are in force while a training builds its graph",
            ));
        };
        let Some(layer_targets) = self.layer_targets.get(layer_index) else {
            return Err(recomputation_failure(
                "a layer index is past the model's layers",
            ));
        };
        let mut layer_inputs = vec![hidden_states.retain()?];
        layer_inputs.extend(outer_additions.adapters.arrays_of(layer_targets)?);
        let layer_run = LayerRun {
            loaded_state: Rc::clone(&self.loaded_state),
            layer_targets: layer_targets.clone(),
            differentiable_recurrence: outer_additions.differentiable_recurrence.clone(),
            layer_index,
            token_count,
        };
        // MLX's checkpoint transform takes a function, not a graph.
        let checkpointed = MlxClosure::from_fn(move |closure_inputs: &[MlxArray]| {
            Ok(vec![layer_run.run(closure_inputs)?])
        })?
        .checkpointed()?;
        let mut borrowed_inputs = Vec::with_capacity(layer_inputs.len());
        for layer_input in &layer_inputs {
            borrowed_inputs.push(layer_input);
        }
        let outputs = checkpointed.apply(&MlxArrayVector::new(&borrowed_inputs)?)?;
        outputs.array_at(0, "read a recomputed layer's output")
    }
}

impl DecoderLayerRecomputation for CheckpointedDecoderLayers {
    fn recomputed_layer(
        &self,
        hidden_states: &MlxArray,
        layer_index: usize,
        token_count: i32,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        self.checkpointed_layer(hidden_states, layer_index, token_count)
            .map_err(Qwen3_5ExecutionError::from)
    }
}

/// The texts a training recomputed, each counted once however many steps,
/// passes and seeds ran it: what the report tells the person.
#[derive(Default)]
pub(crate) struct RecomputedTexts {
    text_indices: RefCell<BTreeSet<usize>>,
}

impl RecomputedTexts {
    pub(crate) fn record(&self, text_index: usize) {
        self.text_indices.borrow_mut().insert(text_index);
    }

    pub(crate) fn count(&self) -> usize {
        self.text_indices.borrow().len()
    }
}

/// Which of a training's passes recompute their layers, and which texts did.
pub(crate) struct PassRecomputation {
    checkpointed_layers: Rc<CheckpointedDecoderLayers>,
    recomputed_texts: RecomputedTexts,
    /// Padded tokens above which a pass recomputes
    /// (`recomputation_threshold_tokens`).
    threshold_tokens: usize,
}

impl PassRecomputation {
    /// Recomputation for the backbone `loaded_state` holds, whose layers
    /// carry the add-ons `targets_by_layer` lists, for passes over
    /// `threshold_tokens` padded tokens.
    pub(crate) fn new(
        loaded_state: Rc<dyn HoldsDecisionBackbone>,
        targets_by_layer: &[Vec<AdapterTarget>],
        threshold_tokens: usize,
    ) -> Self {
        Self {
            checkpointed_layers: Rc::new(CheckpointedDecoderLayers::new(
                loaded_state,
                targets_by_layer,
            )),
            recomputed_texts: RecomputedTexts::default(),
            threshold_tokens,
        }
    }

    /// The layers to recompute `forward_pass` with when it holds more padded
    /// tokens than the threshold, recording its text (`text_indices` names
    /// each sequence's text); `None` keeps every layer's activations, as
    /// every pass did before.
    pub(crate) fn for_pass(
        &self,
        forward_pass: &[usize],
        sequence_lengths: &[usize],
        text_indices: &[usize],
    ) -> Option<Rc<CheckpointedDecoderLayers>> {
        if !pass_exceeds_budget(forward_pass, sequence_lengths, self.threshold_tokens) {
            return None;
        }
        for sequence_index in forward_pass {
            self.recomputed_texts.record(text_indices[*sequence_index]);
        }
        Some(Rc::clone(&self.checkpointed_layers))
    }

    /// How many different texts were recomputed so far.
    pub(crate) fn recomputed_text_count(&self) -> usize {
        self.recomputed_texts.count()
    }
}

/// `layers` as the forward pass takes them.
pub(crate) fn as_layer_recomputation(
    layers: Option<&Rc<CheckpointedDecoderLayers>>,
) -> Option<&dyn DecoderLayerRecomputation> {
    match layers {
        Some(layers) => Some(layers.as_ref()),
        None => None,
    }
}

/// What one checkpointed layer needs each time MLX runs it.
struct LayerRun {
    loaded_state: Rc<dyn HoldsDecisionBackbone>,
    layer_targets: Vec<String>,
    differentiable_recurrence: Option<Rc<DifferentiableGatedDelta>>,
    layer_index: usize,
    token_count: i32,
}

impl LayerRun {
    /// The layer over `closure_inputs` (hidden states, then this layer's
    /// add-ons as `down, up` per target), with exactly those add-ons in
    /// force; whatever was in force before is put back.
    fn run(&self, closure_inputs: &[MlxArray]) -> Result<MlxArray, MlxCError> {
        let Some((hidden_states, adapter_arrays)) = closure_inputs.split_first() else {
            return Err(recomputation_failure("a recomputed layer got no inputs"));
        };
        check_adapter_input_count(adapter_arrays.len(), self.layer_targets.len())?;
        let model = &self.loaded_state.decision_backbone().model;
        let layer_adapters =
            LinearAdapters::from_trainable_arrays(&self.layer_targets, adapter_arrays)?;
        let _restores_outer_additions = OuterAdditionsRestorer {
            model,
            outer_additions: model.layer_additions_in_force(),
        };
        model.set_layer_additions(Some(Rc::new(Qwen3_5LayerAdditions {
            adapters: layer_adapters,
            differentiable_recurrence: self.differentiable_recurrence.clone(),
        })));
        model
            .empty_state_decoder_layer(hidden_states, self.layer_index, self.token_count)
            .map_err(layer_failure)
    }
}

/// Puts the add-ons that were in force back when a recomputed layer ends,
/// however it ends; otherwise the next layer would run with this one's.
struct OuterAdditionsRestorer<'model> {
    model: &'model Qwen3_5ResidentModel,
    outer_additions: Option<Rc<Qwen3_5LayerAdditions>>,
}

impl Drop for OuterAdditionsRestorer<'_> {
    fn drop(&mut self) {
        self.model.set_layer_additions(self.outer_additions.take());
    }
}

/// A layer's add-on inputs come as `down, up` for each of its targets; any
/// other count would pair arrays with the wrong targets, or leave some out,
/// and give wrong gradients without an error.
fn check_adapter_input_count(
    adapter_array_count: usize,
    target_count: usize,
) -> Result<(), MlxCError> {
    if adapter_array_count == 2 * target_count {
        return Ok(());
    }
    Err(recomputation_failure(&format!(
        "a recomputed layer got {adapter_array_count} add-on arrays for {target_count} targets"
    )))
}

fn layer_failure(layer_error: Qwen3_5ExecutionError) -> MlxCError {
    recomputation_failure(&layer_error.to_string())
}

fn recomputation_failure(description: &str) -> MlxCError {
    MlxCError {
        operation: "recompute a decoder layer",
        description: description.to_owned(),
    }
}

#[cfg(test)]
#[path = "layer_recomputation_tests.rs"]
mod tests;
