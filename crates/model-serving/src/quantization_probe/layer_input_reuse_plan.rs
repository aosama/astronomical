//! The arithmetic behind the sweep's layer-input reuse: which decoder layer
//! a swapped module first affects, how many layer inputs the reference
//! forward keeps, and which kept input a probe resumes from. Pure, so the
//! rules are unit-tested without a model.

use super::module_locations::ModuleLocation;

#[cfg(test)]
#[path = "layer_input_reuse_plan_tests.rs"]
mod layer_input_reuse_plan_tests;

/// The recorded layer inputs may take at most one part in this many of what
/// the sweep already holds as reference logits. The logits are the sweep's
/// dominant standing cost (sequence length x vocabulary per calibration
/// sequence), so tying the records to a quarter of them keeps the reuse from
/// changing which models fit, on any machine, without naming a byte figure.
const REFERENCE_LOGITS_PARTS_PER_RECORDED_INPUT_PART: usize = 4;

/// Every how many layers the reference forward keeps a layer input. Both the
/// logits and one layer input have a row per token, so their sizes compare
/// as vocabulary size to hidden size: keeping every layer costs
/// `layer_count * hidden_size` against the logits' `vocabulary_size`. A
/// small base keeps every layer; a wide, deep one keeps every second or
/// third, and a probe then recomputes at most that many unchanged layers.
#[must_use]
pub(crate) fn recording_layer_stride(
    decoder_layer_count: usize,
    hidden_size: usize,
    vocabulary_size: usize,
) -> usize {
    let every_layer_cost =
        REFERENCE_LOGITS_PARTS_PER_RECORDED_INPUT_PART * decoder_layer_count * hidden_size;
    every_layer_cost.div_ceil(vocabulary_size.max(1)).max(1)
}

/// The first decoder layer whose output a swap at `location` changes, or
/// `None` when the swap changes the rows entering the stack itself. A swap
/// of the head changes no layer, which is `decoder_layer_count`: the probe
/// resumes after the last layer and runs only the head.
#[must_use]
pub(crate) const fn first_affected_decoder_layer(
    location: ModuleLocation,
    decoder_layer_count: usize,
) -> Option<usize> {
    match location {
        ModuleLocation::Embedding => None,
        ModuleLocation::LanguageModelHead => Some(decoder_layer_count),
        ModuleLocation::DecoderLayerAttention {
            decoder_layer_index,
            ..
        }
        | ModuleLocation::DecoderLayerDenseFeedForward {
            decoder_layer_index,
            ..
        }
        | ModuleLocation::DecoderLayerRoutedExperts {
            decoder_layer_index,
            ..
        }
        | ModuleLocation::DecoderLayerSharedExpert {
            decoder_layer_index,
            ..
        }
        | ModuleLocation::DecoderLayerSharedExpertOutputGate {
            decoder_layer_index,
        }
        | ModuleLocation::DecoderLayerRouter {
            decoder_layer_index,
        } => Some(decoder_layer_index),
    }
}

/// The position, in ascending `recorded_layer_indices`, of the latest record
/// at or before `first_affected_layer`: the record a probe resumes from.
#[must_use]
pub(crate) fn resume_record_position(
    recorded_layer_indices: &[usize],
    first_affected_layer: usize,
) -> Option<usize> {
    let mut resume_position = None;
    for (record_position, recorded_layer_index) in recorded_layer_indices.iter().enumerate() {
        if *recorded_layer_index <= first_affected_layer {
            resume_position = Some(record_position);
        }
    }
    resume_position
}
