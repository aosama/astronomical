//! Flattened persistence ownership for append-only, rotating, and recurrent state.

use super::layout::{
    DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCachePersistedTensorLayout,
};

impl DecoderCacheLayout {
    /// Returns deterministic append-only sequence tensors for one persisted block.
    #[must_use]
    pub fn sequence_tensor_layouts(&self) -> Vec<DecoderCachePersistedTensorLayout> {
        collect_layout_tensors(self, true)
    }

    /// Returns deterministic rotating and recurrent tensors for one boundary.
    #[must_use]
    pub fn boundary_tensor_layouts(&self) -> Vec<DecoderCachePersistedTensorLayout> {
        collect_layout_tensors(self, false)
    }
}

fn collect_layout_tensors(
    decoder_cache_layout: &DecoderCacheLayout,
    include_sequence_tensors: bool,
) -> Vec<DecoderCachePersistedTensorLayout> {
    let expected_tensor_count = if include_sequence_tensors {
        decoder_cache_layout.sequence_tensor_count()
    } else {
        decoder_cache_layout.boundary_tensor_count()
    };
    let mut persisted_tensor_layouts = Vec::with_capacity(expected_tensor_count);
    for decoder_layer_index in 0..decoder_cache_layout.layer_count() {
        if let Some(layer_layout) = decoder_cache_layout.layer(decoder_layer_index) {
            collect_layer_tensors(
                layer_layout,
                decoder_layer_index,
                include_sequence_tensors,
                &mut persisted_tensor_layouts,
            );
        }
    }
    persisted_tensor_layouts
}

fn collect_layer_tensors(
    layer_layout: &DecoderCacheLayerLayout,
    decoder_layer_index: usize,
    include_sequence_tensors: bool,
    persisted_tensor_layouts: &mut Vec<DecoderCachePersistedTensorLayout>,
) {
    match layer_layout {
        DecoderCacheLayerLayout::AppendOnlyAttention { keys, values, .. } => {
            if include_sequence_tensors {
                push_tensor(persisted_tensor_layouts, decoder_layer_index, keys.clone());
                push_tensor(
                    persisted_tensor_layouts,
                    decoder_layer_index,
                    values.clone(),
                );
            }
        }
        DecoderCacheLayerLayout::RotatingWindowAttention { keys, values, .. } => {
            if !include_sequence_tensors {
                push_tensor(persisted_tensor_layouts, decoder_layer_index, keys.clone());
                push_tensor(
                    persisted_tensor_layouts,
                    decoder_layer_index,
                    values.clone(),
                );
                for counter_layout in super::rotating_layout::rotating_window_counter_layouts() {
                    push_tensor(
                        persisted_tensor_layouts,
                        decoder_layer_index,
                        counter_layout,
                    );
                }
            }
        }
        DecoderCacheLayerLayout::RecurrentTensor { tensor } => {
            if !include_sequence_tensors {
                push_tensor(
                    persisted_tensor_layouts,
                    decoder_layer_index,
                    tensor.clone(),
                );
            }
        }
        DecoderCacheLayerLayout::Composite { components } => {
            for component_layout in components {
                collect_layer_tensors(
                    component_layout,
                    decoder_layer_index,
                    include_sequence_tensors,
                    persisted_tensor_layouts,
                );
            }
        }
    }
}

fn push_tensor(
    persisted_tensor_layouts: &mut Vec<DecoderCachePersistedTensorLayout>,
    decoder_layer_index: usize,
    tensor_layout: super::layout::DecoderCacheTensorLayout,
) {
    persisted_tensor_layouts.push(DecoderCachePersistedTensorLayout::new(
        decoder_layer_index,
        tensor_layout,
    ));
}
