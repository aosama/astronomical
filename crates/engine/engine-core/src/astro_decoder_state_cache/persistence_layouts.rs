//! Flattened persistence ownership for append-only, rotating, and recurrent state.

use super::layout::{
    AstroDecoderStateCacheLayerLayout, AstroDecoderStateCacheLayout,
    AstroDecoderStateCachePersistedTensorLayout,
};

impl AstroDecoderStateCacheLayout {
    /// Returns deterministic append-only sequence tensors for one persisted block.
    #[must_use]
    pub fn sequence_tensor_layouts(&self) -> Vec<AstroDecoderStateCachePersistedTensorLayout> {
        collect_layout_tensors(self, true)
    }

    /// Returns deterministic rotating and recurrent tensors for one boundary.
    #[must_use]
    pub fn boundary_tensor_layouts(&self) -> Vec<AstroDecoderStateCachePersistedTensorLayout> {
        collect_layout_tensors(self, false)
    }
}

fn collect_layout_tensors(
    astro_decoder_state_cache_layout: &AstroDecoderStateCacheLayout,
    include_sequence_tensors: bool,
) -> Vec<AstroDecoderStateCachePersistedTensorLayout> {
    let expected_tensor_count = if include_sequence_tensors {
        astro_decoder_state_cache_layout.sequence_tensor_count()
    } else {
        astro_decoder_state_cache_layout.boundary_tensor_count()
    };
    let mut persisted_tensor_layouts = Vec::with_capacity(expected_tensor_count);
    for decoder_layer_index in 0..astro_decoder_state_cache_layout.layer_count() {
        if let Some(layer_layout) = astro_decoder_state_cache_layout.layer(decoder_layer_index) {
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
    layer_layout: &AstroDecoderStateCacheLayerLayout,
    decoder_layer_index: usize,
    include_sequence_tensors: bool,
    persisted_tensor_layouts: &mut Vec<AstroDecoderStateCachePersistedTensorLayout>,
) {
    match layer_layout {
        AstroDecoderStateCacheLayerLayout::AppendOnlyAttention { keys, values, .. } => {
            if include_sequence_tensors {
                push_tensor(persisted_tensor_layouts, decoder_layer_index, keys.clone());
                push_tensor(
                    persisted_tensor_layouts,
                    decoder_layer_index,
                    values.clone(),
                );
            }
        }
        AstroDecoderStateCacheLayerLayout::RotatingWindowAttention { keys, values, .. } => {
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
        AstroDecoderStateCacheLayerLayout::RecurrentTensor { tensor } => {
            if !include_sequence_tensors {
                push_tensor(
                    persisted_tensor_layouts,
                    decoder_layer_index,
                    tensor.clone(),
                );
            }
        }
        AstroDecoderStateCacheLayerLayout::Composite { components } => {
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
    persisted_tensor_layouts: &mut Vec<AstroDecoderStateCachePersistedTensorLayout>,
    decoder_layer_index: usize,
    tensor_layout: super::layout::AstroDecoderStateCacheTensorLayout,
) {
    persisted_tensor_layouts.push(AstroDecoderStateCachePersistedTensorLayout::new(
        decoder_layer_index,
        tensor_layout,
    ));
}
