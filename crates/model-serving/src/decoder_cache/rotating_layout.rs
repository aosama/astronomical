use super::layout::{DecoderCacheLayerLayout, DecoderCacheTensorDtype, DecoderCacheTensorLayout};

impl DecoderCacheLayerLayout {
    /// Defines bounded rotating attention key/value state.
    #[must_use]
    pub fn rotating_window_attention(
        keys: DecoderCacheTensorLayout,
        values: DecoderCacheTensorLayout,
        window_size: usize,
    ) -> Self {
        Self::RotatingWindowAttention {
            keys,
            values,
            window_size,
        }
    }
}

/// Boundary tensors that persist rotating counters beside key/value slabs.
#[must_use]
pub(super) fn rotating_window_counter_layouts() -> [DecoderCacheTensorLayout; 2] {
    [
        DecoderCacheTensorLayout::fixed(
            "attention.absolute_position",
            DecoderCacheTensorDtype::Float32,
            vec![1],
        ),
        DecoderCacheTensorLayout::fixed(
            "attention.ring_write_index",
            DecoderCacheTensorDtype::Float32,
            vec![1],
        ),
    ]
}
