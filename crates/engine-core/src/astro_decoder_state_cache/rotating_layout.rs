use super::layout::{
    AstroDecoderStateCacheLayerLayout, AstroDecoderStateCacheTensorDtype,
    AstroDecoderStateCacheTensorLayout,
};

impl AstroDecoderStateCacheLayerLayout {
    /// Defines bounded rotating attention key/value state.
    #[must_use]
    pub fn rotating_window_attention(
        keys: AstroDecoderStateCacheTensorLayout,
        values: AstroDecoderStateCacheTensorLayout,
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
pub(super) fn rotating_window_counter_layouts() -> [AstroDecoderStateCacheTensorLayout; 2] {
    [
        AstroDecoderStateCacheTensorLayout::fixed(
            "attention.absolute_position",
            AstroDecoderStateCacheTensorDtype::Float32,
            vec![1],
        ),
        AstroDecoderStateCacheTensorLayout::fixed(
            "attention.ring_write_index",
            AstroDecoderStateCacheTensorDtype::Float32,
            vec![1],
        ),
    ]
}
