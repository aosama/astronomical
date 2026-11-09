//! How many decoder layers of a dense model stay in memory under a limit.
//!
//! A dense model that does not fit keeps a prefix of its decoder layers
//! resident and reads the rest from the SSD through a few reusable layer
//! slots (issue #1985). Every input is measured or read from the artifact:
//! the limit is what macOS granted or the user set, and working memory is
//! observed, so this policy adds no reserve or fraction of its own.
//!
//! The policy never refuses. When even the streaming floor (the always
//! resident tensors, the slots and working memory) is above the limit, the
//! plan keeps no layers resident and says so; MLX and macOS then decide,
//! because a slow model on a small Mac is acceptable and a refused one is not
//! (#1902).

/// What a dense model needs in memory beside its decoder layers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseLayerResidencyRequirements<'layers> {
    /// Tensors that never stream: the output head, the final norm, and the
    /// embedding table when it is resident.
    pub always_resident_bytes: u64,
    /// Every tensor's bytes of each decoder layer, in layer order; a streamed
    /// layer is read whole.
    pub decoder_layer_streamable_bytes: &'layers [u64],
    /// Reusable slots a streamed layer is read into; two let the next layer
    /// load while the current one computes.
    pub streaming_slot_count: u64,
    /// Measured working memory: activations, context state and transients.
    pub working_memory_bytes: u64,
    /// The effective MLX limit.
    pub active_memory_ceiling_bytes: u64,
}

/// The planned split between resident and streamed decoder layers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseLayerResidencyPlan {
    /// Layers `0..resident_layer_count` stay resident; the rest stream.
    pub resident_layer_count: usize,
    /// Bytes the streaming slots occupy; zero when nothing streams.
    pub streaming_slot_bytes: u64,
    /// What one streamed forward adds beyond the resident tensors: the slots
    /// and the working memory. Reported even when nothing streams, since
    /// giving a layer back makes the forward stream.
    pub streamed_forward_reserve_bytes: u64,
    /// The smallest limit that streams every layer: always resident tensors,
    /// slots and working memory.
    pub streaming_floor_bytes: u64,
    /// Whether the limit is below the streaming floor, so the plan streams
    /// everything and still exceeds the limit.
    pub is_below_streaming_floor: bool,
}

impl DenseLayerResidencyRequirements<'_> {
    /// Keeps every layer resident when the whole model fits; otherwise keeps
    /// the longest prefix of layers that fits beside the streaming slots.
    ///
    /// Streaming starts by paying for its slots, so a limit one byte short
    /// of complete residency keeps `streaming_slot_count + 1` fewer layers,
    /// not one fewer.
    #[must_use]
    pub fn plan(&self) -> DenseLayerResidencyPlan {
        let mut all_layer_bytes = 0_u64;
        let mut largest_layer_bytes = 0_u64;
        for layer_bytes in self.decoder_layer_streamable_bytes {
            all_layer_bytes = all_layer_bytes.saturating_add(*layer_bytes);
            largest_layer_bytes = largest_layer_bytes.max(*layer_bytes);
        }
        let complete_residency_bytes = self
            .always_resident_bytes
            .saturating_add(all_layer_bytes)
            .saturating_add(self.working_memory_bytes);
        let streaming_slot_bytes = largest_layer_bytes.saturating_mul(self.streaming_slot_count);
        let streamed_forward_reserve_bytes = self
            .working_memory_bytes
            .saturating_add(streaming_slot_bytes);
        let streaming_floor_bytes = self
            .always_resident_bytes
            .saturating_add(streamed_forward_reserve_bytes);
        if complete_residency_bytes <= self.active_memory_ceiling_bytes {
            return DenseLayerResidencyPlan {
                resident_layer_count: self.decoder_layer_streamable_bytes.len(),
                streaming_slot_bytes: 0,
                streamed_forward_reserve_bytes,
                streaming_floor_bytes,
                is_below_streaming_floor: false,
            };
        }
        let Some(mut remaining_layer_bytes) = self
            .active_memory_ceiling_bytes
            .checked_sub(streaming_floor_bytes)
        else {
            return DenseLayerResidencyPlan {
                resident_layer_count: 0,
                streaming_slot_bytes,
                streamed_forward_reserve_bytes,
                streaming_floor_bytes,
                is_below_streaming_floor: true,
            };
        };
        let mut resident_layer_count = 0;
        for layer_bytes in self.decoder_layer_streamable_bytes {
            let Some(bytes_left_after_layer) = remaining_layer_bytes.checked_sub(*layer_bytes)
            else {
                break;
            };
            remaining_layer_bytes = bytes_left_after_layer;
            resident_layer_count += 1;
        }
        DenseLayerResidencyPlan {
            resident_layer_count,
            streaming_slot_bytes,
            streamed_forward_reserve_bytes,
            streaming_floor_bytes,
            is_below_streaming_floor: false,
        }
    }
}
