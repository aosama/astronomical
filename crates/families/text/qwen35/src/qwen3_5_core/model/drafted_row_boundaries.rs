//! The collector a verify forward over a drafted block captures its row
//! boundaries with, shared by the resident and the streaming target.

use runtime_integration::MlxRuntimeError;
use text_generation::AstroSsdPromptCacheBoundaryCheckpointCollector;

use super::Qwen3_5ModelBase;

impl Qwen3_5ModelBase {
    /// Captures every linear-attention layer's state after each row of a
    /// `row_count`-row verify forward but the last, so the block can be
    /// rewound to the rows the target keeps without running them again.
    pub fn drafted_row_boundary_collector(
        &self,
        row_count: usize,
    ) -> Result<AstroSsdPromptCacheBoundaryCheckpointCollector, MlxRuntimeError> {
        AstroSsdPromptCacheBoundaryCheckpointCollector::after_every_row_but_the_last(
            row_count,
            self.astro_decoder_state_cache_layout
                .boundary_tensor_count(),
        )
    }
}
