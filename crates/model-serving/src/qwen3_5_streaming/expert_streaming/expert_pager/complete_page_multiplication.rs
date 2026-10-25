//! Decides whether a forward may multiply a seated complete expert layer as-is.
//!
//! Affine-quantized and native bfloat16 pages feed `gather_qmm` / `gather_mm`
//! straight from the stacked table, so the full expert axis costs nothing
//! extra: the kernel only touches the routed rows. Remapping the route through
//! the table's slot map then stays on the graphics processor, which removes
//! the per-layer host copy of router indices and the routed-subset gather copy
//! that a seated layer would otherwise pay on every forward.
//!
//! Jangtq2 codebook pages are the exception: without the fused decode-token
//! kernel, every forward decodes a transient bfloat16 copy of the page it
//! multiplies, so handing it the complete table would decode every expert of
//! the layer. Those forwards keep the routed-subset path.

use super::Qwen3_5PagedExpertWeights;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;

impl Qwen3_5PagedExpertWeights {
    /// True when a `token_count`-token forward can multiply this complete page
    /// without materializing a decoded copy of its experts.
    pub(crate) fn multiplies_complete_page_without_transient_decode(
        &self,
        token_count: i32,
        has_fused_codebook_decode_kernel: bool,
    ) -> bool {
        [
            &self.gate_projection,
            &self.up_projection,
            &self.down_projection,
        ]
        .into_iter()
        .all(|projection| match projection {
            Qwen3_5AffineWeights::Quantized { .. }
            | Qwen3_5AffineWeights::NativeBfloat16 { .. } => true,
            Qwen3_5AffineWeights::Jangtq2Codebook { .. } => {
                codebook_page_multiplies_without_transient_decode(
                    token_count,
                    has_fused_codebook_decode_kernel,
                )
            }
        })
    }
}

/// The codebook rule shared by built pages and by planners that must decide
/// from layer geometry before a page exists: only a single decode token with
/// the fused kernel multiplies codebook codes without decoding them first.
#[must_use]
pub(crate) const fn codebook_page_multiplies_without_transient_decode(
    token_count: i32,
    has_fused_codebook_decode_kernel: bool,
) -> bool {
    token_count == 1 && has_fused_codebook_decode_kernel
}
