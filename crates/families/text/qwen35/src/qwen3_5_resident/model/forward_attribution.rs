//! The in-memory model's part of the prefill forward the sensitivity probe
//! runs: how a prompt chunk's decoder state is evaluated. The prefill
//! forwards themselves are shared, in `qwen3_5_core::model::prefill_forward`.

use crate::qwen3_5_core::model::prefill_forward::Qwen3_5PrefillForward;

use crate::PerformanceAttribution;

use super::Qwen3_5ResidentModel;
use super::{RequestDecoderStateStack, TextExecutionError};
use text_generation::AstroSsdPromptCacheBoundaryCheckpointCollector;

impl Qwen3_5PrefillForward for Qwen3_5ResidentModel {
    const PRODUCTION_FORWARD_MODE: () = ();

    fn evaluate_prefill_decoder_state(
        &self,
        request_decoder_state: &RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<&AstroSsdPromptCacheBoundaryCheckpointCollector>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), TextExecutionError> {
        self.evaluate_decoder_state_with_performance_attribution(
            request_decoder_state,
            boundary_checkpoint_collector,
            performance_attribution,
        )
    }
}
