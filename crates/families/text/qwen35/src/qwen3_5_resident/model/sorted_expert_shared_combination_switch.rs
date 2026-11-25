//! Switches one loaded resident model between the fused sorted-expert and
//! shared-expert combination and the two-step path, so a measurement can
//! compare both on the same resident weights without reloading. Serving never
//! calls these: the resident loader decides once from the capability probe.
//! The streaming engine's paged MoE forward never reads the fused kernel, so
//! the switch lives on the resident model only.

use crate::qwen3_5_core::model_math::sorted_expert_shared_combination::qwen3_5_sorted_expert_shared_combination_kernel;
use crate::qwen3_5_resident::model::Qwen3_5ResidentModel;
use text_generation::execution_error::TextExecutionError;

impl Qwen3_5ResidentModel {
    /// Routes prompt-chunk MoE combinations through the fused kernel. Only
    /// layers that already take the sorted reduction use it, because the
    /// fused kernel reads the sorted assignment layout.
    pub fn enable_fused_sorted_expert_shared_combination(
        &mut self,
    ) -> Result<(), TextExecutionError> {
        self.optional_accelerators
            .sorted_expert_shared_combination_kernel =
            Some(qwen3_5_sorted_expert_shared_combination_kernel()?);
        Ok(())
    }

    /// Restores the two-step sorted weighted sum plus compiled combination.
    pub fn disable_fused_sorted_expert_shared_combination(&mut self) {
        self.optional_accelerators
            .sorted_expert_shared_combination_kernel = None;
    }
}
