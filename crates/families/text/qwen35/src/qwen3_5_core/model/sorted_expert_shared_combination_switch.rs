//! Whether a loaded model combines a prompt chunk's routed experts and its
//! shared expert in one fused pass, decided once at load from the GPU's
//! capability verdict, and a switch for measurements.
//!
//! Both text engines keep the kernel: its inputs are the routed rows, so it
//! does not matter whether the experts were held in memory or streamed.

use mlx_bridge::MlxMetalKernel;
use text_generation::execution_error::TextExecutionError;

use super::base::Qwen3_5ModelBase;
use crate::qwen3_5_core::model_math::sorted_expert_shared_combination::qwen3_5_sorted_expert_shared_combination_kernel;

/// Keeps the fused sorted-expert and shared-expert combination when its
/// probe passed. It reads the sorted assignment layout, so it is kept only
/// alongside the sorted-reduction kernel. On the 35B-A3B 4-bit resident model
/// it measured +0.3 to +0.4% median prefill speed for single 2,048 to
/// 4,096-token chunks and +0.03% over a 10,176-token prompt, with identical
/// logits: a small, free gain, near run-to-run spread.
pub fn retained_sorted_expert_shared_combination_kernel(
    worker_kernel_capabilities: &crate::kernel_capability::WorkerKernelCapabilities,
    is_sorted_reduction_retained: bool,
) -> Result<Option<MlxMetalKernel>, TextExecutionError> {
    if !is_sorted_reduction_retained {
        return Ok(None);
    }
    let fused_combination_family = crate::kernel_capability::SORTED_EXPERT_SHARED_COMBINATION;
    if !worker_kernel_capabilities.is_custom_kernel_supported(fused_combination_family) {
        tracing::info!(
            verdict = ?worker_kernel_capabilities.verdict(fused_combination_family),
            "fused sorted-expert and shared-expert combination kernel demoted to the two-step path for this worker process"
        );
        return Ok(None);
    }
    Ok(Some(qwen3_5_sorted_expert_shared_combination_kernel()?))
}

/// For measurements: switches one loaded model between the fused
/// combination and the two-step path, so both can be compared on the same
/// weights without reloading. Serving never calls these.
impl Qwen3_5ModelBase {
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
