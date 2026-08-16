//! Laguna-owned Mixture-of-Experts routing and resident execution.
//!
//! Routers, correction bias, shared-expert combination, and score scaling stay
//! in this package. Neutral stacked-assignment sort and weighted reduction are
//! imported from `sparse_experts`. This package must not import Qwen.

mod router_selection;

#[cfg(feature = "direct-mlx")]
mod paged;
#[cfg(feature = "direct-mlx")]
mod resident;
#[cfg(feature = "direct-mlx")]
mod router;

pub use router_selection::{
    LagunaRouterSelection, apply_router_logit_softcap, select_laguna_router_experts,
};

#[cfg(feature = "direct-mlx")]
pub(in crate::laguna) use paged::{
    execute_paged_mixture_on_page, forward_paged_mixture_of_experts,
    forward_retained_complete_mixture_of_experts, route_laguna_layer_experts,
    unique_routed_expert_ids,
};
#[cfg(feature = "direct-mlx")]
pub(in crate::laguna) use resident::forward_resident_mixture_of_experts;

#[cfg(feature = "direct-mlx")]
mod public_router {
    use astronomical_runtime_integration::{MlxArray, MlxRuntime};

    use crate::laguna::normalization::LagunaMoeDescriptor;
    use crate::performance_attribution::PerformanceAttribution;

    use super::super::model::LagunaExecutionError;
    use super::router::route_laguna_experts;

    /// Routes native Laguna logits with the same formula as resident execution.
    pub fn route_laguna_native_experts(
        runtime: &MlxRuntime,
        router_logits: &MlxArray,
        correction_bias: Option<&MlxArray>,
        moe_descriptor: &LagunaMoeDescriptor,
        router_logit_softcap: f64,
    ) -> Result<(MlxArray, MlxArray), LagunaExecutionError> {
        let mut performance_attribution = PerformanceAttribution::disabled();
        route_laguna_experts(
            runtime,
            router_logits,
            correction_bias,
            moe_descriptor,
            router_logit_softcap,
            &mut performance_attribution,
        )
    }
}

#[cfg(feature = "direct-mlx")]
pub use public_router::route_laguna_native_experts;
