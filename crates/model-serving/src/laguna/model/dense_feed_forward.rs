//! Dense SwiGLU: `down(silu(gate(x)) * up(x))`.

use astronomical_runtime_integration::{MlxArray, MlxCompiledSwiGlu, MlxRuntime};

use crate::laguna::artifacts::{LagunaExpertProjection, LagunaLayerTensorRole};
use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};

use super::error::LagunaExecutionError;
use super::weights::LagunaNativeWeights;

pub(super) fn dense_swiglu(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    weights: &LagunaNativeWeights,
    layer_index: usize,
    compiled_swiglu: &MlxCompiledSwiGlu,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, LagunaExecutionError> {
    performance_attribution.measure_operation(PerformanceOperation::MlpForwardSpan, |_| {
        let gate_states = weights
            .linear(
                layer_index,
                LagunaLayerTensorRole::DenseFeedForward(LagunaExpertProjection::Gate),
            )?
            .project(runtime, hidden_states)?;
        let up_states = weights
            .linear(
                layer_index,
                LagunaLayerTensorRole::DenseFeedForward(LagunaExpertProjection::Up),
            )?
            .project(runtime, hidden_states)?;
        let hidden_product =
            runtime.apply_compiled_swiglu(compiled_swiglu, &gate_states, &up_states)?;
        weights
            .linear(
                layer_index,
                LagunaLayerTensorRole::DenseFeedForward(LagunaExpertProjection::Down),
            )?
            .project(runtime, &hidden_product)
    })
}
