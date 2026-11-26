//! What the engine does differently for a prompt while Neural
//! Engine programs are attached to its model (issue #2585).

use super::Qwen3_5EngineState;
use super::layer_major_span_plan::LayerMajorSpanPlan;

impl Qwen3_5EngineState {
    /// The plan cut to its first span while programs are attached, which
    /// choose their engines and are timed a span at a time; otherwise the
    /// plan as it is.
    pub(super) fn one_span_while_neural_engine_programs_are_attached(
        &self,
        whole_stretch_plan: LayerMajorSpanPlan,
        prefill_start: usize,
    ) -> LayerMajorSpanPlan {
        let has_neural_engine_programs = self.core.model.as_ref().is_some_and(|model| {
            model
                .optional_accelerators
                .neural_engine_mlp_prefill
                .is_some()
        });
        if has_neural_engine_programs {
            whole_stretch_plan.limited_to_first_span(prefill_start)
        } else {
            whole_stretch_plan
        }
    }

    /// Before a span runs: a waiting program measurement must not read
    /// through this GPU work, and attached programs decide whether the span
    /// uses them.
    pub(super) fn ready_neural_engine_programs_for_span(&mut self, span_token_count: usize) {
        self.neural_engine_prefill.note_gpu_used();
        if let Some(model) = self.core.model.as_ref() {
            model.decide_neural_engine_for_prompt_span(span_token_count);
        }
    }

    /// Moves the programs along, or settles what the last request learned
    /// about them. Only called while no request is active, which is when
    /// programs can be attached or released.
    pub(super) fn advance_neural_engine_preparation(
        &mut self,
    ) -> Option<ipc_protocol::NeuralEnginePrefillStatus> {
        let model = self.core.model.as_mut()?;
        self.neural_engine_prefill.advance(model)
    }
}
