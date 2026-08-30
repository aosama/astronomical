//! Text-encoder loader with request-scoped and convenience-complete entry points.

use astronomical_runtime_integration::MlxRuntime;

use crate::{PerformanceAttribution, PerformanceOperation};

use super::super::Flux2KleinResidencyMode;
use super::batch::Flux2KleinPreparedTextBatch;
use super::error::Flux2KleinTextConditioningError;
use super::prompt::Flux2KleinPromptRenderer;
use super::state::{
    Flux2KleinTextConditioning, Flux2KleinTextConditioningAdvance, Flux2KleinTextConditioningState,
};
use super::tokenizer::Flux2KleinTokenizer;
use super::weights::{EXECUTED_LAYER_COUNT, Flux2KleinTextWeights};

pub(crate) struct Flux2KleinTextConditioner {
    tokenizer: Flux2KleinTokenizer,
    weights: Flux2KleinTextWeights,
}

impl Flux2KleinTextConditioner {
    pub(crate) fn load(
        runtime: &MlxRuntime,
        tokenizer: Flux2KleinTokenizer,
        text_shards: std::collections::BTreeMap<String, crate::ValidatedWeightsFile>,
        residency_mode: Flux2KleinResidencyMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, Flux2KleinTextConditioningError> {
        let weights = Flux2KleinTextWeights::load(
            runtime,
            text_shards,
            residency_mode,
            performance_attribution,
        )?;
        Ok(Self { tokenizer, weights })
    }

    // The complete path remains available for parity acceptance and non-interruptible callers.
    #[allow(dead_code)]
    pub(crate) fn condition(
        self,
        runtime: &MlxRuntime,
        user_prompts: &[String],
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Flux2KleinTextConditioning, Flux2KleinTextConditioningError> {
        let rendered_prompts = performance_attribution.measure_operation(
            PerformanceOperation::PromptRendering,
            |_performance_attribution| {
                Ok::<_, Flux2KleinTextConditioningError>(
                    Flux2KleinPromptRenderer::render_user_prompts(user_prompts),
                )
            },
        )?;
        let prepared_batch = performance_attribution.measure_operation(
            PerformanceOperation::PromptTokenization,
            |_performance_attribution| self.tokenizer.prepare_rendered_prompts(&rendered_prompts),
        )?;
        self.condition_prepared(runtime, prepared_batch, performance_attribution)
    }

    pub(crate) fn condition_prepared(
        self,
        runtime: &MlxRuntime,
        prepared_batch: Flux2KleinPreparedTextBatch,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Flux2KleinTextConditioning, Flux2KleinTextConditioningError> {
        let state = Flux2KleinTextConditioningState::initialize(
            runtime,
            prepared_batch,
            self.weights,
            performance_attribution,
        )?;
        match state.advance_layer_group(runtime, EXECUTED_LAYER_COUNT, performance_attribution)? {
            Flux2KleinTextConditioningAdvance::LayerGroupCompleted(_) => {
                Err(Flux2KleinTextConditioningError::WeightsUnavailable)
            }
            Flux2KleinTextConditioningAdvance::ConditioningCompleted(conditioning) => {
                Ok(conditioning)
            }
        }
    }

    pub(crate) fn start(
        self,
        runtime: &MlxRuntime,
        user_prompts: &[String],
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Flux2KleinTextConditioningState, Flux2KleinTextConditioningError> {
        let rendered_prompts = performance_attribution.measure_operation(
            PerformanceOperation::PromptRendering,
            |_| {
                Ok::<_, Flux2KleinTextConditioningError>(
                    Flux2KleinPromptRenderer::render_user_prompts(user_prompts),
                )
            },
        )?;
        let prepared_batch = performance_attribution
            .measure_operation(PerformanceOperation::PromptTokenization, |_| {
                self.tokenizer.prepare_rendered_prompts(&rendered_prompts)
            })?;
        Flux2KleinTextConditioningState::initialize(
            runtime,
            prepared_batch,
            self.weights,
            performance_attribution,
        )
    }
}
