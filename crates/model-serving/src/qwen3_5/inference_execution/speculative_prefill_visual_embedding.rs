#[cfg(feature = "direct-mlx")]
use astronomical_runtime_integration::MlxArray;

#[cfg(feature = "direct-mlx")]
use crate::{PerformanceOperation, Qwen3_5ExecutionError};

#[cfg(feature = "direct-mlx")]
use super::Qwen3_5EngineState;
#[cfg(feature = "direct-mlx")]
use super::engine_request::Qwen3_5EngineRequest;
#[cfg(feature = "direct-mlx")]
use super::speculative_prefill_failure::configured_speculative_prefill_failure;

#[cfg(feature = "direct-mlx")]
impl Qwen3_5EngineState {
    pub(super) fn prepare_speculative_prefill_draft_visual_embeddings(
        &self,
        active_request: &mut Qwen3_5EngineRequest,
        draft_model: &crate::Qwen3_5Model,
        restored_draft_prefix_token_count: usize,
        is_visual_speculative_prefill_request: bool,
    ) -> Result<Option<MlxArray>, crate::InferenceEngineError> {
        if !is_visual_speculative_prefill_request {
            return Ok(None);
        }
        let Some(draft_vision_model) = draft_model.vision_model() else {
            return Err(configured_speculative_prefill_failure(
                active_request.request_id,
                "drafter visual initialization",
                "visual speculative-prefill draft lost its vision tower",
            ));
        };
        let ordered_image_visual_embedding_row_counts = active_request
            .speculative_prefill_processed_visual_images
            .iter()
            .map(|processed_visual_image| {
                processed_visual_image.image_token_count_after_spatial_merge
            })
            .collect::<Vec<_>>();
        let draft_visual_embedding_suffix_plan =
            crate::qwen3_5::plan_qwen3_5_visual_embedding_suffix(
                &active_request.input_token_ids,
                restored_draft_prefix_token_count,
                &ordered_image_visual_embedding_row_counts,
                active_request.image_pad_token_id,
            )
            .map_err(|visual_suffix_planning_error| {
                configured_speculative_prefill_failure(
                    active_request.request_id,
                    "drafter visual suffix planning",
                    visual_suffix_planning_error,
                )
            })?;
        let draft_visual_embedding_result = if draft_visual_embedding_suffix_plan
            .remaining_visual_embedding_row_count()
            == 0
        {
            Ok(None)
        } else {
            active_request
                .performance_attribution
                .measure_operation(
                    PerformanceOperation::SpeculativePrefillDraftVisionEmbeddingGraphConstruction,
                    |_performance_attribution| {
                        draft_vision_model.forward(
                            draft_model.runtime(),
                            &active_request.speculative_prefill_processed_visual_images,
                        )
                    },
                )
                .and_then(|complete_draft_visual_embeddings| {
                    active_request.performance_attribution.measure_operation(
                        PerformanceOperation::SpeculativePrefillDraftVisionEmbeddingEvaluationSynchronizationWait,
                        |_performance_attribution| {
                            draft_model
                                .runtime()
                                .evaluate_arrays(&[&complete_draft_visual_embeddings])
                        },
                    )?;
                    let draft_visual_embedding_shape = complete_draft_visual_embeddings.shape();
                    let total_visual_embedding_row_count =
                        ordered_image_visual_embedding_row_counts.iter().sum::<usize>();
                    if draft_visual_embedding_shape.len() != 2
                        || draft_visual_embedding_shape[0] as usize
                            != total_visual_embedding_row_count
                    {
                        return Err(Qwen3_5ExecutionError::InvalidInput {
                            description:
                                "speculative-prefill draft visual embeddings have an invalid shape",
                        });
                    }
                    let restored_visual_embedding_row_count = draft_visual_embedding_suffix_plan
                        .restored_visual_embedding_row_count();
                    if restored_visual_embedding_row_count == 0 {
                        return Ok(Some(complete_draft_visual_embeddings));
                    }
                    let restored_visual_embedding_row_count_i32 =
                        i32::try_from(restored_visual_embedding_row_count).map_err(|_| {
                            Qwen3_5ExecutionError::InvalidInput {
                                description:
                                    "restored draft visual row count exceeds the MLX range",
                            }
                        })?;
                    let total_visual_embedding_row_count_i32 =
                        i32::try_from(total_visual_embedding_row_count).map_err(|_| {
                            Qwen3_5ExecutionError::InvalidInput {
                                description: "draft visual row count exceeds the MLX range",
                            }
                        })?;
                    let draft_visual_embedding_hidden_size = draft_visual_embedding_shape[1];
                    draft_model
                        .runtime()
                        .slice(
                            &complete_draft_visual_embeddings,
                            &[restored_visual_embedding_row_count_i32, 0],
                            &[
                                total_visual_embedding_row_count_i32,
                                draft_visual_embedding_hidden_size,
                            ],
                            &[1, 1],
                        )
                        .map(Some)
                        .map_err(Into::into)
                })
        };
        match draft_visual_embedding_result {
            Ok(draft_visual_embeddings) => {
                tracing::info!(
                    request_id = active_request.request_id.value(),
                    draft_visual_embedding_row_count = draft_visual_embeddings
                        .as_ref()
                        .map_or(0, |draft_visual_embeddings| draft_visual_embeddings.shape()
                            [0]),
                    restored_draft_visual_embedding_row_count =
                        draft_visual_embedding_suffix_plan.restored_visual_embedding_row_count(),
                    "completed visual speculative-prefill draft projection"
                );
                active_request
                    .speculative_prefill_processed_visual_images
                    .clear();
                Ok(draft_visual_embeddings)
            }
            Err(draft_visual_embedding_error) => {
                Err(configured_speculative_prefill_failure(
                    active_request.request_id,
                    "drafter visual projection",
                    draft_visual_embedding_error,
                ))
            }
        }
    }
}
