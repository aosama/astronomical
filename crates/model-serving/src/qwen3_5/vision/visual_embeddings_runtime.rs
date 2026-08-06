use super::super::inference_execution::{
    Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error,
};
use super::Qwen3_5ProcessedImage;
use super::visual_embeddings::Qwen3_5VisualEmbeddingSuffixPlan;
use crate::{
    InferenceEngineError, PerformanceAttribution, PerformanceOperation,
    PersistentVisualEmbeddingKey,
};
use astronomical_ipc_protocol::RequestId;
use astronomical_runtime_integration::{MlxArray, MlxDtype};

impl Qwen3_5EngineState {
    pub(in crate::qwen3_5) fn resolve_visual_embeddings_for_processed_images(
        &mut self,
        request_id: RequestId,
        processed_visual_images: &[Qwen3_5ProcessedImage],
        visual_embedding_suffix_plan: &Qwen3_5VisualEmbeddingSuffixPlan,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<MlxArray>, InferenceEngineError> {
        if visual_embedding_suffix_plan.remaining_visual_embedding_row_count() == 0 {
            return Ok(None);
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let persistent_visual_embedding_model_contract = self
            .persistent_visual_embedding_model_contract
            .as_ref()
            .ok_or_else(|| {
                fatal_engine_error(
                    "Qwen3.5 persistent visual embedding model contract is not loaded",
                )
            })?;
        let visual_embedding_hidden_size =
            persistent_visual_embedding_model_contract.visual_embedding_hidden_size();
        let runtime = model.runtime();
        let required_images = visual_embedding_suffix_plan.required_images();
        let mut visual_embeddings_by_required_image = Vec::with_capacity(required_images.len());
        visual_embeddings_by_required_image.resize_with(required_images.len(), || None);
        let mut missing_required_image_indexes = Vec::new();
        let mut missing_processed_visual_images = Vec::new();
        let mut persistent_prompt_cache_visual_embedding_hit_row_counts = Vec::new();
        let mut persistent_prompt_cache_visual_embedding_miss_count = 0_u64;

        for (required_image_index, required_image) in required_images.iter().enumerate() {
            let processed_visual_image = processed_visual_images
                .get(required_image.image_index())
                .ok_or_else(|| {
                    fatal_engine_error("visual suffix plan refers to a missing image")
                })?;
            let visual_embedding_key = PersistentVisualEmbeddingKey::for_image(
                processed_visual_image.encoded_image_sha256,
                persistent_visual_embedding_model_contract.model_id(),
                persistent_visual_embedding_model_contract.model_revision(),
            );
            let persistent_prompt_cache_visual_embedding_lookup_was_attempted =
                self.persistent_prompt_cache.is_some();
            let loaded_visual_embeddings = match self.persistent_prompt_cache.as_ref() {
                Some(persistent_prompt_cache) => {
                    match persistent_prompt_cache.load_visual_embedding(
                        runtime,
                        &visual_embedding_key,
                        persistent_visual_embedding_model_contract,
                    ) {
                        Ok(Some(loaded_visual_embeddings))
                            if visual_embedding_array_has_expected_layout(
                                &loaded_visual_embeddings,
                                required_image.image_visual_embedding_row_count(),
                                visual_embedding_hidden_size,
                            ) =>
                        {
                            tracing::debug!(
                                request_id = request_id.value(),
                                image_ordinal = required_image.image_index(),
                                visual_embedding_row_count =
                                    required_image.image_visual_embedding_row_count(),
                                "persistent visual embedding cache hit"
                            );
                            persistent_prompt_cache_visual_embedding_hit_row_counts
                                .push(required_image.image_visual_embedding_row_count());
                            Some(loaded_visual_embeddings)
                        }
                        Ok(Some(_stale_visual_embeddings)) => {
                            tracing::warn!(
                                request_id = request_id.value(),
                                image_ordinal = required_image.image_index(),
                                expected_visual_embedding_row_count =
                                    required_image.image_visual_embedding_row_count(),
                                "persistent visual embedding shape mismatch; recomputing"
                            );
                            None
                        }
                        Ok(None) => None,
                        Err(visual_embedding_load_error) => {
                            tracing::warn!(
                                request_id = request_id.value(),
                                image_ordinal = required_image.image_index(),
                                "persistent visual embedding load failed; recomputing: \
                                 {visual_embedding_load_error}"
                            );
                            None
                        }
                    }
                }
                None => None,
            };
            if let Some(loaded_visual_embeddings) = loaded_visual_embeddings {
                visual_embeddings_by_required_image[required_image_index] =
                    Some(loaded_visual_embeddings);
            } else {
                if persistent_prompt_cache_visual_embedding_lookup_was_attempted {
                    persistent_prompt_cache_visual_embedding_miss_count =
                        persistent_prompt_cache_visual_embedding_miss_count.saturating_add(1);
                }
                tracing::debug!(
                    request_id = request_id.value(),
                    image_ordinal = required_image.image_index(),
                    visual_embedding_row_count = required_image.image_visual_embedding_row_count(),
                    "persistent visual embedding cache miss"
                );
                missing_required_image_indexes.push(required_image_index);
                missing_processed_visual_images.push(processed_visual_image.clone());
            }
        }

        if !missing_processed_visual_images.is_empty() {
            let vision_model = model.vision_model().ok_or_else(|| {
                fatal_engine_error("Qwen3.5 image request lost the loaded vision model")
            })?;
            let computed_missing_visual_embeddings = performance_attribution
                .measure_operation(
                    PerformanceOperation::VisionEmbeddingGraphConstruction,
                    |_performance_attribution| {
                        vision_model.forward(runtime, &missing_processed_visual_images)
                    },
                )
                .map_err(qwen3_5_runtime_error)?;
            performance_attribution
                .measure_operation(
                    PerformanceOperation::VisionEmbeddingEvaluationSynchronizationWait,
                    |_performance_attribution| {
                        runtime.evaluate_arrays(&[&computed_missing_visual_embeddings])
                    },
                )
                .map_err(qwen3_5_runtime_error)?;
            let total_missing_visual_embedding_row_count = missing_required_image_indexes
                .iter()
                .try_fold(
                    0_usize,
                    |total_missing_visual_embedding_row_count, required_image_index| {
                        total_missing_visual_embedding_row_count.checked_add(
                            required_images[*required_image_index]
                                .image_visual_embedding_row_count(),
                        )
                    },
                )
                .ok_or_else(|| {
                    fatal_engine_error("missing visual embedding row count overflowed")
                })?;
            if !visual_embedding_array_has_expected_layout(
                &computed_missing_visual_embeddings,
                total_missing_visual_embedding_row_count,
                visual_embedding_hidden_size,
            ) {
                return Err(fatal_engine_error(
                    "computed visual embeddings do not match expected Qwen3.5 text width",
                ));
            }
            let mut missing_visual_embedding_start_row = 0_usize;
            for required_image_index in missing_required_image_indexes {
                let required_image = &required_images[required_image_index];
                let missing_visual_embedding_end_row = missing_visual_embedding_start_row
                    .checked_add(required_image.image_visual_embedding_row_count())
                    .ok_or_else(|| fatal_engine_error("missing visual row offset overflowed"))?;
                let image_visual_embeddings = slice_visual_embedding_rows(
                    runtime,
                    &computed_missing_visual_embeddings,
                    missing_visual_embedding_start_row,
                    missing_visual_embedding_end_row,
                    visual_embedding_hidden_size,
                )?;
                if let Some(persistent_prompt_cache) = self.persistent_prompt_cache.as_ref() {
                    let processed_visual_image = processed_visual_images
                        .get(required_image.image_index())
                        .ok_or_else(|| {
                            fatal_engine_error("visual suffix plan refers to a missing image")
                        })?;
                    let visual_embedding_key = PersistentVisualEmbeddingKey::for_image(
                        processed_visual_image.encoded_image_sha256,
                        persistent_visual_embedding_model_contract.model_id(),
                        persistent_visual_embedding_model_contract.model_revision(),
                    );
                    if let Err(visual_embedding_save_error) = persistent_prompt_cache
                        .save_visual_embedding(
                            runtime,
                            &visual_embedding_key,
                            &image_visual_embeddings,
                        )
                    {
                        tracing::warn!(
                            request_id = request_id.value(),
                            image_ordinal = required_image.image_index(),
                            "persistent visual embedding save failed: {visual_embedding_save_error}"
                        );
                    }
                }
                visual_embeddings_by_required_image[required_image_index] =
                    Some(image_visual_embeddings);
                missing_visual_embedding_start_row = missing_visual_embedding_end_row;
            }
        }

        let mut suffix_visual_embedding_arrays = Vec::with_capacity(required_images.len());
        for (required_image_index, required_image) in required_images.iter().enumerate() {
            let full_image_visual_embeddings = visual_embeddings_by_required_image
                [required_image_index]
                .take()
                .ok_or_else(|| fatal_engine_error("visual embedding resolution lost an image"))?;
            let suffix_visual_embeddings = if required_image.suffix_start_row() == 0
                && required_image.suffix_row_count()
                    == required_image.image_visual_embedding_row_count()
            {
                full_image_visual_embeddings
            } else {
                slice_visual_embedding_rows(
                    runtime,
                    &full_image_visual_embeddings,
                    required_image.suffix_start_row(),
                    required_image
                        .suffix_start_row()
                        .checked_add(required_image.suffix_row_count())
                        .ok_or_else(|| fatal_engine_error("visual suffix row end overflowed"))?,
                    visual_embedding_hidden_size,
                )?
            };
            suffix_visual_embedding_arrays.push(suffix_visual_embeddings);
        }
        let resolved_visual_embeddings = if suffix_visual_embedding_arrays.len() == 1 {
            suffix_visual_embedding_arrays.pop()
        } else {
            let suffix_visual_embedding_references =
                suffix_visual_embedding_arrays.iter().collect::<Vec<_>>();
            Some(
                runtime
                    .concatenate_axis(&suffix_visual_embedding_references, 0)
                    .map_err(qwen3_5_runtime_error)?,
            )
        };
        runtime
            .clear_allocator_cache()
            .map_err(qwen3_5_runtime_error)?;
        for persistent_prompt_cache_visual_embedding_hit_row_count in
            persistent_prompt_cache_visual_embedding_hit_row_counts
        {
            self.persistent_prompt_cache_counters
                .record_persistent_prompt_cache_visual_embedding_hit(
                    persistent_prompt_cache_visual_embedding_hit_row_count,
                );
        }
        for _persistent_prompt_cache_visual_embedding_miss_index in
            0..persistent_prompt_cache_visual_embedding_miss_count
        {
            self.persistent_prompt_cache_counters
                .record_persistent_prompt_cache_visual_embedding_miss();
        }
        Ok(resolved_visual_embeddings)
    }
}

fn visual_embedding_array_has_expected_layout(
    visual_embeddings: &MlxArray,
    visual_embedding_row_count: usize,
    visual_embedding_hidden_size: usize,
) -> bool {
    let Ok(visual_embedding_row_count) = i32::try_from(visual_embedding_row_count) else {
        return false;
    };
    let Ok(visual_embedding_hidden_size) = i32::try_from(visual_embedding_hidden_size) else {
        return false;
    };
    visual_embeddings.dtype() == MlxDtype::BFloat16
        && visual_embeddings.shape() == [visual_embedding_row_count, visual_embedding_hidden_size]
}

fn slice_visual_embedding_rows(
    runtime: &astronomical_runtime_integration::MlxRuntime,
    visual_embeddings: &MlxArray,
    visual_embedding_start_row: usize,
    visual_embedding_end_row: usize,
    visual_embedding_hidden_size: usize,
) -> Result<MlxArray, InferenceEngineError> {
    let visual_embedding_hidden_size = i32::try_from(visual_embedding_hidden_size)
        .map_err(|_| fatal_engine_error("visual embedding hidden size exceeds the i32 range"))?;
    runtime
        .slice(
            visual_embeddings,
            &[
                i32::try_from(visual_embedding_start_row).map_err(|_| {
                    fatal_engine_error("visual embedding start row exceeds the i32 range")
                })?,
                0,
            ],
            &[
                i32::try_from(visual_embedding_end_row).map_err(|_| {
                    fatal_engine_error("visual embedding end row exceeds the i32 range")
                })?,
                visual_embedding_hidden_size,
            ],
            &[1, 1],
        )
        .map_err(qwen3_5_runtime_error)
}
