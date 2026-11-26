//! The stable report identifier of every operation, kept as declarative
//! data apart from the enum and the rules around it.

use super::PerformanceOperation;

impl PerformanceOperation {
    /// Every operation in declaration order with its stable report
    /// identifier. The table length is `COUNT`, so an entry added or removed
    /// here without the matching change to the enum stops the build, and
    /// `ALL` and `identifier` both read this one table.
    pub(super) const CATALOG: [(Self, &'static str); Self::COUNT] = [
        (Self::ArtifactValidation, "artifact_validation"),
        (Self::TokenizerInitialization, "tokenizer_initialization"),
        (Self::MlxRuntimeInitialization, "mlx_runtime_initialization"),
        (Self::ModelSafetensorsMapping, "model_safetensors_mapping"),
        (Self::ModelTensorBinding, "model_tensor_binding"),
        (
            Self::ExpertPagerPlanConstruction,
            "expert_pager_plan_construction",
        ),
        (
            Self::ResidentWeightMaterializationSynchronizationWait,
            "resident_weight_materialization_synchronization_wait",
        ),
        (
            Self::AstroSsdPromptCacheOpenAndScan,
            "astro_ssd_prompt_cache_open_and_scan",
        ),
        (Self::ChatCommandValidation, "chat_command_validation"),
        (Self::ImagePreprocessing, "image_preprocessing"),
        (
            Self::VisionEmbeddingGraphConstruction,
            "vision_embedding_graph_construction",
        ),
        (
            Self::VisionEmbeddingEvaluationSynchronizationWait,
            "vision_embedding_evaluation_synchronization_wait",
        ),
        (Self::PromptRendering, "prompt_rendering"),
        (Self::PromptTokenization, "prompt_tokenization"),
        (
            Self::GenerationOutputDecoderInitialization,
            "generation_output_decoder_initialization",
        ),
        (Self::MemoryAdmissionSnapshot, "memory_admission_snapshot"),
        (
            Self::LinearAttentionGraphConstruction,
            "linear_attention_graph_construction",
        ),
        (
            Self::FullAttentionGraphConstruction,
            "full_attention_graph_construction",
        ),
        (
            Self::AstroSsdPromptCacheCausalInputPlanning,
            "astro_ssd_prompt_cache_causal_input_planning",
        ),
        (
            Self::AstroSsdPromptCachePrefixLookup,
            "astro_ssd_prompt_cache_prefix_lookup",
        ),
        (
            Self::AstroSsdPromptCacheKvBlockRead,
            "astro_ssd_prompt_cache_kv_block_read",
        ),
        (
            Self::AstroSsdPromptCacheRecurrentSnapshotRead,
            "astro_ssd_prompt_cache_recurrent_snapshot_read",
        ),
        (
            Self::AstroSsdPromptCacheStateReconstruction,
            "astro_ssd_prompt_cache_state_reconstruction",
        ),
        (
            Self::AstroSsdPromptCacheStateMaterializationSynchronizationWait,
            "astro_ssd_prompt_cache_state_materialization_synchronization_wait",
        ),
        (
            Self::AstroSsdPromptCacheStateExtraction,
            "astro_ssd_prompt_cache_state_extraction",
        ),
        (
            Self::AstroSsdPromptCacheKvBlockSerialization,
            "astro_ssd_prompt_cache_kv_block_serialization",
        ),
        (
            Self::AstroSsdPromptCacheRecurrentSnapshotSerialization,
            "astro_ssd_prompt_cache_recurrent_snapshot_serialization",
        ),
        (
            Self::AstroSsdPromptCachePublicationValidation,
            "astro_ssd_prompt_cache_publication_validation",
        ),
        (
            Self::AstroSsdPromptCacheGlobalQuotaEviction,
            "astro_ssd_prompt_cache_global_quota_eviction",
        ),
        (
            Self::AstroSsdPromptCacheRetentionCleanup,
            "astro_ssd_prompt_cache_retention_cleanup",
        ),
        (
            Self::AstroSsdPromptCachePublicationSynchronizationWait,
            "astro_ssd_prompt_cache_publication_synchronization_wait",
        ),
        (
            Self::AstroSsdPromptCacheAtomicCommit,
            "astro_ssd_prompt_cache_atomic_commit",
        ),
        (
            Self::PagedRouterGraphConstruction,
            "paged_router_graph_construction",
        ),
        (
            Self::RetainedExpertPagePlanning,
            "retained_expert_page_planning",
        ),
        (
            Self::ExpertResidencyPlanning,
            "phase_aware_expert_residency_planning",
        ),
        (
            Self::RustExpertStreamingLayerPreparation,
            "rust_expert_streaming_layer_preparation",
        ),
        (
            Self::MandatoryPrefillCompleteLayerMaterializationWait,
            "mandatory_prefill_complete_layer_materialization_wait",
        ),
        (
            Self::MandatoryDecodeRoutePageMaterializationWait,
            "mandatory_decode_route_page_materialization_wait",
        ),
        (Self::ExpertResidencyCommit, "expert_residency_commit"),
        (Self::GenerationPreparation, "generation_preparation"),
        (
            Self::ExpertRetentionReclamation,
            "expert_retention_reclamation",
        ),
        (
            Self::PagedMoeGraphConstruction,
            "paged_moe_graph_construction",
        ),
        (
            Self::PagedMoeOutputMaterializationSynchronizationWait,
            "paged_moe_output_materialization_synchronization_wait",
        ),
        (
            Self::ResidentMoeGraphConstruction,
            "resident_moe_graph_construction",
        ),
        (
            Self::FinalLogitsGraphConstruction,
            "final_logits_graph_construction",
        ),
        (
            Self::TokenSamplingGraphConstruction,
            "token_sampling_graph_construction",
        ),
        (
            Self::ForcedThinkingTransitionTokenArrayConstruction,
            "forced_thinking_transition_token_array_construction",
        ),
        (
            Self::PrefillStateAsyncEvaluationSubmission,
            "prefill_state_async_evaluation_submission",
        ),
        (
            Self::PrefillStateGraphicsProcessorCompletionWait,
            "prefill_state_graphics_processor_completion_wait",
        ),
        (
            Self::PreviousLayerCompletionWait,
            "previous_layer_completion_wait",
        ),
        (
            Self::PrefillLinearAttentionGraphicsProcessorCompletionWait,
            "prefill_linear_attention_graphics_processor_completion_wait",
        ),
        (
            Self::PrefillLinearAttentionProjectionsGraphicsProcessorCompletionWait,
            "prefill_linear_attention_projections_graphics_processor_completion_wait",
        ),
        (
            Self::PrefillLinearAttentionConvolutionGraphicsProcessorCompletionWait,
            "prefill_linear_attention_convolution_graphics_processor_completion_wait",
        ),
        (
            Self::PrefillLinearAttentionNormalizationGraphicsProcessorCompletionWait,
            "prefill_linear_attention_normalization_graphics_processor_completion_wait",
        ),
        (
            Self::PrefillLinearAttentionRecurrenceGraphicsProcessorCompletionWait,
            "prefill_linear_attention_recurrence_graphics_processor_completion_wait",
        ),
        (
            Self::PrefillLinearAttentionEpilogueGraphicsProcessorCompletionWait,
            "prefill_linear_attention_epilogue_graphics_processor_completion_wait",
        ),
        (
            Self::PrefillFullAttentionGraphicsProcessorCompletionWait,
            "prefill_full_attention_graphics_processor_completion_wait",
        ),
        (
            Self::PrefillFeedForwardGraphicsProcessorCompletionWait,
            "prefill_feed_forward_graphics_processor_completion_wait",
        ),
        (
            Self::PrefillMixtureOfValuesGraphicsProcessorCompletionWait,
            "prefill_mixture_of_values_graphics_processor_completion_wait",
        ),
        (
            Self::PrefillSharedExpertGraphicsProcessorCompletionWait,
            "prefill_shared_expert_graphics_processor_completion_wait",
        ),
        (Self::LoadWarmupForwardSpan, "load_warmup_forward_span"),
        (
            Self::ExpertPagingDiagnosticLogging,
            "expert_paging_diagnostic_logging",
        ),
        (
            Self::DecodeAsyncEvaluationSubmission,
            "decode_async_evaluation_submission",
        ),
        (
            Self::GeneratedTokenItemSynchronizationWait,
            "generated_token_item_synchronization_wait",
        ),
        (
            Self::StructuredLogitMaskComputation,
            "structured_logit_mask_computation",
        ),
        (
            Self::RepetitionPenaltyLogitTransform,
            "repetition_penalty_logit_transform",
        ),
        (
            Self::CompletedForwardMemorySnapshot,
            "completed_forward_memory_snapshot",
        ),
        (
            Self::PromptPrefillAdvanceSpan,
            "prompt_prefill_advance_span",
        ),
        (Self::DecodeAdvanceSpan, "decode_advance_span"),
        (
            Self::DecodeExpertRouteHostMaterialization,
            "decode_expert_route_host_materialization",
        ),
        (
            Self::DecodeExpertRouteContiguousArrayConstruction,
            "decode_expert_route_contiguous_array_construction",
        ),
        (
            Self::DecodeExpertRouteArrayEvaluation,
            "decode_expert_route_array_evaluation",
        ),
        (
            Self::DecodeExpertRouteHostCopy,
            "decode_expert_route_host_copy",
        ),
        (
            Self::DecodeAttentionGraphicsProcessorCompletionWait,
            "decode_attention_graphics_processor_completion_wait",
        ),
        (
            Self::DecodeMixtureOfValuesGraphicsProcessorCompletionWait,
            "decode_mixture_of_values_graphics_processor_completion_wait",
        ),
        (
            Self::DecodeFeedForwardGraphicsProcessorCompletionWait,
            "decode_feed_forward_graphics_processor_completion_wait",
        ),
        (
            Self::DecodeSharedExpertGraphicsProcessorCompletionWait,
            "decode_shared_expert_graphics_processor_completion_wait",
        ),
        (
            Self::DecodeFusedJangtq2GatheredProjection,
            "decode_fused_jangtq2_gathered_projection",
        ),
        (
            Self::Jangtq2CodebookPageDecode,
            "jangtq2_codebook_page_decode",
        ),
        (
            Self::Jangtq2ActivationRotationHadamard32,
            "jangtq2_activation_rotation_hadamard32",
        ),
        (Self::DecodeSamplingSpan, "decode_sampling_span"),
        (Self::AttentionForwardSpan, "attention_forward_span"),
        (
            Self::SlidingWindowMaskConstruction,
            "sliding_window_mask_construction",
        ),
        (
            Self::RotaryEmbeddingApplication,
            "rotary_embedding_application",
        ),
        (
            Self::RotatingKeyValueStateUpdate,
            "rotating_key_value_state_update",
        ),
        (
            Self::SoftplusAttentionGateApplication,
            "softplus_attention_gate_application",
        ),
        (
            Self::ExpertAssignmentPreparation,
            "expert_assignment_preparation",
        ),
        (Self::GatheredExpertExecution, "gathered_expert_execution"),
        (Self::ExpertWeightedReduction, "expert_weighted_reduction"),
        (Self::RouterScoreSelection, "router_score_selection"),
        (Self::SharedExpertExecution, "shared_expert_execution"),
        (Self::MlpForwardSpan, "mlp_forward_span"),
        (Self::GenerationFinalization, "generation_finalization"),
        (Self::MlxBufferCacheCleanup, "mlx_buffer_cache_cleanup"),
        (Self::ImageRuntimeSetup, "image_runtime_setup"),
        (
            Self::ImageTextComponentMapping,
            "image_text_component_mapping",
        ),
        (
            Self::ImageTextComponentLoading,
            "image_text_component_loading",
        ),
        (
            Self::ImageQwenLayerGraphConstruction,
            "image_qwen_layer_graph_construction",
        ),
        (
            Self::ImageQwenLayerSynchronizationWait,
            "image_qwen_layer_synchronization_wait",
        ),
        (
            Self::SeededNoiseGraphConstruction,
            "seeded_noise_graph_construction",
        ),
        (
            Self::SeededNoiseSynchronizationWait,
            "seeded_noise_synchronization_wait",
        ),
        (
            Self::ImageScheduleConstruction,
            "image_schedule_construction",
        ),
        (
            Self::ImagePositionGraphConstruction,
            "image_position_graph_construction",
        ),
        (
            Self::ImagePositionSynchronizationWait,
            "image_position_synchronization_wait",
        ),
        (
            Self::ImageTransformerComponentMapping,
            "image_transformer_component_mapping",
        ),
        (
            Self::ImageTransformerComponentLoading,
            "image_transformer_component_loading",
        ),
        (Self::ImageDenoisingStepSpan, "image_denoising_step_span"),
        (
            Self::ImageTransformerBlockGroupGraphConstruction,
            "image_transformer_block_group_graph_construction",
        ),
        (
            Self::ImageTransformerBlockGroupSynchronizationWait,
            "image_transformer_block_group_synchronization_wait",
        ),
        (
            Self::ImageSchedulerUpdateGraphConstruction,
            "image_scheduler_update_graph_construction",
        ),
        (
            Self::ImageSchedulerUpdateSynchronizationWait,
            "image_scheduler_update_synchronization_wait",
        ),
        (
            Self::ImagePipelineConstruction,
            "image_pipeline_construction",
        ),
        (Self::ImageRenderBoundary, "image_render_boundary"),
        (Self::ImageTransformerRelease, "image_transformer_release"),
        (
            Self::ImageVaeComponentMapping,
            "image_vae_component_mapping",
        ),
        (
            Self::ImageVaeComponentLoading,
            "image_vae_component_loading",
        ),
        (
            Self::ImageVaeCompleteDecodeGraphConstruction,
            "image_vae_complete_decode_graph_construction",
        ),
        (
            Self::ImageVaeDecodeSynchronizationWait,
            "image_vae_decode_synchronization_wait",
        ),
        (
            Self::ImagePixelConversionGraphConstruction,
            "image_pixel_conversion_graph_construction",
        ),
        (Self::ImagePixelTransfer, "image_pixel_transfer"),
        (Self::ImagePngEncoding, "image_png_encoding"),
        (
            Self::ImageCancellationSynchronization,
            "image_cancellation_synchronization",
        ),
        (Self::ImageComponentRelease, "image_component_release"),
        (Self::ImageFinalCleanup, "image_final_cleanup"),
        (
            Self::FinalizedMlxMemorySnapshot,
            "finalized_mlx_memory_snapshot",
        ),
        (Self::EmbeddingsTokenization, "embeddings_tokenization"),
        (Self::EmbeddingsForwardSpan, "embeddings_forward_span"),
        (
            Self::CustomKernelCapabilityProbe,
            "custom_kernel_capability_probe",
        ),
        (
            Self::RouteObservationFinalization,
            "route_observation_finalization",
        ),
        (Self::PreviousTokenPrefetch, "previous_token_prefetch"),
        (
            Self::CompiledElementwiseGraphConstruction,
            "compiled_elementwise_graph_construction",
        ),
        (
            Self::ExpertLayerPlanConstruction,
            "expert_layer_plan_construction",
        ),
        (Self::ArtifactMemoryGeometry, "artifact_memory_geometry"),
        (
            Self::SpeculativeDraftBlockPass,
            "speculative_draft_block_pass",
        ),
        (
            Self::SpeculativeRoundStateCheckpoint,
            "speculative_round_state_checkpoint",
        ),
        (
            Self::SpeculativeVerificationForward,
            "speculative_verification_forward",
        ),
        (
            Self::SpeculativeTokenChoiceReadback,
            "speculative_token_choice_readback",
        ),
        (
            Self::SpeculativeRejectedPrefixReplay,
            "speculative_rejected_prefix_replay",
        ),
        (
            Self::SpeculativeDraftKeyValueConditioning,
            "speculative_draft_key_value_conditioning",
        ),
    ];
}
