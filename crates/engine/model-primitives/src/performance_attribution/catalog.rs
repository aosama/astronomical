//! Stable timing boundaries for model loading and request execution.
//!
//! Enum order indexes a fixed enabled-only accumulator. Outer spans locate
//! latency by request phase, while leaf operations attribute concrete work;
//! overlapping outer spans are serialized but excluded from the attributed sum.

mod identifier_table;

/// One stable, domain-specific operation measured on a model-serving critical path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum PerformanceOperation {
    ArtifactValidation,
    TokenizerInitialization,
    MlxRuntimeInitialization,
    ModelSafetensorsMapping,
    ModelTensorBinding,
    ExpertPagerPlanConstruction,
    ResidentWeightMaterializationSynchronizationWait,
    AstroSsdPromptCacheOpenAndScan,
    ChatCommandValidation,
    ImagePreprocessing,
    VisionEmbeddingGraphConstruction,
    VisionEmbeddingEvaluationSynchronizationWait,
    PromptRendering,
    PromptTokenization,
    GenerationOutputDecoderInitialization,
    MemoryAdmissionSnapshot,
    LinearAttentionGraphConstruction,
    FullAttentionGraphConstruction,
    AstroSsdPromptCacheCausalInputPlanning,
    AstroSsdPromptCachePrefixLookup,
    AstroSsdPromptCacheKvBlockRead,
    AstroSsdPromptCacheRecurrentSnapshotRead,
    AstroSsdPromptCacheStateReconstruction,
    AstroSsdPromptCacheStateMaterializationSynchronizationWait,
    AstroSsdPromptCacheStateExtraction,
    AstroSsdPromptCacheKvBlockSerialization,
    AstroSsdPromptCacheRecurrentSnapshotSerialization,
    AstroSsdPromptCachePublicationValidation,
    AstroSsdPromptCacheGlobalQuotaEviction,
    AstroSsdPromptCacheRetentionCleanup,
    AstroSsdPromptCachePublicationSynchronizationWait,
    AstroSsdPromptCacheAtomicCommit,
    PagedRouterGraphConstruction,
    RetainedExpertPagePlanning,
    ExpertResidencyPlanning,
    RustExpertStreamingLayerPreparation,
    MandatoryPrefillCompleteLayerMaterializationWait,
    MandatoryDecodeRoutePageMaterializationWait,
    ExpertResidencyCommit,
    GenerationPreparation,
    ExpertRetentionReclamation,
    PagedMoeGraphConstruction,
    PagedMoeOutputMaterializationSynchronizationWait,
    ResidentMoeGraphConstruction,
    FinalLogitsGraphConstruction,
    TokenSamplingGraphConstruction,
    ForcedThinkingTransitionTokenArrayConstruction,
    PrefillStateAsyncEvaluationSubmission,
    PrefillStateGraphicsProcessorCompletionWait,
    /// Waiting for the previous layer before the next is submitted, which
    /// bounds the layers in flight to two (issues #1985 and #2221).
    PreviousLayerCompletionWait,
    PrefillLinearAttentionGraphicsProcessorCompletionWait,
    PrefillLinearAttentionProjectionsGraphicsProcessorCompletionWait,
    PrefillLinearAttentionConvolutionGraphicsProcessorCompletionWait,
    PrefillLinearAttentionNormalizationGraphicsProcessorCompletionWait,
    PrefillLinearAttentionRecurrenceGraphicsProcessorCompletionWait,
    PrefillLinearAttentionEpilogueGraphicsProcessorCompletionWait,
    PrefillFullAttentionGraphicsProcessorCompletionWait,
    PrefillFeedForwardGraphicsProcessorCompletionWait,
    PrefillMixtureOfValuesGraphicsProcessorCompletionWait,
    PrefillSharedExpertGraphicsProcessorCompletionWait,
    LoadWarmupForwardSpan,
    ExpertPagingDiagnosticLogging,
    DecodeAsyncEvaluationSubmission,
    GeneratedTokenItemSynchronizationWait,
    StructuredLogitMaskComputation,
    RepetitionPenaltyLogitTransform,
    CompletedForwardMemorySnapshot,
    PromptPrefillAdvanceSpan,
    DecodeAdvanceSpan,
    DecodeExpertRouteHostMaterialization,
    DecodeExpertRouteContiguousArrayConstruction,
    DecodeExpertRouteArrayEvaluation,
    DecodeExpertRouteHostCopy,
    DecodeAttentionGraphicsProcessorCompletionWait,
    DecodeMixtureOfValuesGraphicsProcessorCompletionWait,
    DecodeFeedForwardGraphicsProcessorCompletionWait,
    DecodeSharedExpertGraphicsProcessorCompletionWait,
    /// One decode token's jangtq2 routed projection multiplied straight from
    /// packed codes by the fused gathered projection kernel.
    DecodeFusedJangtq2GatheredProjection,
    /// A multi-token forward decoding one jangtq2 codebook page to a
    /// transient bfloat16 copy before `gather_mm`.
    Jangtq2CodebookPageDecode,
    /// One blockwise Hadamard-32 rotation of expert-bound activations,
    /// graph construction plus its synchronous evaluation (the rotation
    /// materializes and detaches, so it owns real graphics-processor time
    /// that no other operation would otherwise attribute).
    Jangtq2ActivationRotationHadamard32,
    DecodeSamplingSpan,
    AttentionForwardSpan,
    SlidingWindowMaskConstruction,
    RotaryEmbeddingApplication,
    RotatingKeyValueStateUpdate,
    SoftplusAttentionGateApplication,
    ExpertAssignmentPreparation,
    GatheredExpertExecution,
    ExpertWeightedReduction,
    RouterScoreSelection,
    SharedExpertExecution,
    MlpForwardSpan,
    GenerationFinalization,
    MlxBufferCacheCleanup,
    ImageRuntimeSetup,
    ImageTextComponentMapping,
    ImageTextComponentLoading,
    ImageQwenLayerGraphConstruction,
    ImageQwenLayerSynchronizationWait,
    SeededNoiseGraphConstruction,
    SeededNoiseSynchronizationWait,
    ImageScheduleConstruction,
    ImagePositionGraphConstruction,
    ImagePositionSynchronizationWait,
    ImageTransformerComponentMapping,
    ImageTransformerComponentLoading,
    ImageDenoisingStepSpan,
    ImageTransformerBlockGroupGraphConstruction,
    ImageTransformerBlockGroupSynchronizationWait,
    ImageSchedulerUpdateGraphConstruction,
    ImageSchedulerUpdateSynchronizationWait,
    ImagePipelineConstruction,
    ImageRenderBoundary,
    ImageTransformerRelease,
    ImageVaeComponentMapping,
    ImageVaeComponentLoading,
    ImageVaeCompleteDecodeGraphConstruction,
    ImageVaeDecodeSynchronizationWait,
    ImagePixelConversionGraphConstruction,
    ImagePixelTransfer,
    ImagePngEncoding,
    ImageCancellationSynchronization,
    ImageComponentRelease,
    ImageFinalCleanup,
    FinalizedMlxMemorySnapshot,
    EmbeddingsTokenization,
    EmbeddingsForwardSpan,
    CustomKernelCapabilityProbe,
    RouteObservationFinalization,
    PreviousTokenPrefetch,
    CompiledElementwiseGraphConstruction,
    ExpertLayerPlanConstruction,
    ArtifactMemoryGeometry,
    /// One speculative round's parallel draft block pass: embedding, the
    /// six-layer trunk, the language-model head, and the proposal readback.
    SpeculativeDraftBlockPass,
    /// One speculative round's transactional decoder-state checkpoint
    /// capture before the verification forward.
    SpeculativeRoundStateCheckpoint,
    /// One speculative round's multi-token verification forward over the
    /// drafted block, including the conditioning hidden-state taps.
    SpeculativeVerificationForward,
    /// One speculative round's per-position target token choice readback:
    /// the argmax or tempered-sample transfer from the device.
    SpeculativeTokenChoiceReadback,
    /// One speculative round's rejection work: the checkpoint restore and
    /// the accepted-prefix replay forward through the target.
    SpeculativeRejectedPrefixReplay,
    /// One speculative round's post-decision draft conditioning: tap
    /// slicing, hidden-state projection, and the draft's context key-value
    /// growth for the accepted prefix.
    SpeculativeDraftKeyValueConditioning,
}

impl PerformanceOperation {
    pub(super) const COUNT: usize = Self::SpeculativeDraftKeyValueConditioning as usize + 1;

    /// Every operation in declaration order, so report serialization can pair
    /// an operation with its stored measurement by position.
    pub(super) const ALL: [Self; Self::COUNT] = {
        let mut operations_in_declaration_order = [Self::ArtifactValidation; Self::COUNT];
        let mut operation_position = 0;
        while operation_position < Self::COUNT {
            operations_in_declaration_order[operation_position] =
                Self::CATALOG[operation_position].0;
            operation_position += 1;
        }
        operations_in_declaration_order
    };

    /// Stable report identifier for this operation, read from the catalog
    /// position that equals its discriminant.
    pub(super) const fn identifier(self) -> &'static str {
        Self::CATALOG[self as usize].1
    }

    pub(super) const fn contributes_to_attributed_elapsed(self) -> bool {
        // These operations contain one or more separately recorded leaves. They
        // remain useful timeline evidence but counting them again would inflate
        // attributed time beyond the report's wall-clock duration.
        !matches!(
            self,
            Self::PromptPrefillAdvanceSpan
                | Self::DecodeAdvanceSpan
                | Self::AttentionForwardSpan
                | Self::MlpForwardSpan
                | Self::GenerationPreparation
                | Self::ImageDenoisingStepSpan
        )
    }
}
