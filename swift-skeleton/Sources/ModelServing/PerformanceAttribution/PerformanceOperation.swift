import Foundation;

/// One stable, domain-specific operation measured on a model-serving critical path.
///
/// Raw values index a fixed enabled-only accumulator: outer diagnostic spans
/// locate latency by request phase, while leaf operations attribute concrete
/// work. Overlapping outer spans stay serialized but are excluded from the
/// attributed elapsed sum, which keeps attributed time inside the report's
/// wall-clock duration.
public enum PerformanceOperation: Int, CaseIterable, Sendable {

    case artifactValidation
    case tokenizerInitialization
    case mlxRuntimeInitialization
    case modelSafetensorsMapping
    case modelTensorBinding
    case expertPagerPlanConstruction
    case residentWeightMaterializationSynchronizationWait
    case persistentPromptCacheOpenAndScan
    case chatCommandValidation
    case imagePreprocessing
    case visionEmbeddingGraphConstruction
    case visionEmbeddingEvaluationSynchronizationWait
    case promptRendering
    case promptTokenization
    case generationOutputDecoderInitialization
    case memoryAdmissionSnapshot
    case adaptiveRamGrowthMemoryAdmission
    case linearAttentionGraphConstruction
    case fullAttentionGraphConstruction
    case persistentPromptCacheCausalInputPlanning
    case persistentPromptCachePrefixLookup
    case persistentPromptCacheKvBlockRead
    case persistentPromptCacheRecurrentSnapshotRead
    case persistentPromptCacheStateReconstruction
    case persistentPromptCacheStateMaterializationSynchronizationWait
    case persistentPromptCacheStateExtraction
    case persistentPromptCacheKvBlockSerialization
    case persistentPromptCacheRecurrentSnapshotSerialization
    case persistentPromptCachePublicationValidation
    case persistentPromptCacheGlobalQuotaEviction
    case persistentPromptCacheRetentionCleanup
    case persistentPromptCachePublicationSynchronizationWait
    case persistentPromptCacheAtomicCommit
    case pagedRouterGraphConstruction
    case retainedExpertPagePlanning
    case expertResidencyPlanning
    case rustExpertStreamingLayerPreparation
    case mandatoryPrefillCompleteLayerMaterializationWait
    case mandatoryDecodeRoutePageMaterializationWait
    case expertResidencyCommit
    case generationPreparation
    case expertRetentionReclamation
    case pagedMoeGraphConstruction
    case pagedMoeOutputMaterializationSynchronizationWait
    case residentMoeGraphConstruction
    case finalLogitsGraphConstruction
    case tokenSamplingGraphConstruction
    case forcedThinkingTransitionTokenArrayConstruction
    case prefillStateAsyncEvaluationSubmission
    case prefillStateGraphicsProcessorCompletionWait
    case prefillLinearAttentionGraphicsProcessorCompletionWait
    case prefillLinearAttentionProjectionsGraphicsProcessorCompletionWait
    case prefillLinearAttentionConvolutionGraphicsProcessorCompletionWait
    case prefillLinearAttentionNormalizationGraphicsProcessorCompletionWait
    case prefillLinearAttentionRecurrenceGraphicsProcessorCompletionWait
    case prefillLinearAttentionEpilogueGraphicsProcessorCompletionWait
    case prefillFullAttentionGraphicsProcessorCompletionWait
    case prefillFeedForwardGraphicsProcessorCompletionWait
    case expertPagingDiagnosticLogging
    case decodeAsyncEvaluationSubmission
    case generatedTokenItemSynchronizationWait
    case structuredLogitMaskComputation
    case completedForwardMemorySnapshot
    case promptPrefillAdvanceSpan
    case decodeAdvanceSpan
    case decodeAttentionGraphicsProcessorCompletionWait
    case decodeMixtureOfValuesGraphicsProcessorCompletionWait
    case decodeFeedForwardGraphicsProcessorCompletionWait
    case decodeSharedExpertGraphicsProcessorCompletionWait
    case decodeFusedValueExpertDecode
    case decodeFusedRoutedExpertDecode
    case decodeSamplingSpan
    case attentionForwardSpan
    case slidingWindowMaskConstruction
    case rotaryEmbeddingApplication
    case rotatingKeyValueStateUpdate
    case softplusAttentionGateApplication
    case expertAssignmentPreparation
    case gatheredExpertExecution
    case expertWeightedReduction
    case routerScoreSelection
    case sharedExpertExecution
    case mlpForwardSpan
    case generationFinalization
    case mlxAllocatorCacheCleanup
    case imageRuntimeSetup
    case imageTextComponentMapping
    case imageTextComponentLoading
    case imageQwenLayerGraphConstruction
    case imageQwenLayerSynchronizationWait
    case seededNoiseGraphConstruction
    case seededNoiseSynchronizationWait
    case imageScheduleConstruction
    case imagePositionGraphConstruction
    case imagePositionSynchronizationWait
    case imageTransformerComponentMapping
    case imageTransformerComponentLoading
    case imageDenoisingStepSpan
    case imageTransformerBlockGroupGraphConstruction
    case imageTransformerBlockGroupSynchronizationWait
    case imageSchedulerUpdateGraphConstruction
    case imageSchedulerUpdateSynchronizationWait
    case imagePipelineConstruction
    case imageRenderBoundary
    case imageTransformerRelease
    case imageVaeComponentMapping
    case imageVaeComponentLoading
    case imageVaeCompleteDecodeGraphConstruction
    case imageVaeDecodeSynchronizationWait
    case imagePixelConversionGraphConstruction
    case imagePixelTransfer
    case imagePngEncoding
    case imageCancellationSynchronization
    case imageComponentRelease
    case imageFinalCleanup
    case finalizedMlxMemorySnapshot
    case embeddingsTokenization
    case embeddingsForwardSpan
    case customKernelCapabilityProbe
    case routeObservationFinalization
    case previousTokenPrefetch
    case compiledElementwiseGraphConstruction

    /// Whether this operation's elapsed time joins the report's attributed sum.
    ///
    /// Outer spans contain one or more separately recorded leaves, so counting
    /// them again would inflate attributed time beyond the report's wall-clock
    /// duration; they remain serialized as timeline evidence only.
    public var contributesToAttributedElapsed: Bool {
        switch self {
        case .promptPrefillAdvanceSpan,
             .decodeAdvanceSpan,
             .attentionForwardSpan,
             .mlpForwardSpan,
             .generationPreparation,
             .imageDenoisingStepSpan:
            return false;
        default:
            return true;
        }
    }
}
