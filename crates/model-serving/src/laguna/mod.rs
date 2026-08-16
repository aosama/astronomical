//! Laguna-owned artifact, normalization, and text protocol contracts.

mod artifacts;
mod normalization;
mod text;

pub use artifacts::{
    LagunaArtifactValidationError, LagunaArtifactValidator, LagunaAttentionProjection,
    LagunaCanonicalSourceLayout, LagunaCanonicalTensorAssemblyKind,
    LagunaCanonicalTensorDescriptor, LagunaExpertGateUpLayout, LagunaExpertProjection,
    LagunaGlobalTensorRole, LagunaIndexTotalSizeSemantics, LagunaLayerTensorRole,
    LagunaNonExecutableMetadataDescriptor, LagunaRawTensorNameRecord, LagunaRetainedArtifactFiles,
    LagunaShardIndex, LagunaShardIndexError, LagunaTensorAssembly, LagunaTensorComponent,
    LagunaTensorContract, LagunaTensorId, LagunaTensorNameContract,
    LagunaTensorNameNormalizationError, LagunaTensorNameNormalizer, LagunaTensorSource,
    LagunaTensorSourceDescriptor, LagunaTensorSourceRole, LagunaTensorStorageEncoding,
    ValidatedLagunaArtifact,
};
pub use normalization::{
    LagunaAffineProfile, LagunaAttentionDescriptor, LagunaAttentionKind, LagunaBlockFp8Profile,
    LagunaCacheDescriptor, LagunaCompressedFeedForwardProjection, LagunaCompressedIgnoreScope,
    LagunaCompressedInputActivationDescriptor, LagunaCompressedModuleScope,
    LagunaCompressedStorageDescriptor, LagunaCompressedWeightEncoding, LagunaDefaultRopeDescriptor,
    LagunaDenseFeedForwardDescriptor, LagunaDirectAffineStorageDescriptor,
    LagunaExactStorageSupport, LagunaExecutionDtype, LagunaFeedForwardDescriptor,
    LagunaFp8InputActivationDescriptor, LagunaFp8KvCacheDescriptor, LagunaGatingKind,
    LagunaLayerDescriptor, LagunaModelDescriptor, LagunaMoeDescriptor, LagunaNormalizationError,
    LagunaNvfp4InputActivationDescriptor, LagunaNvfp4Profile, LagunaRopeDescriptor,
    LagunaRouterKind, LagunaStorageDescriptor, LagunaSymmetricPackedAffineProfile,
    LagunaTargetContract, LagunaTargetNormalizer, LagunaYarnRopeDescriptor,
};
pub use text::{
    LagunaGenerationProcessor, LagunaInferenceRequest, LagunaOutputEvent, LagunaOutputParser,
    LagunaOutputParserError, LagunaPreparationError, LagunaPreparedGeneration,
    LagunaPromptRenderer, LagunaPromptRendererError, LagunaRequestOutput, LagunaRequestOutputError,
    LagunaSamplerConfig, LagunaSamplingStrategy, LagunaTextArtifactDescriptor,
    LagunaTextArtifactError, LagunaTextArtifactNormalizer, LagunaTextArtifactSources,
    LagunaTokenDecoder, LagunaTokenizer, LagunaTokenizerError,
};

const LAGUNA_UNAVAILABLE_REASON: &str = "Laguna model execution is not implemented in this build";

/// Returns the bounded reason while this contract-only layer remains non-executable.
#[must_use]
pub const fn laguna_unavailable_reason() -> &'static str {
    LAGUNA_UNAVAILABLE_REASON
}
