// Owns the Swift representation of the worker-applied runtime policy. The Rust producer exposes
// loaded models as a tagged family union, so this boundary must preserve that discriminator before
// decoding each family's distinct configuration.

struct WorkerRuntimeFeatureConfiguration: Codable, Equatable {
  let configurationGeneration: String
  let persistentPromptCacheEnabled: Bool
  let promptCacheMaximumSizeBytes: UInt64
  let loadedModel: WorkerLoadedModelRuntimeConfiguration?

  enum CodingKeys: String, CodingKey, CaseIterable {
    case configurationGeneration = "configuration_generation"
    case persistentPromptCacheEnabled = "persistent_prompt_cache_enabled"
    case promptCacheMaximumSizeBytes = "prompt_cache_maximum_size_bytes"
    case loadedModel = "loaded_model"
  }

  init(
    configurationGeneration: String,
    persistentPromptCacheEnabled: Bool,
    promptCacheMaximumSizeBytes: UInt64,
    loadedModel: WorkerLoadedModelRuntimeConfiguration?
  ) {
    self.configurationGeneration = configurationGeneration
    self.persistentPromptCacheEnabled = persistentPromptCacheEnabled
    self.promptCacheMaximumSizeBytes = promptCacheMaximumSizeBytes
    self.loadedModel = loadedModel
  }

  init(from decoder: Decoder) throws {
    try decoder.rejectUnknownKeys(CodingKeys.self)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    configurationGeneration = try container.decode(String.self, forKey: .configurationGeneration)
    persistentPromptCacheEnabled = try container.decode(
      Bool.self, forKey: .persistentPromptCacheEnabled)
    promptCacheMaximumSizeBytes = try container.decode(
      UInt64.self, forKey: .promptCacheMaximumSizeBytes)
    loadedModel = try container.decodeRequiredNullable(
      WorkerLoadedModelRuntimeConfiguration.self, forKey: .loadedModel)
  }
}

enum WorkerLoadedModelRuntimeConfiguration: Codable, Equatable {
  case autoregressive(WorkerLoadedAutoregressiveModelRuntimeConfiguration)
  case flux2Klein(WorkerLoadedFlux2KleinModelRuntimeConfiguration)

  private enum CodingKeys: String, CodingKey, CaseIterable {
    case kind
    case configuration
  }

  private enum Kind: String, Codable {
    case autoregressive
    case flux2Klein = "flux2_klein"
  }

  init(from decoder: Decoder) throws {
    try decoder.rejectUnknownKeys(CodingKeys.self)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let kind = try container.decode(Kind.self, forKey: .kind)
    switch kind {
    case .autoregressive:
      self = .autoregressive(
        try container.decode(
          WorkerLoadedAutoregressiveModelRuntimeConfiguration.self,
          forKey: .configuration
        ))
    case .flux2Klein:
      self = .flux2Klein(
        try container.decode(
          WorkerLoadedFlux2KleinModelRuntimeConfiguration.self,
          forKey: .configuration
        ))
    }
  }

  func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    switch self {
    case let .autoregressive(configuration):
      try container.encode(Kind.autoregressive, forKey: .kind)
      try container.encode(configuration, forKey: .configuration)
    case let .flux2Klein(configuration):
      try container.encode(Kind.flux2Klein, forKey: .kind)
      try container.encode(configuration, forKey: .configuration)
    }
  }

  var autoregressiveConfiguration: WorkerLoadedAutoregressiveModelRuntimeConfiguration? {
    guard case let .autoregressive(configuration) = self else { return nil }
    return configuration
  }
}

struct WorkerLoadedAutoregressiveModelRuntimeConfiguration: Codable, Equatable {
  let modelIdentifier: String
  let maximumContextTokens: UInt32
  let maximumOutputTokens: UInt32
  let chunking: WorkerChunkingConfiguration
  let mtpDraftDepth: UInt8?
  let speculativePrefillEnabled: Bool
  let speculativePrefill: WorkerSpeculativePrefillRuntimeConfiguration?

  enum CodingKeys: String, CodingKey, CaseIterable {
    case modelIdentifier = "model_id"
    case maximumContextTokens = "maximum_context_tokens"
    case maximumOutputTokens = "maximum_output_tokens"
    case chunking
    case mtpDraftDepth = "mtp_draft_depth"
    case speculativePrefillEnabled = "speculative_prefill_enabled"
    case speculativePrefill = "speculative_prefill"
  }

  init(from decoder: Decoder) throws {
    try decoder.rejectUnknownKeys(CodingKeys.self)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    modelIdentifier = try container.decode(String.self, forKey: .modelIdentifier)
    maximumContextTokens = try container.decode(UInt32.self, forKey: .maximumContextTokens)
    maximumOutputTokens = try container.decode(UInt32.self, forKey: .maximumOutputTokens)
    chunking = try container.decode(WorkerChunkingConfiguration.self, forKey: .chunking)
    mtpDraftDepth = try container.decodeRequiredNullable(UInt8.self, forKey: .mtpDraftDepth)
    speculativePrefillEnabled = try container.decode(
      Bool.self, forKey: .speculativePrefillEnabled)
    speculativePrefill = try container.decodeRequiredNullable(
      WorkerSpeculativePrefillRuntimeConfiguration.self, forKey: .speculativePrefill)
  }
}

struct WorkerLoadedFlux2KleinModelRuntimeConfiguration: Codable, Equatable {
  let modelIdentifier: String
  let modelFamily: WorkerImageGenerationModelFamily
  let artifactRevision: String

  enum CodingKeys: String, CodingKey, CaseIterable {
    case modelIdentifier = "model_id"
    case modelFamily = "model_family"
    case artifactRevision = "artifact_revision"
  }

  init(from decoder: Decoder) throws {
    try decoder.rejectUnknownKeys(CodingKeys.self)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    modelIdentifier = try container.decode(String.self, forKey: .modelIdentifier)
    modelFamily = try container.decode(WorkerImageGenerationModelFamily.self, forKey: .modelFamily)
    artifactRevision = try container.decode(String.self, forKey: .artifactRevision)
  }
}

enum WorkerImageGenerationModelFamily: String, Codable, Equatable {
  case flux2Klein = "flux2_klein"
}

struct WorkerSpeculativePrefillRuntimeConfiguration: Codable, Equatable {
  let draftModelIdentifier: String
  let minimumPromptTokens: UInt32
  let keepPercentage: UInt32

  enum CodingKeys: String, CodingKey, CaseIterable {
    case draftModelIdentifier = "draft_model_id"
    case minimumPromptTokens = "minimum_prompt_tokens"
    case keepPercentage = "keep_percentage"
  }

  init(from decoder: Decoder) throws {
    try decoder.rejectUnknownKeys(CodingKeys.self)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    draftModelIdentifier = try container.decode(String.self, forKey: .draftModelIdentifier)
    minimumPromptTokens = try container.decode(UInt32.self, forKey: .minimumPromptTokens)
    keepPercentage = try container.decode(UInt32.self, forKey: .keepPercentage)
  }
}

struct WorkerChunkingConfiguration: Codable, Equatable {
  let fixedPromptProcessingChunkSizeTokens: UInt32
  let fixedSsdStreamingPromptProcessingChunkSizeTokens: UInt32?
  let fullAttentionKeyValueGrowthTokens: UInt32
  let speculativePrefillDraftForwardTokens: UInt32
  let prefillGraphSubmissionLayerInterval: UInt32
  let experimentalSsdPagingGenerationGraphSubmissionLayerInterval: UInt32
  let promptCacheBlockTokens: UInt32?
  let promptCacheCommonPrefixStrideBlocks: UInt32

  enum CodingKeys: String, CodingKey, CaseIterable {
    case fixedPromptProcessingChunkSizeTokens = "fixed_prompt_processing_chunk_size_tokens"
    case fixedSsdStreamingPromptProcessingChunkSizeTokens = "fixed_ssd_streaming_prompt_processing_chunk_size_tokens"
    case fullAttentionKeyValueGrowthTokens = "full_attention_key_value_growth_tokens"
    case speculativePrefillDraftForwardTokens = "speculative_prefill_draft_forward_tokens"
    case prefillGraphSubmissionLayerInterval = "prefill_graph_submission_layer_interval"
    case experimentalSsdPagingGenerationGraphSubmissionLayerInterval = "experimental_ssd_paging_generation_graph_submission_layer_interval"
    case promptCacheBlockTokens = "prompt_cache_block_tokens"
    case promptCacheCommonPrefixStrideBlocks = "prompt_cache_common_prefix_stride_blocks"
  }

  init(from decoder: Decoder) throws {
    try decoder.rejectUnknownKeys(CodingKeys.self)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    fixedPromptProcessingChunkSizeTokens = try container.decode(
      UInt32.self, forKey: .fixedPromptProcessingChunkSizeTokens)
    fixedSsdStreamingPromptProcessingChunkSizeTokens = try container.decodeValueOrOmission(
      UInt32.self, forKey: .fixedSsdStreamingPromptProcessingChunkSizeTokens)
    fullAttentionKeyValueGrowthTokens = try container.decode(
      UInt32.self, forKey: .fullAttentionKeyValueGrowthTokens)
    speculativePrefillDraftForwardTokens = try container.decode(
      UInt32.self, forKey: .speculativePrefillDraftForwardTokens)
    prefillGraphSubmissionLayerInterval = try container.decode(
      UInt32.self, forKey: .prefillGraphSubmissionLayerInterval)
    experimentalSsdPagingGenerationGraphSubmissionLayerInterval = try container.decode(
      UInt32.self,
      forKey: .experimentalSsdPagingGenerationGraphSubmissionLayerInterval)
    promptCacheBlockTokens = try container.decodeRequiredNullable(
      UInt32.self, forKey: .promptCacheBlockTokens)
    promptCacheCommonPrefixStrideBlocks = try container.decode(
      UInt32.self, forKey: .promptCacheCommonPrefixStrideBlocks)
  }
}
