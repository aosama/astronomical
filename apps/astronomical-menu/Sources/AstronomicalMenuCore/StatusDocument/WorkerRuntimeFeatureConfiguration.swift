// Owns the Swift representation of the worker-applied runtime policy. The Rust producer exposes
// loaded models as a tagged family union, so this boundary must preserve that discriminator before
// decoding each family's distinct configuration.

struct WorkerRuntimeFeatureConfiguration: Codable, Equatable {
  let configurationGeneration: String
  let astroSsdPromptCacheEnabled: Bool
  let astroSsdPromptCacheMaximumSizeBytes: UInt64
  let loadedModel: WorkerLoadedModelRuntimeConfiguration?

  enum CodingKeys: String, CodingKey, CaseIterable {
    case configurationGeneration = "configuration_generation"
    case astroSsdPromptCacheEnabled = "astro_ssd_prompt_cache_enabled"
    case astroSsdPromptCacheMaximumSizeBytes = "astro_ssd_prompt_cache_maximum_size_bytes"
    case loadedModel = "loaded_model"
  }

  init(
    configurationGeneration: String,
    astroSsdPromptCacheEnabled: Bool,
    astroSsdPromptCacheMaximumSizeBytes: UInt64,
    loadedModel: WorkerLoadedModelRuntimeConfiguration?
  ) {
    self.configurationGeneration = configurationGeneration
    self.astroSsdPromptCacheEnabled = astroSsdPromptCacheEnabled
    self.astroSsdPromptCacheMaximumSizeBytes = astroSsdPromptCacheMaximumSizeBytes
    self.loadedModel = loadedModel
  }

  init(from decoder: Decoder) throws {
    try decoder.rejectUnknownKeys(CodingKeys.self)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    configurationGeneration = try container.decode(String.self, forKey: .configurationGeneration)
    astroSsdPromptCacheEnabled = try container.decode(
      Bool.self, forKey: .astroSsdPromptCacheEnabled)
    astroSsdPromptCacheMaximumSizeBytes = try container.decode(
      UInt64.self, forKey: .astroSsdPromptCacheMaximumSizeBytes)
    loadedModel = try container.decodeRequiredNullable(
      WorkerLoadedModelRuntimeConfiguration.self, forKey: .loadedModel)
  }
}

enum WorkerLoadedModelRuntimeConfiguration: Codable, Equatable {
  case autoregressive(WorkerLoadedAutoregressiveModelRuntimeConfiguration)
  case imageGeneration(WorkerLoadedImageGenerationModelRuntimeConfiguration)
  case embeddings(WorkerLoadedEmbeddingModelRuntimeConfiguration)
  /// A kind of model the agent shows nothing special for, such as a speech or
  /// decision model, or one a newer engine adds. Only its kind and model are
  /// kept. Refusing an unfamiliar kind made every status poll fail for as
  /// long as such a model stayed loaded, and the menu bar then said
  /// "Not running" about an engine that was answering.
  case other(kind: String, modelIdentifier: String?)

  private enum CodingKeys: String, CodingKey, CaseIterable {
    case kind
    case configuration
  }

  private enum Kind: String, Codable {
    case autoregressive
    case imageGeneration = "image_generation"
    case embeddings
  }

  init(from decoder: Decoder) throws {
    try decoder.rejectUnknownKeys(CodingKeys.self)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let kindName = try container.decode(String.self, forKey: .kind)
    guard let kind = Kind(rawValue: kindName) else {
      let otherConfiguration = try container.decodeIfPresent(
        OtherModelConfiguration.self, forKey: .configuration)
      self = .other(kind: kindName, modelIdentifier: otherConfiguration?.modelIdentifier)
      return
    }
    switch kind {
    case .autoregressive:
      self = .autoregressive(
        try container.decode(
          WorkerLoadedAutoregressiveModelRuntimeConfiguration.self,
          forKey: .configuration
        ))
    case .imageGeneration:
      self = .imageGeneration(
        try container.decode(
          WorkerLoadedImageGenerationModelRuntimeConfiguration.self,
          forKey: .configuration
        ))
    case .embeddings:
      self = .embeddings(
        try container.decode(
          WorkerLoadedEmbeddingModelRuntimeConfiguration.self,
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
    case let .imageGeneration(configuration):
      try container.encode(Kind.imageGeneration, forKey: .kind)
      try container.encode(configuration, forKey: .configuration)
    case let .embeddings(configuration):
      try container.encode(Kind.embeddings, forKey: .kind)
      try container.encode(configuration, forKey: .configuration)
    case let .other(kind, modelIdentifier):
      try container.encode(kind, forKey: .kind)
      try container.encode(
        OtherModelConfiguration(modelIdentifier: modelIdentifier), forKey: .configuration)
    }
  }

  /// The one field read from a kind the agent has no dedicated shape for.
  /// Its other fields are skipped on purpose, so a new field never breaks
  /// the poll.
  private struct OtherModelConfiguration: Codable {
    let modelIdentifier: String?

    enum CodingKeys: String, CodingKey {
      case modelIdentifier = "model_id"
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

  enum CodingKeys: String, CodingKey, CaseIterable {
    case modelIdentifier = "model_id"
    case maximumContextTokens = "maximum_context_tokens"
    case maximumOutputTokens = "maximum_output_tokens"
    case chunking
  }

  init(from decoder: Decoder) throws {
    try decoder.rejectUnknownKeys(CodingKeys.self)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    modelIdentifier = try container.decode(String.self, forKey: .modelIdentifier)
    maximumContextTokens = try container.decode(UInt32.self, forKey: .maximumContextTokens)
    maximumOutputTokens = try container.decode(UInt32.self, forKey: .maximumOutputTokens)
    chunking = try container.decode(WorkerChunkingConfiguration.self, forKey: .chunking)
  }
}

/// An image model, whichever image family it is. The family is the
/// engine's identifier for it, such as `flux2_klein` or `qwen_image_21`, kept
/// as text so a family the engine adds never fails the status poll.
struct WorkerLoadedImageGenerationModelRuntimeConfiguration: Codable, Equatable {
  let modelIdentifier: String
  let modelFamily: String
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
    modelFamily = try container.decode(String.self, forKey: .modelFamily)
    artifactRevision = try container.decode(String.self, forKey: .artifactRevision)
  }
}

struct WorkerLoadedEmbeddingModelRuntimeConfiguration: Codable, Equatable {
  let modelIdentifier: String
  /// The engine's identifier for the family, such as `modernbert`.
  let modelFamily: String
  let artifactRevision: String
  let vectorWidth: UInt32
  let maximumInputTokens: UInt32

  enum CodingKeys: String, CodingKey, CaseIterable {
    case modelIdentifier = "model_id"
    case modelFamily = "model_family"
    case artifactRevision = "artifact_revision"
    case vectorWidth = "vector_width"
    case maximumInputTokens = "maximum_input_tokens"
  }

  init(from decoder: Decoder) throws {
    try decoder.rejectUnknownKeys(CodingKeys.self)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    modelIdentifier = try container.decode(String.self, forKey: .modelIdentifier)
    modelFamily = try container.decode(String.self, forKey: .modelFamily)
    artifactRevision = try container.decode(String.self, forKey: .artifactRevision)
    vectorWidth = try container.decode(UInt32.self, forKey: .vectorWidth)
    maximumInputTokens = try container.decode(UInt32.self, forKey: .maximumInputTokens)
  }
}

struct WorkerChunkingConfiguration: Codable, Equatable {
  let fixedPromptProcessingChunkSizeTokens: UInt32
  let fixedSsdStreamingPromptProcessingChunkSizeTokens: UInt32
  let fullAttentionKeyValueGrowthTokens: UInt32
  let prefillGraphSubmissionLayerInterval: UInt32
  let experimentalSsdPagingGenerationGraphSubmissionLayerInterval: UInt32
  let ssdPromptCacheBlockTokens: UInt32?
  // The speculation block length crosses the wire absent (not null) when a
  // worker runs classic decoding: the Rust DTO skips serialization for
  // None, unlike ssd_prompt_cache_block_tokens which always serializes. The
  // decoder must therefore treat a missing key as nil rather than demand
  // key presence.
  let speculativeDecodingBlockLength: UInt64?
  let ssdPromptCacheCommonPrefixStrideBlocks: UInt32
  let experimentalDecodeStageAttributionEnabled: Bool
  let experimentalPrefillStageAttributionEnabled: Bool
  let experimentalInMemoryPrefixAdoptionEnabled: Bool
  let experimentalNeuralEnginePrefillEnabled: Bool

  enum CodingKeys: String, CodingKey, CaseIterable {
    case fixedPromptProcessingChunkSizeTokens = "fixed_prompt_processing_chunk_size_tokens"
    case fixedSsdStreamingPromptProcessingChunkSizeTokens = "fixed_ssd_streaming_prompt_processing_chunk_size_tokens"
    case fullAttentionKeyValueGrowthTokens = "full_attention_key_value_growth_tokens"
    case prefillGraphSubmissionLayerInterval = "prefill_graph_submission_layer_interval"
    case experimentalSsdPagingGenerationGraphSubmissionLayerInterval = "experimental_ssd_paging_generation_graph_submission_layer_interval"
    case ssdPromptCacheBlockTokens = "ssd_prompt_cache_block_tokens"
    case speculativeDecodingBlockLength = "speculative_decoding_block_length"
    case ssdPromptCacheCommonPrefixStrideBlocks = "ssd_prompt_cache_common_prefix_stride_blocks"
    case experimentalDecodeStageAttributionEnabled = "experimental_decode_stage_attribution_enabled"
    case experimentalPrefillStageAttributionEnabled = "experimental_prefill_stage_attribution_enabled"
    case experimentalInMemoryPrefixAdoptionEnabled = "experimental_in_memory_prefix_adoption_enabled"
    case experimentalNeuralEnginePrefillEnabled = "experimental_neural_engine_prefill_enabled"
  }

  init(from decoder: Decoder) throws {
    try decoder.rejectUnknownKeys(CodingKeys.self)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    fixedPromptProcessingChunkSizeTokens = try container.decode(
      UInt32.self, forKey: .fixedPromptProcessingChunkSizeTokens)
    fixedSsdStreamingPromptProcessingChunkSizeTokens = try container.decode(
      UInt32.self, forKey: .fixedSsdStreamingPromptProcessingChunkSizeTokens)
    fullAttentionKeyValueGrowthTokens = try container.decode(
      UInt32.self, forKey: .fullAttentionKeyValueGrowthTokens)
    prefillGraphSubmissionLayerInterval = try container.decode(
      UInt32.self, forKey: .prefillGraphSubmissionLayerInterval)
    experimentalSsdPagingGenerationGraphSubmissionLayerInterval = try container.decode(
      UInt32.self,
      forKey: .experimentalSsdPagingGenerationGraphSubmissionLayerInterval)
    ssdPromptCacheBlockTokens = try container.decodeRequiredNullable(
      UInt32.self, forKey: .ssdPromptCacheBlockTokens)
    speculativeDecodingBlockLength = try container.decodeIfPresent(
      UInt64.self, forKey: .speculativeDecodingBlockLength)
    ssdPromptCacheCommonPrefixStrideBlocks = try container.decode(
      UInt32.self, forKey: .ssdPromptCacheCommonPrefixStrideBlocks)
    experimentalDecodeStageAttributionEnabled = try container.decode(
      Bool.self, forKey: .experimentalDecodeStageAttributionEnabled)
    experimentalPrefillStageAttributionEnabled = try container.decode(
      Bool.self, forKey: .experimentalPrefillStageAttributionEnabled)
    experimentalInMemoryPrefixAdoptionEnabled = try container.decode(
      Bool.self, forKey: .experimentalInMemoryPrefixAdoptionEnabled)
    experimentalNeuralEnginePrefillEnabled = try container.decode(
      Bool.self, forKey: .experimentalNeuralEnginePrefillEnabled)
  }
}
