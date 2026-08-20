import Foundation

struct ServerApplicationIdentity: Codable, Equatable {
  let version: String
  let buildNumber: UInt64
  let commit: String
  let isDirty: Bool
  let channel: String
  let channelDisplayName: String
  let stateDirectory: String

  enum CodingKeys: String, CodingKey {
    case version, commit, channel
    case buildNumber = "build_number"
    case isDirty = "is_dirty"
    case channelDisplayName = "channel_display_name"
    case stateDirectory = "state_directory"
  }

  var buildTitle: String {
    let dirtySuffix = isDirty ? "-dirty" : ""
    return "\(version) · \(channelDisplayName) · \(commit)\(dirtySuffix)"
  }
}

struct SupervisorStatusDocument: Codable, Equatable {
  struct MlxMemoryBreakdown: Equatable {
    let expertPayloadByteCount: UInt64
    let modelCorePayloadByteCount: UInt64
    let contextStatePayloadByteCount: UInt64
    let speculativePrefillDraftMemoryByteCount: UInt64
    let runtimeWorkByteCount: UInt64
    let availableByteCount: UInt64
  }
  struct MlxMemorySnapshot: Codable, Equatable {
    let source: String
    let activeMemoryBytes: UInt64
    let allocatorCacheMemoryBytes: UInt64
    let peakMemoryBytes: UInt64
    let expertPayloadBytes: UInt64
    let modelCorePayloadBytes: UInt64
    let contextStatePayloadBytes: UInt64
    let speculativePrefillDraftMemoryBytes: UInt64

    enum CodingKeys: String, CodingKey {
      case source
      case activeMemoryBytes = "active_memory_bytes"
      case allocatorCacheMemoryBytes = "allocator_cache_memory_bytes"
      case peakMemoryBytes = "peak_memory_bytes"
      case expertPayloadBytes = "expert_payload_bytes"
      case modelCorePayloadBytes = "model_core_payload_bytes"
      case contextStatePayloadBytes = "context_state_payload_bytes"
      case speculativePrefillDraftMemoryBytes = "speculative_prefill_draft_memory_bytes"
    }

    init(from decoder: Decoder) throws {
      let container = try decoder.container(keyedBy: CodingKeys.self)
      source = try container.decode(String.self, forKey: .source)
      activeMemoryBytes = try container.decode(UInt64.self, forKey: .activeMemoryBytes)
      allocatorCacheMemoryBytes =
        try container.decode(UInt64.self, forKey: .allocatorCacheMemoryBytes)
      peakMemoryBytes = try container.decode(UInt64.self, forKey: .peakMemoryBytes)
      expertPayloadBytes = try container.decode(UInt64.self, forKey: .expertPayloadBytes)
      modelCorePayloadBytes = try container.decode(UInt64.self, forKey: .modelCorePayloadBytes)
      contextStatePayloadBytes =
        try container.decode(UInt64.self, forKey: .contextStatePayloadBytes)
      speculativePrefillDraftMemoryBytes =
        try container.decodeIfPresent(UInt64.self, forKey: .speculativePrefillDraftMemoryBytes) ?? 0
    }
  }
  struct Progress: Codable, Equatable {
    let phase: String
    let processedTokens: UInt32
    let totalTokens: UInt32
    let elapsedMilliseconds: UInt64

    enum CodingKeys: String, CodingKey {
      case phase
      case processedTokens = "processed_tokens"
      case totalTokens = "total_tokens"
      case elapsedMilliseconds = "elapsed_ms"
    }
  }

  struct ServingSession: Codable, Equatable {
    let completedRequestCount: UInt64
    let totalPromptTokenCount: UInt64
    let totalReusedPromptTokenCount: UInt64
    let targetPromptWorkTokenCount: UInt64
    let targetReusedPromptWorkTokenCount: UInt64
    let drafterPromptWorkTokenCount: UInt64
    let drafterReusedPromptWorkTokenCount: UInt64
    let averagePrefillTokensPerSecond: Double
    let averageGenerationTokensPerSecond: Double

    static let empty = ServingSession(
      completedRequestCount: 0,
      totalPromptTokenCount: 0,
      totalReusedPromptTokenCount: 0,
      targetPromptWorkTokenCount: 0,
      targetReusedPromptWorkTokenCount: 0,
      drafterPromptWorkTokenCount: 0,
      drafterReusedPromptWorkTokenCount: 0,
      averagePrefillTokensPerSecond: 0,
      averageGenerationTokensPerSecond: 0
    )

    enum CodingKeys: String, CodingKey {
      case completedRequestCount = "completed_request_count"
      case totalPromptTokenCount = "total_prompt_token_count"
      case totalReusedPromptTokenCount = "total_reused_prompt_token_count"
      case targetPromptWorkTokenCount = "target_prompt_work_token_count"
      case targetReusedPromptWorkTokenCount = "target_reused_prompt_work_token_count"
      case drafterPromptWorkTokenCount = "drafter_prompt_work_token_count"
      case drafterReusedPromptWorkTokenCount = "drafter_reused_prompt_work_token_count"
      case averagePrefillTokensPerSecond = "average_prefill_tok_per_second"
      case averageGenerationTokensPerSecond = "average_generation_tok_per_second"
    }

    init(
      completedRequestCount: UInt64,
      totalPromptTokenCount: UInt64,
      totalReusedPromptTokenCount: UInt64,
      targetPromptWorkTokenCount: UInt64,
      targetReusedPromptWorkTokenCount: UInt64,
      drafterPromptWorkTokenCount: UInt64,
      drafterReusedPromptWorkTokenCount: UInt64,
      averagePrefillTokensPerSecond: Double,
      averageGenerationTokensPerSecond: Double
    ) {
      self.completedRequestCount = completedRequestCount
      self.totalPromptTokenCount = totalPromptTokenCount
      self.totalReusedPromptTokenCount = totalReusedPromptTokenCount
      self.targetPromptWorkTokenCount = targetPromptWorkTokenCount
      self.targetReusedPromptWorkTokenCount = targetReusedPromptWorkTokenCount
      self.drafterPromptWorkTokenCount = drafterPromptWorkTokenCount
      self.drafterReusedPromptWorkTokenCount = drafterReusedPromptWorkTokenCount
      self.averagePrefillTokensPerSecond = averagePrefillTokensPerSecond
      self.averageGenerationTokensPerSecond = averageGenerationTokensPerSecond
    }

    init(from decoder: Decoder) throws {
      let container = try decoder.container(keyedBy: CodingKeys.self)
      completedRequestCount = try container.decodeIfPresent(UInt64.self, forKey: .completedRequestCount) ?? 0
      totalPromptTokenCount = try container.decodeIfPresent(UInt64.self, forKey: .totalPromptTokenCount) ?? 0
      totalReusedPromptTokenCount = try container.decodeIfPresent(UInt64.self, forKey: .totalReusedPromptTokenCount) ?? 0
      targetPromptWorkTokenCount = try container.decodeIfPresent(UInt64.self, forKey: .targetPromptWorkTokenCount) ?? 0
      targetReusedPromptWorkTokenCount = try container.decodeIfPresent(UInt64.self, forKey: .targetReusedPromptWorkTokenCount) ?? 0
      drafterPromptWorkTokenCount = try container.decodeIfPresent(UInt64.self, forKey: .drafterPromptWorkTokenCount) ?? 0
      drafterReusedPromptWorkTokenCount = try container.decodeIfPresent(UInt64.self, forKey: .drafterReusedPromptWorkTokenCount) ?? 0
      averagePrefillTokensPerSecond = try container.decodeIfPresent(Double.self, forKey: .averagePrefillTokensPerSecond) ?? 0
      averageGenerationTokensPerSecond = try container.decodeIfPresent(Double.self, forKey: .averageGenerationTokensPerSecond) ?? 0
    }
  }

  let application: ServerApplicationIdentity?
  let status: String
  let activity: String
  let configuration: ConfigurationStatusDocument?
  let readyModelIdentifier: String?
  let readyModelSizeBytes: UInt64?
  let progress: Progress?
  let expertMemoryMode: String?
  let expertResidency: ExpertResidencySnapshot?
  let mtpEnabled: Bool
  let mtpConfiguredDraftDepth: UInt8?
  let mtpArtifactMaximumDraftDepth: UInt8?
  let mtpArtifactDefaultDraftDepth: UInt8?
  let mtpResolvedRequestedDraftDepth: UInt8?
  let mtpEffectiveExecutionDraftDepth: UInt8?
  let mtpRuntimeState: String
  let mtpUnavailableReason: String?
  let configuredSpeculativePrefillEnabled: Bool
  let speculativePrefillEnabled: Bool
  // True only when the current worker has emitted its runtime configuration event; configuration
  // intent alone is insufficient because a replacement worker can still fail before applying it.
  let workerRuntimeFeatureConfigurationApplied: Bool
  // The complete acknowledgement lets control actions compare the exact policy they requested
  // with the policy currently served, rather than inferring it from one derived feature flag.
  let workerRuntimeFeatureConfiguration: WorkerRuntimeFeatureConfiguration?
  let mlxMemorySnapshot: MlxMemorySnapshot?
  let mlxMemoryCeilingBytes: UInt64
  let machineMlxMemoryCeilingBytes: UInt64
  let minimumMlxMemoryCeilingBytes: UInt64
  let configuredMaximumMlxMemoryGigabytes: UInt64?
  let pendingMlxMemoryCeilingBytes: UInt64?
  let mlxMemoryLimitError: String?
  let servingSession: ServingSession

  enum CodingKeys: String, CodingKey {
    case application, status, activity, progress, configuration
    case readyModelIdentifier = "ready_model_id"
    case readyModelSizeBytes = "ready_model_size_bytes"
    case expertMemoryMode = "expert_memory_mode"
    case expertResidency = "expert_residency"
    case mtpEnabled = "mtp_enabled"
    case mtpConfiguredDraftDepth = "mtp_configured_draft_depth"
    case mtpArtifactMaximumDraftDepth = "mtp_artifact_maximum_draft_depth"
    case mtpArtifactDefaultDraftDepth = "mtp_artifact_default_draft_depth"
    case mtpResolvedRequestedDraftDepth = "mtp_resolved_requested_draft_depth"
    case mtpEffectiveExecutionDraftDepth = "mtp_effective_execution_draft_depth"
    case mtpRuntimeState = "mtp_runtime_state"
    case mtpUnavailableReason = "mtp_unavailable_reason"
    case configuredSpeculativePrefillEnabled = "configured_speculative_prefill_enabled"
    case speculativePrefillEnabled = "speculative_prefill_enabled"
    case workerRuntimeFeatureConfigurationApplied = "worker_runtime_feature_configuration_applied"
    case workerRuntimeFeatureConfiguration = "worker_runtime_feature_configuration"
    case mlxMemorySnapshot = "mlx_memory_snapshot"
    case mlxMemoryCeilingBytes = "mlx_memory_ceiling_bytes"
    case machineMlxMemoryCeilingBytes = "machine_mlx_memory_ceiling_bytes"
    case minimumMlxMemoryCeilingBytes = "minimum_mlx_memory_ceiling_bytes"
    case configuredMaximumMlxMemoryGigabytes = "configured_maximum_mlx_memory_gb"
    case pendingMlxMemoryCeilingBytes = "pending_mlx_memory_ceiling_bytes"
    case mlxMemoryLimitError = "mlx_memory_limit_error"
    case servingSession = "serving_session"
  }

  init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    application = try container.decodeIfPresent(ServerApplicationIdentity.self, forKey: .application)
    status = try container.decode(String.self, forKey: .status)
    activity = try container.decode(String.self, forKey: .activity)
    configuration = try container.decodeIfPresent(
      ConfigurationStatusDocument.self,
      forKey: .configuration
    )
    readyModelIdentifier = try container.decodeIfPresent(String.self, forKey: .readyModelIdentifier)
    readyModelSizeBytes = try container.decodeIfPresent(UInt64.self, forKey: .readyModelSizeBytes)
    progress = try container.decodeIfPresent(Progress.self, forKey: .progress)
    expertMemoryMode = try container.decodeIfPresent(String.self, forKey: .expertMemoryMode)
    expertResidency = try container.decodeIfPresent(
      ExpertResidencySnapshot.self, forKey: .expertResidency)
    mtpEnabled = try container.decodeIfPresent(Bool.self, forKey: .mtpEnabled) ?? false
    mtpConfiguredDraftDepth = try container.decodeIfPresent(UInt8.self, forKey: .mtpConfiguredDraftDepth)
    mtpArtifactMaximumDraftDepth = try container.decodeIfPresent(UInt8.self, forKey: .mtpArtifactMaximumDraftDepth)
    mtpArtifactDefaultDraftDepth = try container.decodeIfPresent(UInt8.self, forKey: .mtpArtifactDefaultDraftDepth)
    mtpResolvedRequestedDraftDepth = try container.decodeIfPresent(UInt8.self, forKey: .mtpResolvedRequestedDraftDepth)
    mtpEffectiveExecutionDraftDepth = try container.decodeIfPresent(UInt8.self, forKey: .mtpEffectiveExecutionDraftDepth)
    mtpRuntimeState = try container.decodeIfPresent(String.self, forKey: .mtpRuntimeState) ?? "disabled"
    mtpUnavailableReason = try container.decodeIfPresent(String.self, forKey: .mtpUnavailableReason)
    configuredSpeculativePrefillEnabled = try container.decodeIfPresent(
      Bool.self, forKey: .configuredSpeculativePrefillEnabled) ?? false
    speculativePrefillEnabled = try container.decodeIfPresent(Bool.self, forKey: .speculativePrefillEnabled) ?? false
    workerRuntimeFeatureConfigurationApplied = try container.decodeIfPresent(
      Bool.self, forKey: .workerRuntimeFeatureConfigurationApplied) ?? false
    workerRuntimeFeatureConfiguration = try container.decodeIfPresent(
      WorkerRuntimeFeatureConfiguration.self, forKey: .workerRuntimeFeatureConfiguration)
    mlxMemorySnapshot = try container.decodeIfPresent(MlxMemorySnapshot.self, forKey: .mlxMemorySnapshot)
    mlxMemoryCeilingBytes =
      try container.decodeIfPresent(UInt64.self, forKey: .mlxMemoryCeilingBytes) ?? 0
    machineMlxMemoryCeilingBytes =
      try container.decodeIfPresent(UInt64.self, forKey: .machineMlxMemoryCeilingBytes) ?? 0
    minimumMlxMemoryCeilingBytes =
      try container.decodeIfPresent(UInt64.self, forKey: .minimumMlxMemoryCeilingBytes) ?? 1
    configuredMaximumMlxMemoryGigabytes =
      try container.decodeIfPresent(UInt64.self, forKey: .configuredMaximumMlxMemoryGigabytes)
    pendingMlxMemoryCeilingBytes =
      try container.decodeIfPresent(UInt64.self, forKey: .pendingMlxMemoryCeilingBytes)
    mlxMemoryLimitError = try container.decodeIfPresent(String.self, forKey: .mlxMemoryLimitError)
    servingSession =
      try container.decodeIfPresent(ServingSession.self, forKey: .servingSession) ?? .empty
  }

  init(
    application: ServerApplicationIdentity? = nil,
    status: String,
    activity: String,
    configuration: ConfigurationStatusDocument? = nil,
    readyModelIdentifier: String?,
    readyModelSizeBytes: UInt64? = nil,
    progress: Progress?,
    expertMemoryMode: String?,
    expertResidency: ExpertResidencySnapshot? = nil,
    mtpEnabled: Bool = false,
    mtpConfiguredDraftDepth: UInt8? = nil,
    mtpArtifactMaximumDraftDepth: UInt8? = nil,
    mtpArtifactDefaultDraftDepth: UInt8? = nil,
    mtpResolvedRequestedDraftDepth: UInt8? = nil,
    mtpEffectiveExecutionDraftDepth: UInt8? = nil,
    mtpRuntimeState: String = "disabled",
    mtpUnavailableReason: String? = nil,
    configuredSpeculativePrefillEnabled: Bool = false,
    speculativePrefillEnabled: Bool = false,
    workerRuntimeFeatureConfigurationApplied: Bool = false,
    workerRuntimeFeatureConfiguration: WorkerRuntimeFeatureConfiguration? = nil,
    mlxMemorySnapshot: MlxMemorySnapshot? = nil,
    mlxMemoryCeilingBytes: UInt64,
    machineMlxMemoryCeilingBytes: UInt64 = 0,
    minimumMlxMemoryCeilingBytes: UInt64 = 1,
    configuredMaximumMlxMemoryGigabytes: UInt64? = nil,
    pendingMlxMemoryCeilingBytes: UInt64? = nil,
    mlxMemoryLimitError: String? = nil,
    servingSession: ServingSession
  ) {
    self.application = application
    self.status = status
    self.activity = activity
    self.configuration = configuration
    self.readyModelIdentifier = readyModelIdentifier
    self.readyModelSizeBytes = readyModelSizeBytes
    self.progress = progress
    self.expertMemoryMode = expertMemoryMode
    self.expertResidency = expertResidency
    self.mtpEnabled = mtpEnabled
    self.mtpConfiguredDraftDepth = mtpConfiguredDraftDepth
    self.mtpArtifactMaximumDraftDepth = mtpArtifactMaximumDraftDepth
    self.mtpArtifactDefaultDraftDepth = mtpArtifactDefaultDraftDepth
    self.mtpResolvedRequestedDraftDepth = mtpResolvedRequestedDraftDepth
    self.mtpEffectiveExecutionDraftDepth = mtpEffectiveExecutionDraftDepth
    self.mtpRuntimeState = mtpRuntimeState
    self.mtpUnavailableReason = mtpUnavailableReason
    self.configuredSpeculativePrefillEnabled = configuredSpeculativePrefillEnabled
    self.speculativePrefillEnabled = speculativePrefillEnabled
    self.workerRuntimeFeatureConfigurationApplied = workerRuntimeFeatureConfigurationApplied
    self.workerRuntimeFeatureConfiguration = workerRuntimeFeatureConfiguration
    self.mlxMemorySnapshot = mlxMemorySnapshot
    self.mlxMemoryCeilingBytes = mlxMemoryCeilingBytes
    self.machineMlxMemoryCeilingBytes = machineMlxMemoryCeilingBytes
    self.minimumMlxMemoryCeilingBytes = minimumMlxMemoryCeilingBytes
    self.configuredMaximumMlxMemoryGigabytes = configuredMaximumMlxMemoryGigabytes
    self.pendingMlxMemoryCeilingBytes = pendingMlxMemoryCeilingBytes
    self.mlxMemoryLimitError = mlxMemoryLimitError
    self.servingSession = servingSession
  }

  static let unavailable = SupervisorStatusDocument(
    status: "unavailable", activity: "idle", readyModelIdentifier: nil, progress: nil,
    expertMemoryMode: nil,
    mlxMemoryCeilingBytes: 0, servingSession: .empty
  )

  var isActive: Bool {
    activity == "prompt_processing" || activity == "generation_preparation"
      || activity == "generating"
  }
  // The status endpoint supplies request-elapsed time during prompt processing
  // and phase-elapsed time during generation. Callers label those boundaries
  // explicitly instead of presenting both as interchangeable model-forward rates.
  var currentPhaseTokensPerSecond: Double? {
    guard let progress, progress.processedTokens > 0, progress.elapsedMilliseconds > 0 else {
      return nil
    }
    return Double(progress.processedTokens) / (Double(progress.elapsedMilliseconds) / 1_000)
  }
  var menuBarTitle: String {
    guard status == "ready" else { return status == "loading" ? " Loading" : "" }
    if activity == "generating", let currentPhaseTokensPerSecond {
      return String(format: "GEN %.1f tok/s", currentPhaseTokensPerSecond)
    }
    if activity == "generation_preparation" { return "Preparing…" }
    guard activity == "prompt_processing", let progress else { return "" }
    let completionPercentageTitle = progress.completionPercentageTitle
    if progress.phase == "drafter" { return "Drafting…" }
    guard let currentPhaseTokensPerSecond else { return "Prompt \(completionPercentageTitle)" }
    return "Prompt \(completionPercentageTitle) · \(Int(currentPhaseTokensPerSecond.rounded())) avg tok/s"
  }
  var flightTitle: String {
    guard isActive else { return "Standing by" }
    if activity == "generating" {
      return currentPhaseTokensPerSecond.map { String(format: "Generating · %.1f tok/s", $0) }
        ?? "Generating"
    }
    if activity == "generation_preparation" { return "Preparing generation…" }
    guard let progress else { return phaseTitle }
    if progress.phase == "drafter" { return "Drafting…" }
    guard let currentPhaseTokensPerSecond else {
      return "Prompt processing · \(progress.completionPercentageTitle)"
    }
    return "Prompt processing · \(progress.completionPercentageTitle) · \(Int(currentPhaseTokensPerSecond.rounded())) avg tok/s"
  }
  var phaseTitle: String {
    switch activity {
    case "generating": "Generating"
    case "generation_preparation": "Preparing generation"
    case "prompt_processing": progress?.phase == "drafter" ? "Drafting…" : "Prompt processing"
    default:
      switch status {
      case "ready": "Ready"
      case "loading": "Loading"
      default: "Unavailable"
      }
    }
  }
  var modelFootprintTitle: String {
    if expertMemoryMode == "resident"
      || (expertMemoryMode == "hybrid"
        && expertResidency?.retainsEveryLayerCompletely == true)
    {
      return "Fully in memory"
    }
    return expertMemoryMode == "paged" || expertMemoryMode == "hybrid"
      ? "RAM + SSD streaming" : "Not loaded"
  }
  var modelDiskSizeTitle: String {
    readyModelSizeBytes.map(decimalGigabyteText) ?? "Not measured"
  }
  var mtpRuntimeStateTitle: String {
    guard readyModelIdentifier != nil else { return "Not loaded" }
    switch mtpRuntimeState {
    case "active": return "Active"
    case "target_only": return "Standard generation"
    case "unavailable": return "Unavailable"
    default: return "Disabled"
    }
  }
  var progressTitle: String {
    guard let progress else { return "Standing by" }
    if progress.phase == "generation_preparation" { return "Preparing the first output" }
    if progress.phase == "drafter" { return "Drafting…" }
    let tokenCountTitle = "\(progress.processedTokens) / \(progress.totalTokens) tokens"
    return progress.phase == "generation"
      ? tokenCountTitle
      : "\(progress.completionPercentageTitle) · \(tokenCountTitle)"
  }
  var elapsedTimeMetricTitle: String {
    progress?.phase == "generation" || progress?.phase == "generation_preparation"
      ? "Elapsed" : "Elapsed / ETA"
  }

  var elapsedTimeTitle: String {
    guard let progress else { return "Not active" }
    let elapsedSeconds = Double(progress.elapsedMilliseconds) / 1_000
    guard progress.phase != "generation" && progress.phase != "generation_preparation" else {
      return String(format: "%.1f s", elapsedSeconds)
    }
    guard progress.processedTokens > 0 else {
      return String(format: "%.1f s / Calculating", elapsedSeconds)
    }
    let remainingTokenCount =
      progress.totalTokens > progress.processedTokens
      ? progress.totalTokens - progress.processedTokens : 0
    let estimatedRemainingSeconds =
      elapsedSeconds * Double(remainingTokenCount) / Double(progress.processedTokens)
    return String(format: "%.1f s / %.1f s", elapsedSeconds, estimatedRemainingSeconds)
  }
  var progressProcessedTokenCount: UInt32 { progress?.processedTokens ?? 0 }
  var progressTotalTokenCount: UInt32 { max(1, progress?.totalTokens ?? 1) }
  var mlxMemoryLimitTitle: String { decimalGigabyteText(byteCount: mlxMemoryCeilingBytes) }
  var mlxMemoryActiveBytes: UInt64 { mlxMemorySnapshot?.activeMemoryBytes ?? 0 }
  var mlxMemorySourceTitle: String {
    switch mlxMemorySnapshot?.source {
    case "model_loaded": "Model loaded"
    case "prefill": "Prompt snapshot"
    case "speculative_prefill_draft_scoring": "Live drafter scoring"
    case "decode_submitted": "Live decode"
    case "finalized": "After cleanup"
    case "idle_poll": "Idle sample"
    case "memory_limit_adjusted": "Memory limit adjusted"
    default: "Not measured"
    }
  }
  var mlxMemoryBreakdown: MlxMemoryBreakdown {
    let reconciledExpertPayloadByteCount = min(
      mlxMemorySnapshot?.expertPayloadBytes ?? 0,
      mlxMemoryActiveBytes
    )
    let activeBytesAfterExperts = mlxMemoryActiveBytes.saturatingSubtracting(
      reconciledExpertPayloadByteCount)
    let reconciledModelCorePayloadByteCount = min(
      mlxMemorySnapshot?.modelCorePayloadBytes ?? 0,
      activeBytesAfterExperts
    )
    let activeBytesAfterModelCore = activeBytesAfterExperts.saturatingSubtracting(
      reconciledModelCorePayloadByteCount)
    let reconciledContextStatePayloadByteCount = min(
      mlxMemorySnapshot?.contextStatePayloadBytes ?? 0,
      activeBytesAfterModelCore
    )
    let activeBytesAfterContextState = activeBytesAfterModelCore.saturatingSubtracting(
      reconciledContextStatePayloadByteCount)
    let reconciledSpeculativePrefillDraftMemoryByteCount = min(
      mlxMemorySnapshot?.speculativePrefillDraftMemoryBytes ?? 0,
      activeBytesAfterContextState
    )
    let reconciledRuntimeWorkByteCount = activeBytesAfterContextState.saturatingSubtracting(
      reconciledSpeculativePrefillDraftMemoryByteCount)
    return MlxMemoryBreakdown(
      expertPayloadByteCount: reconciledExpertPayloadByteCount,
      modelCorePayloadByteCount: reconciledModelCorePayloadByteCount,
      contextStatePayloadByteCount: reconciledContextStatePayloadByteCount,
      speculativePrefillDraftMemoryByteCount:
        reconciledSpeculativePrefillDraftMemoryByteCount,
      runtimeWorkByteCount: reconciledRuntimeWorkByteCount,
      availableByteCount: mlxMemoryCeilingBytes.saturatingSubtracting(
        mlxMemoryActiveBytes)
    )
  }
  var sessionTitle: String {
    let requestCount = servingSession.completedRequestCount
    return
      "\(requestCount) \(requestCount == 1 ? "request" : "requests") · \(Int(servingSession.averageGenerationTokensPerSecond.rounded())) tok/s avg"
  }
  private var sessionPromptReuse:
    (
      reusedPromptTokenCount: UInt64, newPromptTokenCount: UInt64
    )?
  {
    let combinedPromptWorkTokenCount = servingSession.targetPromptWorkTokenCount.saturatingAdding(
      servingSession.drafterPromptWorkTokenCount
    )
    let combinedReusedPromptWorkTokenCount = servingSession.targetReusedPromptWorkTokenCount.saturatingAdding(
      servingSession.drafterReusedPromptWorkTokenCount
    )
    let promptTokenCount = combinedPromptWorkTokenCount > 0
      ? combinedPromptWorkTokenCount
      : servingSession.totalPromptTokenCount
    guard promptTokenCount > 0 else { return nil }
    let reusedPromptTokenCount = min(
      combinedPromptWorkTokenCount > 0
        ? combinedReusedPromptWorkTokenCount
        : servingSession.totalReusedPromptTokenCount,
      promptTokenCount
    )
    let newPromptTokenCount = promptTokenCount - reusedPromptTokenCount
    return (reusedPromptTokenCount, newPromptTokenCount)
  }
  var sessionPromptReusePercentageTitle: String {
    guard let sessionPromptReuse else { return "Not measured" }
    return promptReusePercentageText(
      reusedPromptTokenCount: sessionPromptReuse.reusedPromptTokenCount,
      totalPromptTokenCount: sessionPromptReuse.reusedPromptTokenCount
        + sessionPromptReuse.newPromptTokenCount
    )
  }
  var sessionPromptReuseFraction: Double {
    guard let sessionPromptReuse else { return 0 }
    return Double(sessionPromptReuse.reusedPromptTokenCount)
      / Double(sessionPromptReuse.reusedPromptTokenCount + sessionPromptReuse.newPromptTokenCount)
  }
  var sessionPromptReuseBreakdownTitle: String {
    guard let sessionPromptReuse else { return "No completed prompts" }
    return
      "\(groupedTokenCountText(sessionPromptReuse.reusedPromptTokenCount)) reused · \(groupedTokenCountText(sessionPromptReuse.newPromptTokenCount)) new"
  }
}

extension SupervisorStatusDocument.Progress {
  var completionPercentageTitle: String {
    guard totalTokens > 0 else { return "0%" }
    let boundedProcessedTokens = min(processedTokens, totalTokens)
    return "\(Int((Double(boundedProcessedTokens) / Double(totalTokens) * 100).rounded(.down)))%"
  }
}

extension UInt64 {
  fileprivate func saturatingAdding(_ tokenCount: UInt64) -> UInt64 {
    let (summedTokenCount, didOverflow) = addingReportingOverflow(tokenCount)
    return didOverflow ? UInt64.max : summedTokenCount
  }

  fileprivate func saturatingSubtracting(_ byteCount: UInt64) -> UInt64 {
    self >= byteCount ? self - byteCount : 0
  }
}

func groupedTokenCountText(_ tokenCount: UInt64) -> String {
  tokenCount.formatted(
    .number.grouping(.automatic).locale(Locale(identifier: "en_US"))
  )
}

func promptReusePercentageText(
  reusedPromptTokenCount: UInt64,
  totalPromptTokenCount: UInt64
) -> String {
  guard totalPromptTokenCount > 0 else { return "Not measured" }
  let boundedReusedPromptTokenCount = min(reusedPromptTokenCount, totalPromptTokenCount)
  if boundedReusedPromptTokenCount == totalPromptTokenCount { return "100%" }
  let reusedPercentageTenths = min(
    999,
    Int(
      floor(
        Double(boundedReusedPromptTokenCount) / Double(totalPromptTokenCount) * 1_000
      )
    )
  )
  if reusedPercentageTenths.isMultiple(of: 10) {
    return "\(reusedPercentageTenths / 10)%"
  }
  return String(format: "%.1f%%", Double(reusedPercentageTenths) / 10)
}

func decimalGigabyteText(byteCount: UInt64) -> String {
  guard byteCount > 0 else { return "Not measured" }
  return decimalGigabyteValueText(byteCount: byteCount)
}

func decimalGigabyteValueText(byteCount: UInt64) -> String {
  String(format: "%.2f GB", Double(byteCount) / 1_000_000_000)
}
