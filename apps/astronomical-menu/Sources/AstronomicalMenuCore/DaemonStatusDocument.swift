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

struct DaemonStatusDocument: Codable, Equatable {
  struct MlxMemoryBreakdown: Equatable {
    let expertPayloadByteCount: UInt64
    let modelCorePayloadByteCount: UInt64
    let contextStatePayloadByteCount: UInt64
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
    let memoryCeilingUtilization: MlxMemoryCeilingUtilization?

    enum CodingKeys: String, CodingKey {
      case source
      case activeMemoryBytes = "active_memory_bytes"
      case allocatorCacheMemoryBytes = "allocator_cache_memory_bytes"
      case peakMemoryBytes = "peak_memory_bytes"
      case expertPayloadBytes = "expert_payload_bytes"
      case modelCorePayloadBytes = "model_core_payload_bytes"
      case contextStatePayloadBytes = "context_state_payload_bytes"
      case memoryCeilingUtilization = "memory_ceiling_utilization"
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
      memoryCeilingUtilization =
        try container.decodeIfPresent(
          MlxMemoryCeilingUtilization.self, forKey: .memoryCeilingUtilization)
    }
  }
  struct ServingSession: Codable, Equatable {
    let completedRequestCount: UInt64
    let totalPromptTokenCount: UInt64
    let totalReusedPromptTokenCount: UInt64
    let targetPromptWorkTokenCount: UInt64
    let targetReusedPromptWorkTokenCount: UInt64
    let averagePrefillTokensPerSecond: Double
    let averageGenerationTokensPerSecond: Double

    static let empty = ServingSession(
      completedRequestCount: 0,
      totalPromptTokenCount: 0,
      totalReusedPromptTokenCount: 0,
      targetPromptWorkTokenCount: 0,
      targetReusedPromptWorkTokenCount: 0,
      averagePrefillTokensPerSecond: 0,
      averageGenerationTokensPerSecond: 0
    )

    enum CodingKeys: String, CodingKey {
      case completedRequestCount = "completed_request_count"
      case totalPromptTokenCount = "total_prompt_token_count"
      case totalReusedPromptTokenCount = "total_reused_prompt_token_count"
      case targetPromptWorkTokenCount = "target_prompt_work_token_count"
      case targetReusedPromptWorkTokenCount = "target_reused_prompt_work_token_count"
      case averagePrefillTokensPerSecond = "average_prefill_tok_per_second"
      case averageGenerationTokensPerSecond = "average_generation_tok_per_second"
    }

    init(
      completedRequestCount: UInt64,
      totalPromptTokenCount: UInt64,
      totalReusedPromptTokenCount: UInt64,
      targetPromptWorkTokenCount: UInt64,
      targetReusedPromptWorkTokenCount: UInt64,
      averagePrefillTokensPerSecond: Double,
      averageGenerationTokensPerSecond: Double
    ) {
      self.completedRequestCount = completedRequestCount
      self.totalPromptTokenCount = totalPromptTokenCount
      self.totalReusedPromptTokenCount = totalReusedPromptTokenCount
      self.targetPromptWorkTokenCount = targetPromptWorkTokenCount
      self.targetReusedPromptWorkTokenCount = targetReusedPromptWorkTokenCount
      self.averagePrefillTokensPerSecond = averagePrefillTokensPerSecond
      self.averageGenerationTokensPerSecond = averageGenerationTokensPerSecond
    }

    init(from decoder: Decoder) throws {
      let container = try decoder.container(keyedBy: CodingKeys.self)
      // Legacy status documents omit these counters; every missing field
      // falls back to a zero that still renders as "no work recorded yet".
      let decodedCompletedRequestCount = try container.decodeIfPresent(
        UInt64.self, forKey: .completedRequestCount)
      if let decodedCompletedRequestCount {
        completedRequestCount = decodedCompletedRequestCount
      } else {
        completedRequestCount = 0
      }
      let decodedTotalPromptTokenCount = try container.decodeIfPresent(
        UInt64.self, forKey: .totalPromptTokenCount)
      if let decodedTotalPromptTokenCount {
        totalPromptTokenCount = decodedTotalPromptTokenCount
      } else {
        totalPromptTokenCount = 0
      }
      let decodedTotalReusedPromptTokenCount = try container.decodeIfPresent(
        UInt64.self, forKey: .totalReusedPromptTokenCount)
      if let decodedTotalReusedPromptTokenCount {
        totalReusedPromptTokenCount = decodedTotalReusedPromptTokenCount
      } else {
        totalReusedPromptTokenCount = 0
      }
      let decodedTargetPromptWorkTokenCount = try container.decodeIfPresent(
        UInt64.self, forKey: .targetPromptWorkTokenCount)
      if let decodedTargetPromptWorkTokenCount {
        targetPromptWorkTokenCount = decodedTargetPromptWorkTokenCount
      } else {
        targetPromptWorkTokenCount = 0
      }
      let decodedTargetReusedPromptWorkTokenCount = try container.decodeIfPresent(
        UInt64.self, forKey: .targetReusedPromptWorkTokenCount)
      if let decodedTargetReusedPromptWorkTokenCount {
        targetReusedPromptWorkTokenCount = decodedTargetReusedPromptWorkTokenCount
      } else {
        targetReusedPromptWorkTokenCount = 0
      }
      let decodedAveragePrefillTokensPerSecond = try container.decodeIfPresent(
        Double.self, forKey: .averagePrefillTokensPerSecond)
      if let decodedAveragePrefillTokensPerSecond {
        averagePrefillTokensPerSecond = decodedAveragePrefillTokensPerSecond
      } else {
        averagePrefillTokensPerSecond = 0
      }
      let decodedAverageGenerationTokensPerSecond = try container.decodeIfPresent(
        Double.self, forKey: .averageGenerationTokensPerSecond)
      if let decodedAverageGenerationTokensPerSecond {
        averageGenerationTokensPerSecond = decodedAverageGenerationTokensPerSecond
      } else {
        averageGenerationTokensPerSecond = 0
      }
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
  // True only when the current worker has emitted its runtime configuration event; configuration
  // intent alone is insufficient because a replacement worker can still fail before applying it.
  let workerRuntimeFeatureConfigurationApplied: Bool
  // The complete acknowledgement lets control actions compare the exact policy they requested
  // with the policy currently served, rather than inferring it from one derived feature flag.
  let workerRuntimeFeatureConfiguration: WorkerRuntimeFeatureConfiguration?
  let mlxMemorySnapshot: MlxMemorySnapshot?
  let mlxMemoryCeilingBytes: UInt64
  let machineMlxMemoryCeilingBytes: UInt64
  // GPU memory macOS granted the worker as a wired pool; nil when serving is
  // pageable with no wired claim. The effective ceiling never exceeds it, so
  // the memory panel can name the operating-system bound whenever it clamps.
  let wiredResidencyBudgetBytes: UInt64?
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
    case workerRuntimeFeatureConfigurationApplied = "worker_runtime_feature_configuration_applied"
    case workerRuntimeFeatureConfiguration = "worker_runtime_feature_configuration"
    case mlxMemorySnapshot = "mlx_memory_snapshot"
    case mlxMemoryCeilingBytes = "mlx_memory_ceiling_bytes"
    case machineMlxMemoryCeilingBytes = "machine_mlx_memory_ceiling_bytes"
    case wiredResidencyBudgetBytes = "wired_residency_budget_bytes"
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
    let decodedWorkerRuntimeFeatureConfigurationApplied = try container.decodeIfPresent(
      Bool.self, forKey: .workerRuntimeFeatureConfigurationApplied)
    if let decodedWorkerRuntimeFeatureConfigurationApplied {
      workerRuntimeFeatureConfigurationApplied = decodedWorkerRuntimeFeatureConfigurationApplied
    } else {
      workerRuntimeFeatureConfigurationApplied = false
    }
    workerRuntimeFeatureConfiguration = try container.decodeIfPresent(
      WorkerRuntimeFeatureConfiguration.self, forKey: .workerRuntimeFeatureConfiguration)
    mlxMemorySnapshot = try container.decodeIfPresent(MlxMemorySnapshot.self, forKey: .mlxMemorySnapshot)
    let decodedMlxMemoryCeilingBytes = try container.decodeIfPresent(
      UInt64.self, forKey: .mlxMemoryCeilingBytes)
    if let decodedMlxMemoryCeilingBytes {
      mlxMemoryCeilingBytes = decodedMlxMemoryCeilingBytes
    } else {
      mlxMemoryCeilingBytes = 0
    }
    let decodedMachineMlxMemoryCeilingBytes = try container.decodeIfPresent(
      UInt64.self, forKey: .machineMlxMemoryCeilingBytes)
    if let decodedMachineMlxMemoryCeilingBytes {
      machineMlxMemoryCeilingBytes = decodedMachineMlxMemoryCeilingBytes
    } else {
      machineMlxMemoryCeilingBytes = 0
    }
    wiredResidencyBudgetBytes =
      try container.decodeIfPresent(UInt64.self, forKey: .wiredResidencyBudgetBytes)
    let decodedMinimumMlxMemoryCeilingBytes = try container.decodeIfPresent(
      UInt64.self, forKey: .minimumMlxMemoryCeilingBytes)
    if let decodedMinimumMlxMemoryCeilingBytes {
      minimumMlxMemoryCeilingBytes = decodedMinimumMlxMemoryCeilingBytes
    } else {
      minimumMlxMemoryCeilingBytes = 1
    }
    configuredMaximumMlxMemoryGigabytes =
      try container.decodeIfPresent(UInt64.self, forKey: .configuredMaximumMlxMemoryGigabytes)
    pendingMlxMemoryCeilingBytes =
      try container.decodeIfPresent(UInt64.self, forKey: .pendingMlxMemoryCeilingBytes)
    mlxMemoryLimitError = try container.decodeIfPresent(String.self, forKey: .mlxMemoryLimitError)
    let decodedServingSession = try container.decodeIfPresent(
      ServingSession.self, forKey: .servingSession)
    if let decodedServingSession {
      servingSession = decodedServingSession
    } else {
      servingSession = .empty
    }
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
    workerRuntimeFeatureConfigurationApplied: Bool = false,
    workerRuntimeFeatureConfiguration: WorkerRuntimeFeatureConfiguration? = nil,
    mlxMemorySnapshot: MlxMemorySnapshot? = nil,
    mlxMemoryCeilingBytes: UInt64,
    machineMlxMemoryCeilingBytes: UInt64 = 0,
    wiredResidencyBudgetBytes: UInt64? = nil,
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
    self.workerRuntimeFeatureConfigurationApplied = workerRuntimeFeatureConfigurationApplied
    self.workerRuntimeFeatureConfiguration = workerRuntimeFeatureConfiguration
    self.mlxMemorySnapshot = mlxMemorySnapshot
    self.mlxMemoryCeilingBytes = mlxMemoryCeilingBytes
    self.machineMlxMemoryCeilingBytes = machineMlxMemoryCeilingBytes
    self.wiredResidencyBudgetBytes = wiredResidencyBudgetBytes
    self.minimumMlxMemoryCeilingBytes = minimumMlxMemoryCeilingBytes
    self.configuredMaximumMlxMemoryGigabytes = configuredMaximumMlxMemoryGigabytes
    self.pendingMlxMemoryCeilingBytes = pendingMlxMemoryCeilingBytes
    self.mlxMemoryLimitError = mlxMemoryLimitError
    self.servingSession = servingSession
  }

  static let unavailable = DaemonStatusDocument(
    status: "unavailable", activity: "idle", readyModelIdentifier: nil, progress: nil,
    expertMemoryMode: nil,
    mlxMemoryCeilingBytes: 0, servingSession: .empty
  )

  var mlxMemoryLimitTitle: String { gigabyteText(byteCount: mlxMemoryCeilingBytes) }
  var mlxMemoryActiveBytes: UInt64 {
    guard let mlxMemorySnapshot else { return 0 }
    return mlxMemorySnapshot.activeMemoryBytes
  }
  var mlxMemorySourceTitle: String {
    switch mlxMemorySnapshot?.source {
    case "model_loaded": "Model loaded"
    case "prefill": "Prompt snapshot"
    case "decode_submitted": "Live decode"
    case "finalized": "After cleanup"
    case "idle_poll": "Idle sample"
    case "memory_limit_adjusted": "Memory limit adjusted"
    case "image_generation_step": "Image render snapshot"
    default: "Not measured"
    }
  }
  var mlxMemoryBreakdown: MlxMemoryBreakdown {
    // Each owner is reconciled against the memory still unclaimed by the
    // owners above it; a snapshot without the field observes zero bytes.
    let observedExpertPayloadBytes: UInt64
    if let expertPayloadBytes = mlxMemorySnapshot?.expertPayloadBytes {
      observedExpertPayloadBytes = expertPayloadBytes
    } else {
      observedExpertPayloadBytes = 0
    }
    let reconciledExpertPayloadByteCount = min(
      observedExpertPayloadBytes,
      mlxMemoryActiveBytes
    )
    let activeBytesAfterExperts = mlxMemoryActiveBytes.saturatingSubtracting(
      reconciledExpertPayloadByteCount)
    let observedModelCorePayloadBytes: UInt64
    if let modelCorePayloadBytes = mlxMemorySnapshot?.modelCorePayloadBytes {
      observedModelCorePayloadBytes = modelCorePayloadBytes
    } else {
      observedModelCorePayloadBytes = 0
    }
    let reconciledModelCorePayloadByteCount = min(
      observedModelCorePayloadBytes,
      activeBytesAfterExperts
    )
    let activeBytesAfterModelCore = activeBytesAfterExperts.saturatingSubtracting(
      reconciledModelCorePayloadByteCount)
    let observedContextStatePayloadBytes: UInt64
    if let contextStatePayloadBytes = mlxMemorySnapshot?.contextStatePayloadBytes {
      observedContextStatePayloadBytes = contextStatePayloadBytes
    } else {
      observedContextStatePayloadBytes = 0
    }
    let reconciledContextStatePayloadByteCount = min(
      observedContextStatePayloadBytes,
      activeBytesAfterModelCore
    )
    let activeBytesAfterContextState = activeBytesAfterModelCore.saturatingSubtracting(
      reconciledContextStatePayloadByteCount)
    let reconciledRuntimeWorkByteCount = activeBytesAfterContextState
    return MlxMemoryBreakdown(
      expertPayloadByteCount: reconciledExpertPayloadByteCount,
      modelCorePayloadByteCount: reconciledModelCorePayloadByteCount,
      contextStatePayloadByteCount: reconciledContextStatePayloadByteCount,
      runtimeWorkByteCount: reconciledRuntimeWorkByteCount,
      availableByteCount: mlxMemoryCeilingBytes.saturatingSubtracting(
        mlxMemoryActiveBytes)
    )
  }
  var mlxHeadroomSplit: MlxHeadroomSplit {
    MlxHeadroomSplit.from(
      utilization: mlxMemorySnapshot?.memoryCeilingUtilization,
      availableByteCount: mlxMemoryBreakdown.availableByteCount
    )
  }

  /// The wired GPU budget only when it clamps below the machine ceiling, so
  /// the memory panel names the operating-system bound instead of presenting
  /// an unexplained smaller effective number. `nil` when serving is pageable
  /// or when macOS grants at least the machine ceiling (issue #1282).
  var clampedWiredResidencyBudgetBytes: UInt64? {
    guard let wiredResidencyBudgetBytes,
      wiredResidencyBudgetBytes < machineMlxMemoryCeilingBytes
    else { return nil }
    return wiredResidencyBudgetBytes
  }

  /// Plain-language title for the operating-system GPU lock limit, shown only
  /// while the limit actually clamps the ceiling.
  var wiredResidencyLimitTitle: String? {
    clampedWiredResidencyBudgetBytes.map { gigabyteValueText(byteCount: $0) }
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
    let targetPromptWorkTokenCount = servingSession.targetPromptWorkTokenCount
    let promptTokenCount = targetPromptWorkTokenCount > 0
      ? targetPromptWorkTokenCount
      : servingSession.totalPromptTokenCount
    guard promptTokenCount > 0 else { return nil }
    let reusedPromptTokenCount = min(
      targetPromptWorkTokenCount > 0
        ? servingSession.targetReusedPromptWorkTokenCount
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

extension UInt64 {
  fileprivate func saturatingAdding(_ tokenCount: UInt64) -> UInt64 {
    let (summedTokenCount, didOverflow) = addingReportingOverflow(tokenCount)
    return didOverflow ? UInt64.max : summedTokenCount
  }

  func saturatingSubtracting(_ byteCount: UInt64) -> UInt64 {
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

func gigabyteText(byteCount: UInt64) -> String {
  guard byteCount > 0 else { return "Not measured" }
  return gigabyteValueText(byteCount: byteCount)
}

func gigabyteValueText(byteCount: UInt64) -> String {
  // The user-facing gigabyte is the one Apple prints on the box: 2³⁰ bytes.
  String(format: "%.2f GB", Double(byteCount) / 1_073_741_824)
}
