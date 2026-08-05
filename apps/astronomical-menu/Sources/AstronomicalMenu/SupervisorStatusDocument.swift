import Foundation

struct SupervisorStatusDocument: Codable, Equatable {
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

    enum CodingKeys: String, CodingKey {
      case source
      case activeMemoryBytes = "active_memory_bytes"
      case allocatorCacheMemoryBytes = "allocator_cache_memory_bytes"
      case peakMemoryBytes = "peak_memory_bytes"
      case expertPayloadBytes = "expert_payload_bytes"
      case modelCorePayloadBytes = "model_core_payload_bytes"
      case contextStatePayloadBytes = "context_state_payload_bytes"
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
    let averagePrefillTokensPerSecond: Double
    let averageGenerationTokensPerSecond: Double

    static let empty = ServingSession(
      completedRequestCount: 0,
      totalPromptTokenCount: 0,
      totalReusedPromptTokenCount: 0,
      averagePrefillTokensPerSecond: 0,
      averageGenerationTokensPerSecond: 0
    )

    enum CodingKeys: String, CodingKey {
      case completedRequestCount = "completed_request_count"
      case totalPromptTokenCount = "total_prompt_token_count"
      case totalReusedPromptTokenCount = "total_reused_prompt_token_count"
      case averagePrefillTokensPerSecond = "average_prefill_tok_per_second"
      case averageGenerationTokensPerSecond = "average_generation_tok_per_second"
    }
  }

  let status: String
  let activity: String
  let readyModelIdentifier: String?
  let readyModelSizeBytes: UInt64?
  let progress: Progress?
  let expertMemoryMode: String?
  let mtpEnabled: Bool
  let mtpRuntimeState: String
  let mtpUnavailableReason: String?
  let mlxMemorySnapshot: MlxMemorySnapshot?
  let mlxMemoryCeilingBytes: UInt64
  let machineMlxMemoryCeilingBytes: UInt64
  let minimumMlxMemoryCeilingBytes: UInt64
  let configuredMaximumMlxMemoryGigabytes: UInt64?
  let pendingMlxMemoryCeilingBytes: UInt64?
  let mlxMemoryLimitError: String?
  let servingSession: ServingSession

  enum CodingKeys: String, CodingKey {
    case status, activity, progress
    case readyModelIdentifier = "ready_model_id"
    case readyModelSizeBytes = "ready_model_size_bytes"
    case expertMemoryMode = "expert_memory_mode"
    case mtpEnabled = "mtp_enabled"
    case mtpRuntimeState = "mtp_runtime_state"
    case mtpUnavailableReason = "mtp_unavailable_reason"
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
    status = try container.decode(String.self, forKey: .status)
    activity = try container.decode(String.self, forKey: .activity)
    readyModelIdentifier = try container.decodeIfPresent(String.self, forKey: .readyModelIdentifier)
    readyModelSizeBytes = try container.decodeIfPresent(UInt64.self, forKey: .readyModelSizeBytes)
    progress = try container.decodeIfPresent(Progress.self, forKey: .progress)
    expertMemoryMode = try container.decodeIfPresent(String.self, forKey: .expertMemoryMode)
    mtpEnabled = try container.decodeIfPresent(Bool.self, forKey: .mtpEnabled) ?? false
    mtpRuntimeState = try container.decodeIfPresent(String.self, forKey: .mtpRuntimeState) ?? "disabled"
    mtpUnavailableReason = try container.decodeIfPresent(String.self, forKey: .mtpUnavailableReason)
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
    status: String,
    activity: String,
    readyModelIdentifier: String?,
    readyModelSizeBytes: UInt64? = nil,
    progress: Progress?,
    expertMemoryMode: String?,
    mtpEnabled: Bool = false,
    mtpRuntimeState: String = "disabled",
    mtpUnavailableReason: String? = nil,
    mlxMemorySnapshot: MlxMemorySnapshot? = nil,
    mlxMemoryCeilingBytes: UInt64,
    machineMlxMemoryCeilingBytes: UInt64 = 0,
    minimumMlxMemoryCeilingBytes: UInt64 = 1,
    configuredMaximumMlxMemoryGigabytes: UInt64? = nil,
    pendingMlxMemoryCeilingBytes: UInt64? = nil,
    mlxMemoryLimitError: String? = nil,
    servingSession: ServingSession
  ) {
    self.status = status
    self.activity = activity
    self.readyModelIdentifier = readyModelIdentifier
    self.readyModelSizeBytes = readyModelSizeBytes
    self.progress = progress
    self.expertMemoryMode = expertMemoryMode
    self.mtpEnabled = mtpEnabled
    self.mtpRuntimeState = mtpRuntimeState
    self.mtpUnavailableReason = mtpUnavailableReason
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

  var isActive: Bool { activity == "prompt_processing" || activity == "generating" }
  var currentRate: Double? {
    guard let progress, progress.elapsedMilliseconds > 0 else { return nil }
    return Double(progress.processedTokens) / (Double(progress.elapsedMilliseconds) / 1_000)
  }
  var menuBarTitle: String {
    guard status == "ready" else { return status == "loading" ? " Loading" : "" }
    guard let currentRate else { return "" }
    return activity == "generating"
      ? String(format: "GEN %.1f tok/s", currentRate)
      : "PP \(Int(currentRate.rounded())) tok/s"
  }
  var flightTitle: String {
    guard let currentRate else { return "Standing by" }
    return activity == "generating"
      ? String(format: "Generating · %.1f tok/s", currentRate)
      : "Prompt processing · \(Int(currentRate.rounded())) tok/s"
  }
  var phaseTitle: String {
    switch activity {
    case "generating": "Generating"
    case "prompt_processing": "Prompt processing"
    default:
      switch status {
      case "ready": "Ready"
      case "loading": "Loading"
      default: "Unavailable"
      }
    }
  }
  var modelFootprintTitle: String {
    expertMemoryMode == "paged"
      ? "RAM + SSD streaming" : expertMemoryMode == "resident" ? "Fully in memory" : "Not loaded"
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
    progress.map { "\($0.processedTokens) / \($0.totalTokens) tokens" } ?? "Standing by"
  }
  var elapsedTimeMetricTitle: String {
    progress?.phase == "generation" ? "Elapsed" : "Elapsed / ETA"
  }

  var elapsedTimeTitle: String {
    guard let progress else { return "Not active" }
    let elapsedSeconds = Double(progress.elapsedMilliseconds) / 1_000
    guard progress.phase != "generation" else {
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
    let reconciledRuntimeWorkByteCount = activeBytesAfterModelCore.saturatingSubtracting(
      reconciledContextStatePayloadByteCount)
    return MlxMemoryBreakdown(
      expertPayloadByteCount: reconciledExpertPayloadByteCount,
      modelCorePayloadByteCount: reconciledModelCorePayloadByteCount,
      contextStatePayloadByteCount: reconciledContextStatePayloadByteCount,
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
    let promptTokenCount = servingSession.totalPromptTokenCount
    guard promptTokenCount > 0 else { return nil }
    let reusedPromptTokenCount = min(
      servingSession.totalReusedPromptTokenCount,
      promptTokenCount
    )
    let newPromptTokenCount = promptTokenCount - reusedPromptTokenCount
    return (reusedPromptTokenCount, newPromptTokenCount)
  }
  var sessionPromptReusePercentageTitle: String {
    guard let sessionPromptReuse else { return "Not measured" }
    return promptReusePercentageText(
      reusedPromptTokenCount: sessionPromptReuse.reusedPromptTokenCount,
      totalPromptTokenCount: servingSession.totalPromptTokenCount
    )
  }
  var sessionPromptReuseFraction: Double {
    guard let sessionPromptReuse else { return 0 }
    return Double(sessionPromptReuse.reusedPromptTokenCount)
      / Double(servingSession.totalPromptTokenCount)
  }
  var sessionPromptReuseBreakdownTitle: String {
    guard let sessionPromptReuse else { return "No completed prompts" }
    return
      "\(groupedTokenCountText(sessionPromptReuse.reusedPromptTokenCount)) reused · \(groupedTokenCountText(sessionPromptReuse.newPromptTokenCount)) new"
  }
}

extension UInt64 {
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
