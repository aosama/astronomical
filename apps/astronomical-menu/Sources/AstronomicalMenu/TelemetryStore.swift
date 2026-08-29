import Foundation

private let maximumStatusRefreshErrorCharacterCount = 512
private let maximumWorkerPolicyConfirmationAttempts = 3
private let workerPolicyConfirmationRetryDelay = Duration.milliseconds(100)

enum ControlActionFeedback: Equatable {
  case inProgress(String)
  case success(String)
  case failure(String)

  var message: String {
    switch self {
    case let .inProgress(message), let .success(message), let .failure(message): message
    }
  }

  var isFailure: Bool {
    if case .failure = self { return true }
    return false
  }

  var isInProgress: Bool {
    if case .inProgress = self { return true }
    return false
  }
}

@MainActor
final class TelemetryStore: ObservableObject {
  @Published private(set) var statusDocument = SupervisorStatusDocument.unavailable
  @Published private(set) var systemTelemetrySnapshot = SystemTelemetrySnapshot.unavailable
  @Published private(set) var controlActionFeedback: ControlActionFeedback?
  @Published private(set) var lastStatusRefreshErrorMessage: String?
  @Published private(set) var hasDiscoveredModels = false
  @Published var editableMaximumMlxMemoryGigabytes: UInt64 = 0
  var onMenuBarTitleChanged: ((String) -> Void)?

  private let supervisorClient: any SupervisorClient
  private let systemTelemetrySampler = SystemTelemetrySampler()
  private let systemTelemetryClock = ContinuousClock()
  private var pollingTask: Task<Void, Never>?
  private var latestStatusRefreshTask: Task<SupervisorStatusDocument, Never>?
  private var latestStatusRefreshSequence = 0
  private var controlActionFeedbackDismissalTask: Task<Void, Never>?
  private var controlActionFeedbackGeneration = 0
  private var pendingMaximumMlxMemoryCeilingBytes: UInt64?
  private var popoverIsVisible = false
  private var lastSystemTelemetrySampleTime: ContinuousClock.Instant?

  init(supervisorClient: any SupervisorClient = LocalSupervisorClient()) {
    self.supervisorClient = supervisorClient
  }

  func startPolling() {
    pollingTask?.cancel()
    pollingTask = Task { [weak self] in
      while !Task.isCancelled {
        await self?.refresh()
        let refreshInterval =
          self?.statusDocument.isActive == true
          ? Duration.milliseconds(250) : Duration.seconds(1)
        try? await Task.sleep(for: refreshInterval)
      }
    }
  }

  func stopPolling() {
    pollingTask?.cancel()
    pollingTask = nil
  }

  func refreshNow() { Task { await refresh() } }

  func setPopoverVisible(_ popoverIsVisible: Bool) {
    self.popoverIsVisible = popoverIsVisible
    if popoverIsVisible { updateSystemTelemetryIfDue(forceUpdate: true) }
  }

  func reloadConfiguration() {
    Task { await performConfigurationReload() }
  }

  func updateMaximumMlxMemoryLimit() {
    Task { await performMaximumMlxMemoryLimitUpdate(editableMaximumMlxMemoryGigabytes) }
  }

  func restoreMacMaximumMlxMemoryLimit() {
    Task { await performMaximumMlxMemoryLimitUpdate(nil) }
  }

  func performMaximumMlxMemoryLimitUpdate(_ maximumMlxMemoryGigabytes: UInt64?) async {
    pendingMaximumMlxMemoryCeilingBytes = requestedMaximumMlxMemoryCeilingBytes(
      maximumMlxMemoryGigabytes)
    let memoryUpdateFeedbackGeneration = presentControlActionFeedback(
      .inProgress("Updating maximum model RAM…"))
    do {
      let updateMessage = try await supervisorClient.updateMaximumMlxMemoryGigabytes(
        maximumMlxMemoryGigabytes)
      guard controlActionFeedbackGeneration == memoryUpdateFeedbackGeneration else { return }
      let successFeedbackGeneration = presentControlActionFeedback(.success(updateMessage))
      let refreshedStatusDocument = await refresh()
      guard controlActionFeedbackGeneration == successFeedbackGeneration else { return }
      editableMaximumMlxMemoryGigabytes = refreshedStatusDocument.configuredMaximumMlxMemoryGigabytes
        ?? maximumWholeDecimalGigabytes(for: refreshedStatusDocument)
    } catch {
      guard controlActionFeedbackGeneration == memoryUpdateFeedbackGeneration else { return }
      pendingMaximumMlxMemoryCeilingBytes = nil
      presentControlActionFeedback(.failure(controlActionErrorMessage(error)))
    }
  }

  func performConfigurationReload() async {
    pendingMaximumMlxMemoryCeilingBytes = nil
    let configurationReloadFeedbackGeneration = presentControlActionFeedback(
      .inProgress("Reloading configuration…"))
    do {
      let configurationReloadResult = try await supervisorClient.reloadConfiguration()
      guard controlActionFeedbackGeneration == configurationReloadFeedbackGeneration else {
        return
      }
      if configurationReloadResult.candidateGeneration
        != configurationReloadResult.effectiveGeneration
      {
        await refresh()
        presentControlActionFeedback(
          configurationReloadResult.restApiRestartRequired == true
            ? .inProgress(configurationReloadResult.message)
            : .failure(configurationReloadResult.message))
        return
      }
      // A restart response proves the replacement worker acknowledged its startup policy, but
      // the menu must also observe that exact policy in a fresh status document. Otherwise a
      // poll racing with worker replacement could turn stale readiness into a false success.
      if configurationReloadResult.workerRestartCompleted {
        guard controlActionFeedbackGeneration == configurationReloadFeedbackGeneration else {
          return
        }
        let workerPolicyWasConfirmed = await workerPolicyIsConfirmed(
          configurationReloadResult.workerRuntimeFeatureConfiguration)
        guard controlActionFeedbackGeneration == configurationReloadFeedbackGeneration else {
          return
        }
        guard workerPolicyWasConfirmed else {
          presentControlActionFeedback(
            .failure("Worker restart completed, but its applied configuration was not confirmed")
          )
          return
        }
      }
      presentControlActionFeedback(.success(configurationReloadResult.message))
    } catch {
      guard controlActionFeedbackGeneration == configurationReloadFeedbackGeneration else {
        return
      }
      presentControlActionFeedback(.failure(controlActionErrorMessage(error)))
    }
    // Non-restart edits still refresh telemetry, while restart edits already refreshed above
    // before success feedback was allowed.
    await refresh()
  }

  private func workerPolicyIsConfirmed(
    _ acknowledgedWorkerPolicy: WorkerRuntimeFeatureConfiguration?
  ) async -> Bool {
    for confirmationAttempt in 1...maximumWorkerPolicyConfirmationAttempts {
      let refreshedStatusDocument = await refresh()
      if refreshedStatusDocument.workerRuntimeFeatureConfigurationApplied,
        refreshedStatusDocument.workerRuntimeFeatureConfiguration == acknowledgedWorkerPolicy
      {
        return true
      }
      guard confirmationAttempt < maximumWorkerPolicyConfirmationAttempts else { return false }
      do {
        try await Task.sleep(for: workerPolicyConfirmationRetryDelay)
      } catch {
        return false
      }
    }
    return false
  }

  func beginServerRestart() {
    pendingMaximumMlxMemoryCeilingBytes = nil
    presentControlActionFeedback(.inProgress("Restarting server…"))
  }

  func completeServerRestart(restartMessage: String) {
    pendingMaximumMlxMemoryCeilingBytes = nil
    presentControlActionFeedback(.success(restartMessage))
  }

  func failServerRestart(_ restartError: Error) {
    pendingMaximumMlxMemoryCeilingBytes = nil
    presentControlActionFeedback(.failure(controlActionErrorMessage(restartError)))
  }

  func failServerStartup(_ startupError: Error) {
    presentControlActionFeedback(.failure(controlActionErrorMessage(startupError)))
  }

  func completeServerStartup() {
    if case .failure = controlActionFeedback {
      controlActionFeedbackDismissalTask?.cancel()
      controlActionFeedbackDismissalTask = nil
      controlActionFeedback = nil
    }
  }

  @discardableResult
  func refresh() async -> SupervisorStatusDocument {
    latestStatusRefreshSequence += 1
    let statusRefreshSequence = latestStatusRefreshSequence
    let precedingStatusRefreshTask = latestStatusRefreshTask
    let statusRefreshTask = Task { @MainActor [weak self] in
      if let precedingStatusRefreshTask { _ = await precedingStatusRefreshTask.value }
      guard let self else { return SupervisorStatusDocument.unavailable }
      return await self.performStatusRefresh()
    }
    latestStatusRefreshTask = statusRefreshTask
    let refreshedStatusDocument = await statusRefreshTask.value
    if statusRefreshSequence == latestStatusRefreshSequence {
      latestStatusRefreshTask = nil
    }
    return refreshedStatusDocument
  }

  private func performStatusRefresh() async -> SupervisorStatusDocument {
    let refreshResult: (statusDocument: SupervisorStatusDocument, errorMessage: String?)
    do {
      refreshResult = (try await supervisorClient.fetchStatus(), nil)
      hasDiscoveredModels = await supervisorClient.modelsAreAvailable()
    } catch {
      // Polling must remain self-healing, while retaining enough bounded context to distinguish
      // contract drift from an ordinary stopped server during qualification and support.
      refreshResult = (.unavailable, boundedStatusRefreshErrorMessage(error))
      hasDiscoveredModels = false
    }
    lastStatusRefreshErrorMessage = refreshResult.errorMessage
    apply(refreshResult.statusDocument)
    if systemTelemetrySamplingIsRequired(
      popoverIsVisible: popoverIsVisible,
      requestIsActive: refreshResult.statusDocument.isActive
    ) {
      updateSystemTelemetryIfDue(forceUpdate: false)
    }
    return refreshResult.statusDocument
  }

  private func updateSystemTelemetryIfDue(forceUpdate: Bool) {
    let currentSampleTime = systemTelemetryClock.now
    let elapsedSinceLastSample = lastSystemTelemetrySampleTime.map {
      $0.duration(to: currentSampleTime)
    }
    guard forceUpdate || systemTelemetrySampleIsDue(elapsedSinceLastSample: elapsedSinceLastSample)
    else { return }
    systemTelemetrySnapshot = systemTelemetrySampler.sample()
    lastSystemTelemetrySampleTime = currentSampleTime
  }

  private func apply(_ nextStatusDocument: SupervisorStatusDocument) {
    statusDocument = nextStatusDocument
    if !popoverIsVisible {
      editableMaximumMlxMemoryGigabytes = nextStatusDocument.configuredMaximumMlxMemoryGigabytes
        ?? maximumWholeDecimalGigabytes
    }
    onMenuBarTitleChanged?(nextStatusDocument.menuBarTitle)
    resolveMaximumMlxMemoryFeedbackIfReady()
  }

  @discardableResult
  private func presentControlActionFeedback(_ feedback: ControlActionFeedback) -> Int {
    controlActionFeedbackDismissalTask?.cancel()
    controlActionFeedbackDismissalTask = nil
    controlActionFeedbackGeneration += 1
    controlActionFeedback = feedback
    return controlActionFeedbackGeneration
  }

  private func resolveMaximumMlxMemoryFeedbackIfReady() {
    guard let pendingMaximumMlxMemoryCeilingBytes,
      statusDocument.status != "unavailable",
      statusDocument.pendingMlxMemoryCeilingBytes == nil,
      case .success = controlActionFeedback
    else { return }

    if statusDocument.mlxMemoryCeilingBytes == pendingMaximumMlxMemoryCeilingBytes {
      self.pendingMaximumMlxMemoryCeilingBytes = nil
      scheduleMaximumMlxMemoryFeedbackDismissal()
    } else if let mlxMemoryLimitError = statusDocument.mlxMemoryLimitError {
      self.pendingMaximumMlxMemoryCeilingBytes = nil
      presentControlActionFeedback(.failure(mlxMemoryLimitError))
    }
  }

  private func scheduleMaximumMlxMemoryFeedbackDismissal() {
    let feedbackGeneration = controlActionFeedbackGeneration
    controlActionFeedbackDismissalTask?.cancel()
    controlActionFeedbackDismissalTask = Task { [weak self] in
      do {
        try await Task.sleep(for: .seconds(1))
      } catch {
        return
      }
      guard let self, self.controlActionFeedbackGeneration == feedbackGeneration else { return }
      self.controlActionFeedback = nil
      self.controlActionFeedbackDismissalTask = nil
    }
  }

  private func requestedMaximumMlxMemoryCeilingBytes(
    _ maximumMlxMemoryGigabytes: UInt64?
  ) -> UInt64? {
    guard let maximumMlxMemoryGigabytes else {
      return statusDocument.machineMlxMemoryCeilingBytes
    }
    let (requestedMlxMemoryCeilingBytes, overflowOccurred) =
      maximumMlxMemoryGigabytes.multipliedReportingOverflow(by: 1_000_000_000)
    return overflowOccurred ? nil : requestedMlxMemoryCeilingBytes
  }

  var minimumWholeDecimalGigabytes: UInt64 {
    (statusDocument.minimumMlxMemoryCeilingBytes + 999_999_999) / 1_000_000_000
  }

  var maximumWholeDecimalGigabytes: UInt64 {
    maximumWholeDecimalGigabytes(for: statusDocument)
  }

  private func maximumWholeDecimalGigabytes(
    for statusDocument: SupervisorStatusDocument
  ) -> UInt64 {
    statusDocument.machineMlxMemoryCeilingBytes / 1_000_000_000
  }

  private func controlActionErrorMessage(_ controlActionError: Error) -> String {
    if let localizedControlActionError = controlActionError as? LocalizedError,
      let localizedMessage = localizedControlActionError.errorDescription
    {
      return localizedMessage
    }
    return controlActionError.localizedDescription
  }

  private func boundedStatusRefreshErrorMessage(_ statusRefreshError: Error) -> String {
    let detailedErrorMessage: String
    switch statusRefreshError {
    case let DecodingError.keyNotFound(missingKey, context):
      detailedErrorMessage = decodingErrorMessage(
        kind: "Missing field", codingPath: context.codingPath + [missingKey])
    case let DecodingError.typeMismatch(_, context):
      detailedErrorMessage = decodingErrorMessage(
        kind: "Unexpected field type", codingPath: context.codingPath)
    case let DecodingError.valueNotFound(_, context):
      detailedErrorMessage = decodingErrorMessage(
        kind: "Missing field value", codingPath: context.codingPath)
    case let DecodingError.dataCorrupted(context):
      detailedErrorMessage = decodingErrorMessage(
        kind: context.debugDescription, codingPath: context.codingPath)
    default:
      detailedErrorMessage = controlActionErrorMessage(statusRefreshError)
    }
    return String(detailedErrorMessage.prefix(maximumStatusRefreshErrorCharacterCount))
  }

  private func decodingErrorMessage(kind: String, codingPath: [CodingKey]) -> String {
    let fieldPath = codingPath.map(\.stringValue).joined(separator: ".")
    return fieldPath.isEmpty ? kind : "\(kind) at \(fieldPath)"
  }
}
