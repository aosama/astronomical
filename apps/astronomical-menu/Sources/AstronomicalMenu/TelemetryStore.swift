import Foundation

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
  @Published var editableMaximumMlxMemoryGigabytes: UInt64 = 0
  var onMenuBarTitleChanged: ((String) -> Void)?

  private let supervisorClient: any SupervisorClient
  private let systemTelemetrySampler = SystemTelemetrySampler()
  private let systemTelemetryClock = ContinuousClock()
  private var pollingTask: Task<Void, Never>?
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
      await refresh()
      guard controlActionFeedbackGeneration == successFeedbackGeneration else { return }
      editableMaximumMlxMemoryGigabytes = statusDocument.configuredMaximumMlxMemoryGigabytes
        ?? maximumWholeDecimalGigabytes
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
      let configurationReloadMessage = try await supervisorClient.reloadConfiguration()
      guard controlActionFeedbackGeneration == configurationReloadFeedbackGeneration else {
        return
      }
      presentControlActionFeedback(.success(configurationReloadMessage))
    } catch {
      guard controlActionFeedbackGeneration == configurationReloadFeedbackGeneration else {
        return
      }
      presentControlActionFeedback(.failure(controlActionErrorMessage(error)))
    }
    await refresh()
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

  private func refresh() async {
    let nextStatusDocument = (try? await supervisorClient.fetchStatus()) ?? .unavailable
    apply(nextStatusDocument)
    if systemTelemetrySamplingIsRequired(
      popoverIsVisible: popoverIsVisible,
      requestIsActive: nextStatusDocument.isActive
    ) {
      updateSystemTelemetryIfDue(forceUpdate: false)
    }
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
}
