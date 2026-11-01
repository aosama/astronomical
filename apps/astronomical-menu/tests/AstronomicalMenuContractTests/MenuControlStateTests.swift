import Foundation
import XCTest

@testable import AstronomicalMenuCore

final class MenuControlStateTests: XCTestCase {
  @MainActor
  func test_should_surface_a_successful_configuration_reload_to_the_popover() async {
    let telemetryStore = contractTelemetryStore(
      daemonClient: SuccessfulReloadDaemonClient()
    )

    await telemetryStore.performConfigurationReload()

    XCTAssertEqual(
      telemetryStore.controlActionFeedback,
      .success("Config reloaded and applied by the worker")
    )
  }

  @MainActor
  func test_should_not_report_worker_restart_success_before_status_confirms_the_acknowledged_policy() async {
    let telemetryStore = contractTelemetryStore(
      daemonClient: RestartAcknowledgedButStatusUnconfirmedDaemonClient()
    )

    await telemetryStore.performConfigurationReload()

    XCTAssertEqual(
      telemetryStore.controlActionFeedback,
      .failure("Worker restart completed, but its applied configuration was not confirmed")
    )
  }

  @MainActor
  func test_should_report_worker_restart_success_after_status_confirms_the_acknowledged_policy() async {
    let telemetryStore = contractTelemetryStore(
      daemonClient: RestartAcknowledgedAndStatusConfirmedDaemonClient()
    )

    await telemetryStore.performConfigurationReload()

    XCTAssertEqual(
      telemetryStore.controlActionFeedback,
      .success("Config reloaded and applied by the worker")
    )
  }

  @MainActor
  func test_should_confirm_a_worker_restart_when_polling_overlaps_delayed_confirmation() async {
    let daemonClient = OverlappingRestartConfirmationDaemonClient()
    let telemetryStore = contractTelemetryStore(daemonClient: daemonClient)
    let configurationReloadTask = Task { @MainActor in
      await telemetryStore.performConfigurationReload()
    }
    await daemonClient.waitUntilFirstStatusRequestIsSuspended()

    let pollingRefreshTask = Task { @MainActor in _ = await telemetryStore.refresh() }
    await Task.yield()
    await daemonClient.resumeFirstStatusRequest()
    await configurationReloadTask.value
    await pollingRefreshTask.value

    XCTAssertEqual(
      telemetryStore.controlActionFeedback,
      .success("Config reloaded and applied by the worker")
    )
  }

  @MainActor
  func test_should_dismiss_a_successful_memory_update_one_second_after_application() async throws {
    let feedbackDismissalGate = FeedbackDismissalGate()
    let telemetryStore = contractTelemetryStore(
      daemonClient: SuccessfulMaximumMlxMemoryUpdateDaemonClient(),
      feedbackDismissalGate: feedbackDismissalGate
    )

    await telemetryStore.performMaximumMlxMemoryLimitUpdate(32)

    XCTAssertEqual(
      telemetryStore.controlActionFeedback,
      .success("MLX memory setting persisted and applied")
    )

    try await Task.sleep(for: .milliseconds(10))
    XCTAssertEqual(
      telemetryStore.controlActionFeedback,
      .success("MLX memory setting persisted and applied")
    )

    feedbackDismissalGate.release()
    try await waitForControlActionFeedbackToDismiss(from: telemetryStore)
  }

  @MainActor
  func test_should_wait_for_a_queued_memory_update_to_take_effect_before_dismissing() async throws {
    let daemonClient = QueuedMaximumMlxMemoryUpdateDaemonClient()
    let feedbackDismissalGate = FeedbackDismissalGate()
    let telemetryStore = contractTelemetryStore(
      daemonClient: daemonClient,
      feedbackDismissalGate: feedbackDismissalGate
    )

    await telemetryStore.performMaximumMlxMemoryLimitUpdate(32)
    XCTAssertEqual(
      telemetryStore.controlActionFeedback,
      .success("MLX memory setting persisted and queued until generation finalizes")
    )

    await daemonClient.markMaximumMlxMemoryUpdateAsApplied()
    telemetryStore.refreshNow()
    try await Task.sleep(for: .milliseconds(10))
    XCTAssertNotNil(telemetryStore.controlActionFeedback)

    feedbackDismissalGate.release()
    try await waitForControlActionFeedbackToDismiss(from: telemetryStore)
  }

  @MainActor
  func test_should_surface_a_late_memory_update_rejection_as_failure_feedback() async {
    let telemetryStore = contractTelemetryStore(
      daemonClient: RejectedMaximumMlxMemoryUpdateDaemonClient()
    )

    await telemetryStore.performMaximumMlxMemoryLimitUpdate(32)

    XCTAssertEqual(
      telemetryStore.controlActionFeedback,
      .failure("The worker could not apply the requested MLX memory ceiling")
    )
  }

  @MainActor
  func test_should_start_dismissal_after_a_status_refresh_precedes_the_update_response() async throws {
    let feedbackDismissalGate = FeedbackDismissalGate()
    let telemetryStore = contractTelemetryStore(
      daemonClient: DelayedMaximumMlxMemoryUpdateDaemonClient(),
      feedbackDismissalGate: feedbackDismissalGate
    )
    let memoryUpdateTask = Task { @MainActor in
      await telemetryStore.performMaximumMlxMemoryLimitUpdate(32)
    }

    try? await Task.sleep(for: .milliseconds(20))
    telemetryStore.refreshNow()
    try? await Task.sleep(for: .milliseconds(50))
    await memoryUpdateTask.value

    XCTAssertEqual(
      telemetryStore.controlActionFeedback,
      .success("MLX memory setting persisted and applied")
    )
    feedbackDismissalGate.release()
    try await waitForControlActionFeedbackToDismiss(from: telemetryStore)
  }

  @MainActor
  func test_should_dismiss_the_applied_sentence_when_the_engine_clamps_the_request() async throws {
    let feedbackDismissalGate = FeedbackDismissalGate()
    let telemetryStore = contractTelemetryStore(
      daemonClient: ClampedMaximumMlxMemoryUpdateDaemonClient(),
      feedbackDismissalGate: feedbackDismissalGate
    )

    await telemetryStore.performMaximumMlxMemoryLimitUpdate(37)

    XCTAssertEqual(
      telemetryStore.controlActionFeedback,
      .success("MLX memory setting persisted and applied")
    )
    feedbackDismissalGate.release()
    try await waitForControlActionFeedbackToDismiss(from: telemetryStore)
  }

  @MainActor
  func test_should_keep_the_applied_sentence_while_the_live_ceiling_is_still_the_boot_ceiling() async throws {
    let telemetryStore = contractTelemetryStore(
      daemonClient: UnappliedMaximumMlxMemoryUpdateDaemonClient()
    )

    await telemetryStore.performMaximumMlxMemoryLimitUpdate(37)
    try await Task.sleep(for: .milliseconds(200))

    XCTAssertEqual(
      telemetryStore.controlActionFeedback,
      .success("MLX memory setting persisted and applied")
    )
  }

  @MainActor
  func test_should_not_let_memory_success_dismiss_newer_server_feedback() async throws {
    let telemetryStore = contractTelemetryStore(
      daemonClient: SuccessfulMaximumMlxMemoryUpdateDaemonClient()
    )

    await telemetryStore.performMaximumMlxMemoryLimitUpdate(32)
    telemetryStore.beginServerRestart()

    try await Task.sleep(for: .milliseconds(50))

    XCTAssertEqual(telemetryStore.controlActionFeedback, .inProgress("Restarting server…"))
  }

  @MainActor
  func test_should_ignore_a_memory_update_that_finishes_after_newer_feedback_begins() async {
    let telemetryStore = contractTelemetryStore(
      daemonClient: DelayedMaximumMlxMemoryUpdateDaemonClient()
    )
    let memoryUpdateTask = Task { @MainActor in
      await telemetryStore.performMaximumMlxMemoryLimitUpdate(32)
    }

    try? await Task.sleep(for: .milliseconds(20))
    telemetryStore.beginServerRestart()
    await memoryUpdateTask.value

    XCTAssertEqual(telemetryStore.controlActionFeedback, .inProgress("Restarting server…"))
  }

  @MainActor
  func test_should_surface_a_configuration_reload_failure_to_the_popover() async {
    let telemetryStore = contractTelemetryStore(
      daemonClient: FailingReloadDaemonClient()
    )

    await telemetryStore.performConfigurationReload()

    XCTAssertEqual(
      telemetryStore.controlActionFeedback,
      .failure("Configuration validation failed")
    )
  }

  @MainActor
  private func waitForControlActionFeedbackToDismiss(from telemetryStore: TelemetryStore) async throws {
    let feedbackDismissalClock = ContinuousClock()
    let feedbackDismissalDeadline = feedbackDismissalClock.now.advanced(by: .milliseconds(400))
    while telemetryStore.controlActionFeedback != nil,
      feedbackDismissalClock.now < feedbackDismissalDeadline
    {
      try await Task.sleep(for: .milliseconds(5))
    }
    XCTAssertNil(telemetryStore.controlActionFeedback)
  }
}

/// Holds the feedback-dismissal task until the test releases it, so assertions
/// on transient feedback cannot race the real wall-clock window (issue #524).
final class FeedbackDismissalGate: @unchecked Sendable {
  private let lock = NSLock()
  private var continuations: [CheckedContinuation<Void, Never>] = []
  private var isReleased = false

  func suspendDismissal() async {
    await withCheckedContinuation { continuation in
      self.lock.lock()
      defer { self.lock.unlock() }
      if self.isReleased {
        continuation.resume()
      } else {
        self.continuations.append(continuation)
      }
    }
  }

  func release() {
    lock.lock()
    defer { lock.unlock() }
    isReleased = true
    continuations.forEach { $0.resume() }
    continuations.removeAll()
  }
}

@MainActor
private func contractTelemetryStore(
  daemonClient: any DaemonClient,
  feedbackDismissalGate: FeedbackDismissalGate? = nil
) -> TelemetryStore {
  let controlActionFeedbackDismissalSleep: TelemetryStore.ControlActionFeedbackDismissalSleep?
  if let feedbackDismissalGate {
    controlActionFeedbackDismissalSleep = { _ in await feedbackDismissalGate.suspendDismissal() }
  } else {
    controlActionFeedbackDismissalSleep = nil
  }
  return TelemetryStore(
    daemonClient: daemonClient,
    controlActionFeedbackDismissalDelay: .milliseconds(80),
    controlActionFeedbackDismissalSleep: controlActionFeedbackDismissalSleep,
    workerPolicyConfirmationRetryDelay: .milliseconds(1)
  )
}

private struct SuccessfulReloadDaemonClient: DaemonClient {
  func fetchStatus() async throws -> DaemonStatusDocument { .unavailable }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    ConfigurationReloadResult(message: "Config reloaded and applied by the worker")
  }

  func requestShutdown() async throws {}

  func healthIsAvailable() async -> Bool { true }
}

private struct RestartAcknowledgedButStatusUnconfirmedDaemonClient: DaemonClient {
  private let acknowledgedFeatureConfiguration = WorkerRuntimeFeatureConfiguration(
    configurationGeneration: String(repeating: "a", count: 64),
    persistentPromptCacheEnabled: true,
    promptCacheMaximumSizeBytes: 50_000_000_000,
    loadedModel: nil
  )

  func fetchStatus() async throws -> DaemonStatusDocument {
    // The prior worker's matching policy is not proof that the replacement has applied it. The
    // acknowledgment flag is the status-side owner for that lifecycle boundary.
    DaemonStatusDocument(
      status: "ready",
      activity: "idle",
      readyModelIdentifier: nil,
      progress: nil,
      expertMemoryMode: nil,
      workerRuntimeFeatureConfigurationApplied: false,
      workerRuntimeFeatureConfiguration: acknowledgedFeatureConfiguration,
      mlxMemoryCeilingBytes: 34_359_738_368,
      servingSession: .empty
    )
  }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    ConfigurationReloadResult(
      message: "Config reloaded and applied by the worker",
      workerRestartCompleted: true,
      workerRuntimeFeatureConfiguration: acknowledgedFeatureConfiguration
    )
  }

  func requestShutdown() async throws {}

  func healthIsAvailable() async -> Bool { true }
}

private struct RestartAcknowledgedAndStatusConfirmedDaemonClient: DaemonClient {
  private let appliedFeatureConfiguration = WorkerRuntimeFeatureConfiguration(
    configurationGeneration: String(repeating: "a", count: 64),
    persistentPromptCacheEnabled: true,
    promptCacheMaximumSizeBytes: 50_000_000_000,
    loadedModel: nil
  )

  func fetchStatus() async throws -> DaemonStatusDocument {
    DaemonStatusDocument(
      status: "ready",
      activity: "idle",
      readyModelIdentifier: nil,
      progress: nil,
      expertMemoryMode: nil,
      workerRuntimeFeatureConfigurationApplied: true,
      workerRuntimeFeatureConfiguration: appliedFeatureConfiguration,
      mlxMemoryCeilingBytes: 34_359_738_368,
      servingSession: .empty
    )
  }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    ConfigurationReloadResult(
      message: "Config reloaded and applied by the worker",
      workerRestartCompleted: true,
      workerRuntimeFeatureConfiguration: appliedFeatureConfiguration
    )
  }

  func requestShutdown() async throws {}

  func healthIsAvailable() async -> Bool { true }
}

private actor OverlappingRestartConfirmationDaemonClient: DaemonClient {
  private let appliedFeatureConfiguration = WorkerRuntimeFeatureConfiguration(
    configurationGeneration: String(repeating: "a", count: 64),
    persistentPromptCacheEnabled: true,
    promptCacheMaximumSizeBytes: 50_000_000_000,
    loadedModel: nil
  )
  private var statusRequestCount = 0
  private var firstStatusRequestContinuation: CheckedContinuation<Void, Never>?

  func fetchStatus() async throws -> DaemonStatusDocument {
    statusRequestCount += 1
    let currentStatusRequestCount = statusRequestCount
    if currentStatusRequestCount == 1 {
      await withCheckedContinuation { continuation in
        firstStatusRequestContinuation = continuation
      }
    }
    return DaemonStatusDocument(
      status: "ready",
      activity: "idle",
      readyModelIdentifier: nil,
      progress: nil,
      expertMemoryMode: nil,
      workerRuntimeFeatureConfigurationApplied: currentStatusRequestCount > 1,
      workerRuntimeFeatureConfiguration: appliedFeatureConfiguration,
      mlxMemoryCeilingBytes: 34_359_738_368,
      servingSession: .empty
    )
  }

  func waitUntilFirstStatusRequestIsSuspended() async {
    while firstStatusRequestContinuation == nil { await Task.yield() }
  }

  func resumeFirstStatusRequest() {
    firstStatusRequestContinuation?.resume()
    firstStatusRequestContinuation = nil
  }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    ConfigurationReloadResult(
      message: "Config reloaded and applied by the worker",
      workerRestartCompleted: true,
      workerRuntimeFeatureConfiguration: appliedFeatureConfiguration
    )
  }

  func requestShutdown() async throws {}

  func healthIsAvailable() async -> Bool { true }
}

private struct SuccessfulMaximumMlxMemoryUpdateDaemonClient: DaemonClient {
  func fetchStatus() async throws -> DaemonStatusDocument {
    maximumMlxMemoryStatusDocument(effectiveMlxMemoryCeilingBytes: 34_359_738_368)
  }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    ConfigurationReloadResult(message: "Config reloaded")
  }

  func updateMaximumMlxMemoryGigabytes(_ maximumMlxMemoryGigabytes: UInt64?) async throws -> String {
    "MLX memory setting persisted and applied"
  }

  func requestShutdown() async throws {}

  func healthIsAvailable() async -> Bool { true }
}

private actor QueuedMaximumMlxMemoryUpdateDaemonClient: DaemonClient {
  private var maximumMlxMemoryUpdateHasTakenEffect = false

  func fetchStatus() async throws -> DaemonStatusDocument {
    if maximumMlxMemoryUpdateHasTakenEffect {
      return maximumMlxMemoryStatusDocument(effectiveMlxMemoryCeilingBytes: 34_359_738_368)
    }
    return maximumMlxMemoryStatusDocument(
      effectiveMlxMemoryCeilingBytes: 30_000_000_000,
      pendingMlxMemoryCeilingBytes: 34_359_738_368,
      activity: "generating"
    )
  }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    ConfigurationReloadResult(message: "Config reloaded")
  }

  func updateMaximumMlxMemoryGigabytes(_ maximumMlxMemoryGigabytes: UInt64?) async throws -> String {
    "MLX memory setting persisted and queued until generation finalizes"
  }

  func requestShutdown() async throws {}

  func healthIsAvailable() async -> Bool { true }

  func markMaximumMlxMemoryUpdateAsApplied() {
    maximumMlxMemoryUpdateHasTakenEffect = true
  }
}

private struct RejectedMaximumMlxMemoryUpdateDaemonClient: DaemonClient {
  func fetchStatus() async throws -> DaemonStatusDocument {
    maximumMlxMemoryStatusDocument(
      effectiveMlxMemoryCeilingBytes: 30_000_000_000,
      mlxMemoryLimitError: "The worker could not apply the requested MLX memory ceiling"
    )
  }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    ConfigurationReloadResult(message: "Config reloaded")
  }

  func updateMaximumMlxMemoryGigabytes(_ maximumMlxMemoryGigabytes: UInt64?) async throws -> String {
    "MLX memory setting persisted and queued until generation finalizes"
  }

  func requestShutdown() async throws {}

  func healthIsAvailable() async -> Bool { true }
}

private struct DelayedMaximumMlxMemoryUpdateDaemonClient: DaemonClient {
  func fetchStatus() async throws -> DaemonStatusDocument {
    maximumMlxMemoryStatusDocument(effectiveMlxMemoryCeilingBytes: 34_359_738_368)
  }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    ConfigurationReloadResult(message: "Config reloaded")
  }

  func updateMaximumMlxMemoryGigabytes(_ maximumMlxMemoryGigabytes: UInt64?) async throws -> String {
    try await Task.sleep(for: .milliseconds(200))
    return "MLX memory setting persisted and applied"
  }

  func requestShutdown() async throws {}

  func healthIsAvailable() async -> Bool { true }
}

private struct ClampedMaximumMlxMemoryUpdateDaemonClient: DaemonClient {
  func fetchStatus() async throws -> DaemonStatusDocument {
    maximumMlxMemoryStatusDocument(
      effectiveMlxMemoryCeilingBytes: 39_195_874_112,
      wiredResidencyBudgetBytes: 40_200_896_512,
      machineMlxMemoryCeilingBytes: 42_949_672_960,
      configuredMaximumMlxMemoryGigabytes: 37
    )
  }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    ConfigurationReloadResult(message: "Config reloaded")
  }

  func updateMaximumMlxMemoryGigabytes(_ maximumMlxMemoryGigabytes: UInt64?) async throws -> String {
    "MLX memory setting persisted and applied"
  }

  func requestShutdown() async throws {}

  func healthIsAvailable() async -> Bool { true }
}

private struct UnappliedMaximumMlxMemoryUpdateDaemonClient: DaemonClient {
  func fetchStatus() async throws -> DaemonStatusDocument {
    maximumMlxMemoryStatusDocument(
      effectiveMlxMemoryCeilingBytes: 22_548_578_304,
      wiredResidencyBudgetBytes: 40_200_896_512,
      machineMlxMemoryCeilingBytes: 42_949_672_960,
      configuredMaximumMlxMemoryGigabytes: 37
    )
  }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    ConfigurationReloadResult(message: "Config reloaded")
  }

  func updateMaximumMlxMemoryGigabytes(_ maximumMlxMemoryGigabytes: UInt64?) async throws -> String {
    "MLX memory setting persisted and applied"
  }

  func requestShutdown() async throws {}

  func healthIsAvailable() async -> Bool { true }
}

private func maximumMlxMemoryStatusDocument(
  effectiveMlxMemoryCeilingBytes: UInt64,
  pendingMlxMemoryCeilingBytes: UInt64? = nil,
  activity: String = "idle",
  mlxMemoryLimitError: String? = nil,
  wiredResidencyBudgetBytes: UInt64? = nil,
  machineMlxMemoryCeilingBytes: UInt64 = 40_000_000_000,
  configuredMaximumMlxMemoryGigabytes: UInt64? = 32
) -> DaemonStatusDocument {
  DaemonStatusDocument(
    status: "ready",
    activity: activity,
    readyModelIdentifier: nil,
    progress: nil,
    expertMemoryMode: nil,
    mlxMemoryCeilingBytes: effectiveMlxMemoryCeilingBytes,
    machineMlxMemoryCeilingBytes: machineMlxMemoryCeilingBytes,
    wiredResidencyBudgetBytes: wiredResidencyBudgetBytes,
    minimumMlxMemoryCeilingBytes: 1_000_000_000,
    configuredMaximumMlxMemoryGigabytes: configuredMaximumMlxMemoryGigabytes,
    pendingMlxMemoryCeilingBytes: pendingMlxMemoryCeilingBytes,
    mlxMemoryLimitError: mlxMemoryLimitError,
    servingSession: .empty
  )
}

private struct FailingReloadDaemonClient: DaemonClient {
  func fetchStatus() async throws -> DaemonStatusDocument { .unavailable }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    throw ConfigurationReloadFailure.validationFailed
  }

  func requestShutdown() async throws {}

  func healthIsAvailable() async -> Bool { true }
}

private enum ConfigurationReloadFailure: LocalizedError {
  case validationFailed

  var errorDescription: String? {
    switch self {
    case .validationFailed: "Configuration validation failed"
    }
  }
}
