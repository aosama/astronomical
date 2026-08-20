import Foundation
import XCTest

@testable import AstronomicalMenu

final class MenuControlStateTests: XCTestCase {
  @MainActor
  func test_should_surface_a_successful_configuration_reload_to_the_popover() async {
    let telemetryStore = TelemetryStore(
      supervisorClient: SuccessfulReloadSupervisorClient()
    )

    await telemetryStore.performConfigurationReload()

    XCTAssertEqual(
      telemetryStore.controlActionFeedback,
      .success("Config reloaded and applied by the worker")
    )
  }

  @MainActor
  func test_should_not_report_worker_restart_success_before_status_confirms_the_acknowledged_policy() async {
    let telemetryStore = TelemetryStore(
      supervisorClient: RestartAcknowledgedButStatusUnconfirmedSupervisorClient()
    )

    await telemetryStore.performConfigurationReload()

    XCTAssertEqual(
      telemetryStore.controlActionFeedback,
      .failure("Worker restart completed, but its applied configuration was not confirmed")
    )
  }

  @MainActor
  func test_should_report_worker_restart_success_after_status_confirms_the_acknowledged_policy() async {
    let telemetryStore = TelemetryStore(
      supervisorClient: RestartAcknowledgedAndStatusConfirmedSupervisorClient()
    )

    await telemetryStore.performConfigurationReload()

    XCTAssertEqual(
      telemetryStore.controlActionFeedback,
      .success("Config reloaded and applied by the worker")
    )
  }

  @MainActor
  func test_should_dismiss_a_successful_memory_update_one_second_after_application() async throws {
    let telemetryStore = TelemetryStore(
      supervisorClient: SuccessfulMaximumMlxMemoryUpdateSupervisorClient()
    )

    await telemetryStore.performMaximumMlxMemoryLimitUpdate(32)

    XCTAssertEqual(
      telemetryStore.controlActionFeedback,
      .success("MLX memory setting persisted and applied")
    )

    try await Task.sleep(for: .milliseconds(100))
    XCTAssertEqual(
      telemetryStore.controlActionFeedback,
      .success("MLX memory setting persisted and applied")
    )

    try await waitForControlActionFeedbackToDismiss(from: telemetryStore)
  }

  @MainActor
  func test_should_wait_for_a_queued_memory_update_to_take_effect_before_dismissing() async throws {
    let supervisorClient = QueuedMaximumMlxMemoryUpdateSupervisorClient()
    let telemetryStore = TelemetryStore(supervisorClient: supervisorClient)

    await telemetryStore.performMaximumMlxMemoryLimitUpdate(32)
    XCTAssertEqual(
      telemetryStore.controlActionFeedback,
      .success("MLX memory setting persisted and queued until generation finalizes")
    )

    await supervisorClient.markMaximumMlxMemoryUpdateAsApplied()
    telemetryStore.refreshNow()
    try await Task.sleep(for: .milliseconds(100))
    XCTAssertNotNil(telemetryStore.controlActionFeedback)

    try await waitForControlActionFeedbackToDismiss(from: telemetryStore)
  }

  @MainActor
  func test_should_surface_a_late_memory_update_rejection_as_failure_feedback() async {
    let telemetryStore = TelemetryStore(
      supervisorClient: RejectedMaximumMlxMemoryUpdateSupervisorClient()
    )

    await telemetryStore.performMaximumMlxMemoryLimitUpdate(32)

    XCTAssertEqual(
      telemetryStore.controlActionFeedback,
      .failure("The worker could not apply the requested MLX memory ceiling")
    )
  }

  @MainActor
  func test_should_start_dismissal_after_a_status_refresh_precedes_the_update_response() async throws {
    let telemetryStore = TelemetryStore(
      supervisorClient: DelayedMaximumMlxMemoryUpdateSupervisorClient()
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
    try await waitForControlActionFeedbackToDismiss(from: telemetryStore)
  }

  @MainActor
  func test_should_not_let_memory_success_dismiss_newer_server_feedback() async throws {
    let telemetryStore = TelemetryStore(
      supervisorClient: SuccessfulMaximumMlxMemoryUpdateSupervisorClient()
    )

    await telemetryStore.performMaximumMlxMemoryLimitUpdate(32)
    telemetryStore.beginServerRestart()

    try await Task.sleep(for: .milliseconds(1_100))

    XCTAssertEqual(telemetryStore.controlActionFeedback, .inProgress("Restarting server…"))
  }

  @MainActor
  func test_should_ignore_a_memory_update_that_finishes_after_newer_feedback_begins() async {
    let telemetryStore = TelemetryStore(
      supervisorClient: DelayedMaximumMlxMemoryUpdateSupervisorClient()
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
    let telemetryStore = TelemetryStore(
      supervisorClient: FailingReloadSupervisorClient()
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
    let feedbackDismissalDeadline = feedbackDismissalClock.now.advanced(by: .seconds(2))
    while telemetryStore.controlActionFeedback != nil,
      feedbackDismissalClock.now < feedbackDismissalDeadline
    {
      try await Task.sleep(for: .milliseconds(25))
    }
    XCTAssertNil(telemetryStore.controlActionFeedback)
  }
}

private struct SuccessfulReloadSupervisorClient: SupervisorClient {
  func fetchStatus() async throws -> SupervisorStatusDocument { .unavailable }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    ConfigurationReloadResult(message: "Config reloaded and applied by the worker")
  }

  func requestShutdown() async throws {}

  func healthIsAvailable() async -> Bool { true }
}

private struct RestartAcknowledgedButStatusUnconfirmedSupervisorClient: SupervisorClient {
  private let acknowledgedFeatureConfiguration = WorkerRuntimeFeatureConfiguration(
    configurationGeneration: String(repeating: "a", count: 64),
    persistentPromptCacheEnabled: true,
    promptCacheMaximumSizeBytes: 50_000_000_000,
    loadedModel: nil
  )

  func fetchStatus() async throws -> SupervisorStatusDocument {
    // The prior worker's matching policy is not proof that the replacement has applied it. The
    // acknowledgment flag is the status-side owner for that lifecycle boundary.
    SupervisorStatusDocument(
      status: "ready",
      activity: "idle",
      readyModelIdentifier: nil,
      progress: nil,
      expertMemoryMode: nil,
      workerRuntimeFeatureConfigurationApplied: false,
      workerRuntimeFeatureConfiguration: acknowledgedFeatureConfiguration,
      mlxMemoryCeilingBytes: 32_000_000_000,
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

private struct RestartAcknowledgedAndStatusConfirmedSupervisorClient: SupervisorClient {
  private let appliedFeatureConfiguration = WorkerRuntimeFeatureConfiguration(
    configurationGeneration: String(repeating: "a", count: 64),
    persistentPromptCacheEnabled: true,
    promptCacheMaximumSizeBytes: 50_000_000_000,
    loadedModel: nil
  )

  func fetchStatus() async throws -> SupervisorStatusDocument {
    SupervisorStatusDocument(
      status: "ready",
      activity: "idle",
      readyModelIdentifier: nil,
      progress: nil,
      expertMemoryMode: nil,
      workerRuntimeFeatureConfigurationApplied: true,
      workerRuntimeFeatureConfiguration: appliedFeatureConfiguration,
      mlxMemoryCeilingBytes: 32_000_000_000,
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

private struct SuccessfulMaximumMlxMemoryUpdateSupervisorClient: SupervisorClient {
  func fetchStatus() async throws -> SupervisorStatusDocument {
    maximumMlxMemoryStatusDocument(effectiveMlxMemoryCeilingBytes: 32_000_000_000)
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

private actor QueuedMaximumMlxMemoryUpdateSupervisorClient: SupervisorClient {
  private var maximumMlxMemoryUpdateHasTakenEffect = false

  func fetchStatus() async throws -> SupervisorStatusDocument {
    if maximumMlxMemoryUpdateHasTakenEffect {
      return maximumMlxMemoryStatusDocument(effectiveMlxMemoryCeilingBytes: 32_000_000_000)
    }
    return maximumMlxMemoryStatusDocument(
      effectiveMlxMemoryCeilingBytes: 30_000_000_000,
      pendingMlxMemoryCeilingBytes: 32_000_000_000,
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

private struct RejectedMaximumMlxMemoryUpdateSupervisorClient: SupervisorClient {
  func fetchStatus() async throws -> SupervisorStatusDocument {
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

private struct DelayedMaximumMlxMemoryUpdateSupervisorClient: SupervisorClient {
  func fetchStatus() async throws -> SupervisorStatusDocument {
    maximumMlxMemoryStatusDocument(effectiveMlxMemoryCeilingBytes: 32_000_000_000)
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

private func maximumMlxMemoryStatusDocument(
  effectiveMlxMemoryCeilingBytes: UInt64,
  pendingMlxMemoryCeilingBytes: UInt64? = nil,
  activity: String = "idle",
  mlxMemoryLimitError: String? = nil
) -> SupervisorStatusDocument {
  SupervisorStatusDocument(
    status: "ready",
    activity: activity,
    readyModelIdentifier: nil,
    progress: nil,
    expertMemoryMode: nil,
    mlxMemoryCeilingBytes: effectiveMlxMemoryCeilingBytes,
    machineMlxMemoryCeilingBytes: 40_000_000_000,
    minimumMlxMemoryCeilingBytes: 1_000_000_000,
    configuredMaximumMlxMemoryGigabytes: 32,
    pendingMlxMemoryCeilingBytes: pendingMlxMemoryCeilingBytes,
    mlxMemoryLimitError: mlxMemoryLimitError,
    servingSession: .empty
  )
}

private struct FailingReloadSupervisorClient: SupervisorClient {
  func fetchStatus() async throws -> SupervisorStatusDocument { .unavailable }

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
