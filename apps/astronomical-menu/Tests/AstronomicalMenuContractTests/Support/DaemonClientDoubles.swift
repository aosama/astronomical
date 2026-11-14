import Foundation

@testable import AstronomicalMenuCore

/// A daemon that gives the same answers however often it is asked.
///
/// One configurable stand-in replaces the per-test structs that differed only
/// in a status document or a message. A test states the answers its journey
/// depends on and leaves the rest at defaults that describe a reachable,
/// idle, not-yet-ready daemon. Stand-ins whose answers change during a test
/// (a suspended first request, a counted readiness) stay beside their tests,
/// because their state is the thing under test.
struct FixedAnswerDaemonClient: DaemonClient {
  var statusDocument: DaemonStatusDocument = .unavailable
  /// When set, asking for the status fails with this error, as when the
  /// engine that answers is not this app's.
  var statusFailure: (any Error & Sendable)?
  var reloadResult = ConfigurationReloadResult(message: "Config reloaded")
  /// When set, reloading fails with this error instead of returning `reloadResult`.
  var reloadFailure: (any Error & Sendable)?
  var maximumMlxMemoryUpdateMessage = "MLX memory setting persisted and applied"
  /// How long the daemon takes to answer a memory update, for journeys where
  /// other feedback arrives first.
  var maximumMlxMemoryUpdateDelay: Duration = .zero
  var isHealthAvailable = true
  /// `nil` derives readiness from health and status, as a real client does.
  var isExpectedInstanceHealthy: Bool?

  func fetchStatus() async throws -> DaemonStatusDocument {
    if let statusFailure { throw statusFailure }
    return statusDocument
  }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    if let reloadFailure { throw reloadFailure }
    return reloadResult
  }

  func requestShutdown() async throws {}

  func updateMaximumMlxMemoryGigabytes(_ maximumMlxMemoryGigabytes: UInt64?) async throws -> String {
    if maximumMlxMemoryUpdateDelay > .zero {
      try await Task.sleep(for: maximumMlxMemoryUpdateDelay)
    }
    return maximumMlxMemoryUpdateMessage
  }

  func healthIsAvailable() async -> Bool { isHealthAvailable }

  func expectedInstanceIsHealthy() async -> Bool {
    if let isExpectedInstanceHealthy { return isExpectedInstanceHealthy }
    guard isHealthAvailable else { return false }
    return statusDocument.status == "ready"
      && statusDocument.workerRuntimeFeatureConfigurationApplied
      && statusDocument.configuration?.isEffective == true
  }
}
