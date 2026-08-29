import Foundation
import XCTest

@testable import AstronomicalMenu

final class DaemonControlTests: XCTestCase {
  @MainActor
  func test_should_surface_a_daemon_spawn_error() async throws {
    let testContext = try DaemonLifecycleTestContext(daemonExecutablePath: "/does-not-exist/astronomicald")
    defer { testContext.removeTemporaryDirectory() }
    let daemonLifecycleController = testContext.makeController(
      supervisorClient: DelayedReadinessSupervisorClient(readyAfterCheckCount: 1))

    do {
      try await daemonLifecycleController.startDaemonIfNeeded()
      XCTFail("A daemon spawn error must reach application startup")
    } catch let lifecycleError as DaemonLifecycleError {
      guard case .cannotLaunchDaemon = lifecycleError else {
        return XCTFail("Expected a daemon launch error, received \(lifecycleError)")
      }
    }
  }

  @MainActor
  func test_should_reject_an_occupied_endpoint_until_its_configuration_is_effective() async throws {
    let testContext = try DaemonLifecycleTestContext(daemonExecutablePath: "/bin/sleep")
    defer { testContext.removeTemporaryDirectory() }
    let daemonLifecycleController = testContext.makeController(
      supervisorClient: StuckExternalSupervisorClient(), daemonArguments: ["30"],
      readinessTimeout: .milliseconds(30))

    do {
      try await daemonLifecycleController.startDaemonIfNeeded()
      XCTFail("An occupied but unready endpoint must not be accepted as startup success")
    } catch let lifecycleError as DaemonLifecycleError {
      guard case let .existingDaemonNotReady(configurationFileLabel) = lifecycleError else {
        return XCTFail("Expected an occupied-endpoint readiness error, received \(lifecycleError)")
      }
      XCTAssertEqual(configurationFileLabel, "~/.astronomical/config.json")
    }
    XCTAssertFalse(daemonLifecycleController.ownsDaemon)
  }

  @MainActor
  func test_should_allow_an_existing_daemon_to_finish_starting() async throws {
    let testContext = try DaemonLifecycleTestContext(
      daemonExecutablePath: "/does-not-exist/astronomicald")
    defer { testContext.removeTemporaryDirectory() }
    let daemonLifecycleController = testContext.makeController(
      supervisorClient: ExistingDelayedReadinessSupervisorClient(readyAfterCheckCount: 3),
      readinessTimeout: .seconds(1))

    try await daemonLifecycleController.startDaemonIfNeeded()

    XCTAssertFalse(daemonLifecycleController.ownsDaemon)
  }

  @MainActor
  func test_should_surface_an_early_daemon_exit_and_remove_ownership() async throws {
    let testContext = try DaemonLifecycleTestContext(daemonExecutablePath: "/usr/bin/false")
    defer { testContext.removeTemporaryDirectory() }
    let daemonLifecycleController = testContext.makeController(
      supervisorClient: DelayedReadinessSupervisorClient(readyAfterCheckCount: .max))

    do {
      try await daemonLifecycleController.startDaemonIfNeeded()
      XCTFail("An exited daemon must not be reported as ready")
    } catch let lifecycleError as DaemonLifecycleError {
      guard case let .daemonExitedBeforeReady(configurationFileLabel) = lifecycleError else {
        return XCTFail("Expected an early-exit error, received \(lifecycleError)")
      }
      XCTAssertEqual(configurationFileLabel, "~/.astronomical/config.json")
    }

    XCTAssertFalse(daemonLifecycleController.ownsDaemon)
    XCTAssertFalse(FileManager.default.fileExists(atPath: testContext.ownershipRecordURL.path))
  }

  @MainActor
  func test_should_retry_daemon_startup_after_an_early_exit_until_the_server_is_ready() async throws {
    let scriptDirectoryURL = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: scriptDirectoryURL, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: scriptDirectoryURL) }
    let failOnceScriptURL = scriptDirectoryURL.appendingPathComponent("fail-once-daemon.sh")
    let failOnceMarkerURL = scriptDirectoryURL.appendingPathComponent("started-once")
    try """
    #!/bin/sh
    if [ ! -f "$1" ]; then
      touch "$1"
      exit 1
    fi
    exec /bin/sleep 30
    """.write(to: failOnceScriptURL, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
      [.posixPermissions: 0o755], ofItemAtPath: failOnceScriptURL.path)

    let testContext = try DaemonLifecycleTestContext(daemonExecutablePath: failOnceScriptURL.path)
    defer { testContext.removeTemporaryDirectory() }
    let supervisorClient = DelayedReadinessSupervisorClient(readyAfterCheckCount: 1)
    let daemonLifecycleController = testContext.makeController(
      supervisorClient: supervisorClient,
      daemonArguments: [failOnceMarkerURL.path],
      readinessTimeout: .milliseconds(250))
    defer { daemonLifecycleController.stopOwnedDaemon() }
    let telemetryStore = TelemetryStore(supervisorClient: supervisorClient)
    let maintenanceCompleted = expectation(description: "daemon became ready after retry")

    let maintenanceTask = Task { @MainActor in
      await maintainDaemonForApplication(
        daemonLifecycleController: daemonLifecycleController,
        telemetryStore: telemetryStore,
        retryDelay: .milliseconds(20))
      maintenanceCompleted.fulfill()
    }
    await fulfillment(of: [maintenanceCompleted], timeout: 2)
    maintenanceTask.cancel()

    XCTAssertTrue(daemonLifecycleController.ownsDaemon)
    XCTAssertNil(telemetryStore.controlActionFeedback)
  }

  @MainActor
  func test_should_present_a_startup_configuration_failure_in_control_feedback() async throws {
    let testContext = try DaemonLifecycleTestContext(daemonExecutablePath: "/usr/bin/false")
    defer { testContext.removeTemporaryDirectory() }
    let supervisorClient = DelayedReadinessSupervisorClient(readyAfterCheckCount: .max)
    let daemonLifecycleController = testContext.makeController(supervisorClient: supervisorClient)
    let telemetryStore = TelemetryStore(supervisorClient: supervisorClient)

    await startDaemonForApplication(
      daemonLifecycleController: daemonLifecycleController,
      telemetryStore: telemetryStore)

    guard case let .failure(feedbackMessage) = telemetryStore.controlActionFeedback else {
      return XCTFail("Application startup must retain daemon failure feedback for the popover")
    }
    XCTAssertTrue(feedbackMessage.contains("~/.astronomical/config.json"))
    XCTAssertTrue(feedbackMessage.localizedCaseInsensitiveContains("retry"))
  }

  @MainActor
  func test_should_not_complete_restart_before_expected_instance_health() async throws {
    let testContext = try DaemonLifecycleTestContext(daemonExecutablePath: "/bin/sleep")
    defer { testContext.removeTemporaryDirectory() }
    let supervisorClient = ControlledReadinessSupervisorClient()
    let daemonLifecycleController = testContext.makeController(
      supervisorClient: supervisorClient,
      daemonArguments: ["30"])
    defer { daemonLifecycleController.stopOwnedDaemon() }
    let restartCompletion = RestartCompletionState()

    let restartTask = Task { @MainActor in
      let restartMessage = try await daemonLifecycleController.restartDaemon()
      await restartCompletion.record(message: restartMessage)
    }
    try await supervisorClient.waitUntilReadinessWasChecked()

    let messageBeforeHealth = await restartCompletion.message
    XCTAssertNil(messageBeforeHealth)
    await supervisorClient.confirmExpectedInstanceHealth()
    try await restartTask.value
    let messageAfterHealth = await restartCompletion.message
    XCTAssertNotNil(messageAfterHealth)
  }

  @MainActor
  func test_should_complete_startup_and_restart_after_delayed_health() async throws {
    let testContext = try DaemonLifecycleTestContext(daemonExecutablePath: "/bin/sleep")
    defer { testContext.removeTemporaryDirectory() }
    let startupClient = DelayedReadinessSupervisorClient(readyAfterCheckCount: 3)
    let startupController = testContext.makeController(
      supervisorClient: startupClient,
      daemonArguments: ["30"])
    defer { startupController.stopOwnedDaemon() }

    try await startupController.startDaemonIfNeeded()
    XCTAssertTrue(startupController.ownsDaemon)
    startupController.stopOwnedDaemon()

    let restartClient = DelayedReadinessSupervisorClient(readyAfterCheckCount: 3)
    let restartController = testContext.makeController(
      supervisorClient: restartClient,
      daemonArguments: ["30"])
    defer { restartController.stopOwnedDaemon() }
    let restartMessage = try await restartController.restartDaemon()
    XCTAssertFalse(restartMessage.isEmpty)
    XCTAssertTrue(restartController.ownsDaemon)
  }

  @MainActor
  func test_should_bound_readiness_wait_and_remove_the_unready_daemon() async throws {
    let testContext = try DaemonLifecycleTestContext(daemonExecutablePath: "/bin/sleep")
    defer { testContext.removeTemporaryDirectory() }
    let daemonLifecycleController = testContext.makeController(
      supervisorClient: DelayedReadinessSupervisorClient(readyAfterCheckCount: .max),
      daemonArguments: ["30"],
      readinessTimeout: .milliseconds(30))
    defer { daemonLifecycleController.stopOwnedDaemon() }
    let readinessClock = ContinuousClock()
    let readinessStart = readinessClock.now

    do {
      try await daemonLifecycleController.startDaemonIfNeeded()
      XCTFail("An unready daemon must time out")
    } catch let lifecycleError as DaemonLifecycleError {
      guard case .readinessTimedOut = lifecycleError else {
        return XCTFail("Expected a readiness timeout, received \(lifecycleError)")
      }
    }

    XCTAssertLessThan(readinessStart.duration(to: readinessClock.now), .seconds(1))
    XCTAssertFalse(daemonLifecycleController.ownsDaemon)
  }

  @MainActor
  func test_should_keep_stable_and_development_configuration_labels_isolated() async throws {
    for (applicationIdentity, expectedConfigurationLabel, otherConfigurationLabel) in [
      (
        testApplicationIdentity(channel: .stable), "~/.astronomical/config.json",
        "~/.astronomical-dev/config.json"
      ),
      (
        testApplicationIdentity(channel: .development), "~/.astronomical-dev/config.json",
        "~/.astronomical/config.json"
      ),
    ] {
      let testContext = try DaemonLifecycleTestContext(
        daemonExecutablePath: "/usr/bin/false", applicationIdentity: applicationIdentity)
      defer { testContext.removeTemporaryDirectory() }
      let daemonLifecycleController = testContext.makeController(
        supervisorClient: DelayedReadinessSupervisorClient(readyAfterCheckCount: .max))

      do {
        try await daemonLifecycleController.startDaemonIfNeeded()
        XCTFail("An exited daemon must identify its own configuration")
      } catch let lifecycleError as DaemonLifecycleError {
        guard case let .daemonExitedBeforeReady(configurationFileLabel) = lifecycleError else {
          return XCTFail("Expected an early-exit error, received \(lifecycleError)")
        }
        XCTAssertEqual(configurationFileLabel, expectedConfigurationLabel)
        XCTAssertFalse(lifecycleError.localizedDescription.contains(otherConfigurationLabel))
      }
    }
  }

  @MainActor
  func test_should_report_when_an_unowned_server_does_not_stop() async {
    let daemonLifecycleController = DaemonLifecycleController(
      supervisorClient: StuckExternalSupervisorClient()
    )

    do {
      _ = try await daemonLifecycleController.restartDaemon()
      XCTFail("A still-running unowned server must not be reported as restarted")
    } catch {
      XCTAssertEqual(error.localizedDescription, "The existing server did not stop; quit it and retry")
    }
  }
}

private struct DaemonLifecycleTestContext {
  let temporaryDirectoryURL: URL
  let ownershipRecordURL: URL
  let daemonExecutableURL: URL
  let applicationIdentity: ApplicationIdentity

  init(
    daemonExecutablePath: String,
    applicationIdentity: ApplicationIdentity = testApplicationIdentity(channel: .stable)
  ) throws {
    temporaryDirectoryURL = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(
      at: temporaryDirectoryURL, withIntermediateDirectories: true)
    ownershipRecordURL = temporaryDirectoryURL.appendingPathComponent("menu-owned-daemon.json")
    daemonExecutableURL = URL(fileURLWithPath: daemonExecutablePath)
    self.applicationIdentity = applicationIdentity
  }

  @MainActor
  func makeController(
    supervisorClient: any SupervisorClient,
    daemonArguments: [String]? = nil,
    readinessTimeout: Duration = .milliseconds(250)
  ) -> DaemonLifecycleController {
    DaemonLifecycleController(
      supervisorClient: supervisorClient,
      applicationIdentity: applicationIdentity,
      readinessTimeout: readinessTimeout,
      readinessPollInterval: .milliseconds(5),
      menuExecutableURL: URL(
        fileURLWithPath: "/Applications/Astronomical.app/Contents/MacOS/astronomical-menu"),
      daemonExecutableURL: daemonExecutableURL,
      ownershipRecordURL: ownershipRecordURL,
      daemonArguments: daemonArguments)
  }

  func removeTemporaryDirectory() {
    try? FileManager.default.removeItem(at: temporaryDirectoryURL)
  }
}

private actor DelayedReadinessSupervisorClient: SupervisorClient {
  private let readyAfterCheckCount: Int
  private var readinessCheckCount = 0

  init(readyAfterCheckCount: Int) {
    self.readyAfterCheckCount = readyAfterCheckCount
  }

  func fetchStatus() async throws -> SupervisorStatusDocument { .unavailable }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    ConfigurationReloadResult(message: "Config reloaded")
  }

  func requestShutdown() async throws {}

  func healthIsAvailable() async -> Bool { false }

  func expectedInstanceIsHealthy() async -> Bool {
    readinessCheckCount += 1
    return readinessCheckCount >= readyAfterCheckCount
  }
}

private actor ControlledReadinessSupervisorClient: SupervisorClient {
  private var expectedInstanceIsReady = false
  private var readinessWasChecked = false

  func fetchStatus() async throws -> SupervisorStatusDocument { .unavailable }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    ConfigurationReloadResult(message: "Config reloaded")
  }

  func requestShutdown() async throws {}

  func healthIsAvailable() async -> Bool { false }

  func expectedInstanceIsHealthy() async -> Bool {
    readinessWasChecked = true
    return expectedInstanceIsReady
  }

  func waitUntilReadinessWasChecked() async throws {
    let waitClock = ContinuousClock()
    let waitDeadline = waitClock.now.advanced(by: .seconds(1))
    while !readinessWasChecked, waitClock.now < waitDeadline {
      try await Task.sleep(for: .milliseconds(5))
    }
    if !readinessWasChecked { throw ReadinessTestError.readinessWasNotChecked }
  }

  func confirmExpectedInstanceHealth() {
    expectedInstanceIsReady = true
  }
}

private actor ExistingDelayedReadinessSupervisorClient: SupervisorClient {
  private let readyAfterCheckCount: Int
  private var readinessCheckCount = 0

  init(readyAfterCheckCount: Int) { self.readyAfterCheckCount = readyAfterCheckCount }

  func fetchStatus() async throws -> SupervisorStatusDocument { .unavailable }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    ConfigurationReloadResult(message: "Config reloaded")
  }

  func requestShutdown() async throws {}

  func healthIsAvailable() async -> Bool { true }

  func expectedInstanceIsHealthy() async -> Bool {
    readinessCheckCount += 1
    return readinessCheckCount >= readyAfterCheckCount
  }
}

private actor RestartCompletionState {
  private(set) var message: String?

  func record(message: String) {
    self.message = message
  }
}

private enum ReadinessTestError: Error {
  case readinessWasNotChecked
}

private func testApplicationIdentity(channel: ApplicationChannel) -> ApplicationIdentity {
  ApplicationIdentity(
    channel: channel,
    supervisorPort: channel.defaultSupervisorPort,
    stateDirectoryName: channel.stateDirectoryName,
    version: "1.0.0",
    buildNumber: "1",
    commit: "abcdef0",
    isDirty: false)
}

private struct StuckExternalSupervisorClient: SupervisorClient {
  func fetchStatus() async throws -> SupervisorStatusDocument { .unavailable }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    ConfigurationReloadResult(message: "Config reloaded")
  }

  func requestShutdown() async throws {}

  func healthIsAvailable() async -> Bool { true }
}
