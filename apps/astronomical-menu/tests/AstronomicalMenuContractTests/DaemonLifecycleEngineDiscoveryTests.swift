import Foundation
import XCTest

@testable import AstronomicalMenuCore

/// The agent is told only where it runs from and must find the engine in the
/// main bundle that contains it. These tests build that bundle layout on disk
/// and let the controller discover the engine, rather than handing it one.
final class DaemonLifecycleEngineDiscoveryTests: XCTestCase {
  @MainActor
  func test_should_start_the_engine_of_the_main_bundle_that_contains_the_agent() async throws {
    let bundleFixture = try NestedAgentBundleFixture()
    defer { bundleFixture.remove() }
    try bundleFixture.installEngine()
    let daemonLifecycleController = bundleFixture.makeController()
    defer { daemonLifecycleController.stopOwnedDaemon() }

    try await daemonLifecycleController.startDaemonIfNeeded()

    XCTAssertTrue(daemonLifecycleController.ownsDaemon)
    let ownershipRecord = try DaemonOwnershipStore(ownershipRecordURL: bundleFixture.ownershipRecordURL).load()
    XCTAssertEqual(ownershipRecord.daemonExecutablePath, bundleFixture.engineExecutableURL.path)
    XCTAssertEqual(ownershipRecord.menuExecutablePath, bundleFixture.agentExecutableURL.path)
  }

  @MainActor
  func test_should_report_a_missing_engine_when_the_main_bundle_carries_none() async throws {
    let bundleFixture = try NestedAgentBundleFixture()
    defer { bundleFixture.remove() }
    let daemonLifecycleController = bundleFixture.makeController()

    do {
      try await daemonLifecycleController.startDaemonIfNeeded()
      XCTFail("A bundle without an engine must not be reported as started")
    } catch let lifecycleError as DaemonLifecycleError {
      guard case .bundledDaemonUnavailable = lifecycleError else {
        return XCTFail("Expected a missing-engine error, received \(lifecycleError)")
      }
    }
    XCTAssertFalse(daemonLifecycleController.ownsDaemon)
  }

  /// An engine file without its executable bit cannot be spawned, so it must
  /// read as missing instead of failing later inside the spawn.
  @MainActor
  func test_should_report_a_missing_engine_when_the_engine_file_is_not_executable() async throws {
    let bundleFixture = try NestedAgentBundleFixture()
    defer { bundleFixture.remove() }
    try bundleFixture.installEngine(posixPermissions: 0o644)
    let daemonLifecycleController = bundleFixture.makeController()

    do {
      try await daemonLifecycleController.startDaemonIfNeeded()
      XCTFail("A non-executable engine must not be reported as started")
    } catch let lifecycleError as DaemonLifecycleError {
      guard case .bundledDaemonUnavailable = lifecycleError else {
        return XCTFail("Expected a missing-engine error, received \(lifecycleError)")
      }
    }
  }
}

/// A main bundle with a nested agent bundle, laid out as the packaged app is.
private struct NestedAgentBundleFixture {
  let scratchDirectoryURL: URL
  let agentExecutableURL: URL
  let engineExecutableURL: URL
  let ownershipRecordURL: URL

  init() throws {
    // The temporary directory sits behind a symbolic link on macOS; resolving
    // it here keeps the expected paths equal to the resolved ones recorded.
    scratchDirectoryURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("astronomical-engine-discovery-\(UUID().uuidString)", isDirectory: true)
      .resolvingSymlinksInPath()
    let mainExecutablesDirectoryURL = scratchDirectoryURL
      .appendingPathComponent("Astronomical.app/Contents/MacOS", isDirectory: true)
    let agentExecutablesDirectoryURL = scratchDirectoryURL.appendingPathComponent(
      "Astronomical.app/Contents/Library/LoginItems/Astronomical Agent.app/Contents/MacOS", isDirectory: true)
    try FileManager.default.createDirectory(at: mainExecutablesDirectoryURL, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: agentExecutablesDirectoryURL, withIntermediateDirectories: true)
    agentExecutableURL = agentExecutablesDirectoryURL.appendingPathComponent("astronomical-menu")
    engineExecutableURL = mainExecutablesDirectoryURL.appendingPathComponent("astronomicald")
    ownershipRecordURL = scratchDirectoryURL.appendingPathComponent("state/menu-owned-daemon.json")
  }

  func installEngine(posixPermissions: Int = 0o755) throws {
    let engineScript = "#!/bin/sh\nexec /bin/sleep 30\n"
    XCTAssertTrue(
      FileManager.default.createFile(
        atPath: engineExecutableURL.path, contents: Data(engineScript.utf8),
        attributes: [.posixPermissions: posixPermissions]))
  }

  @MainActor
  func makeController() -> DaemonLifecycleController {
    DaemonLifecycleController(
      daemonClient: ReadyOnceStartedDaemonClient(),
      applicationIdentity: ApplicationIdentity(
        channel: .stable,
        daemonPort: ApplicationChannel.stable.defaultDaemonPort,
        stateDirectoryName: ApplicationChannel.stable.stateDirectoryName,
        version: "1.0.0",
        buildNumber: "1",
        commit: "abcdef0",
        isDirty: false),
      readinessTimeout: .seconds(2),
      readinessPollInterval: .milliseconds(5),
      menuExecutableURL: agentExecutableURL,
      ownershipRecordURL: ownershipRecordURL,
      daemonArguments: [])
  }

  func remove() { try? FileManager.default.removeItem(at: scratchDirectoryURL) }
}

/// No engine answers before the spawn, and the spawned one is ready at once.
private struct ReadyOnceStartedDaemonClient: DaemonClient {
  func fetchStatus() async throws -> DaemonStatusDocument { .unavailable }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    ConfigurationReloadResult(message: "Config reloaded")
  }

  func requestShutdown() async throws {}

  func healthIsAvailable() async -> Bool { false }

  func expectedInstanceIsHealthy() async -> Bool { true }
}
