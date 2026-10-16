import Foundation
import XCTest

@testable import AstronomicalMenuCore

final class DaemonOwnershipContractTests: XCTestCase {
  func test_should_reject_an_ownership_record_for_another_executable() {
    let ownershipRecord = DaemonOwnershipRecord(
      menuProcessIdentifier: 10,
      menuExecutablePath: "/Applications/Astronomical.app/Contents/MacOS/astronomical-menu",
      daemonProcessIdentifier: 11,
      daemonExecutablePath: "/tmp/unrelated-daemon"
    )

    XCTAssertFalse(
      ownershipRecord.matchesExpectedExecutables(
        menuExecutablePath: "/Applications/Astronomical.app/Contents/MacOS/astronomical-menu",
        daemonExecutablePath: "/Applications/Astronomical.app/Contents/MacOS/astronomicald"
      )
    )
  }

  func test_should_atomically_create_and_replace_the_daemon_ownership_record() throws {
    let temporaryDirectoryURL = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString, isDirectory: true)
    defer { try? FileManager.default.removeItem(at: temporaryDirectoryURL) }
    let ownershipRecordURL = temporaryDirectoryURL.appendingPathComponent("menu-owned-daemon.json")
    let ownershipStore = DaemonOwnershipStore(ownershipRecordURL: ownershipRecordURL)
    let initialRecord = DaemonOwnershipRecord(
      menuProcessIdentifier: 10,
      menuExecutablePath: "/menu",
      daemonProcessIdentifier: 11,
      daemonExecutablePath: "/daemon"
    )
    let replacementRecord = DaemonOwnershipRecord(
      menuProcessIdentifier: 20,
      menuExecutablePath: "/menu",
      daemonProcessIdentifier: 21,
      daemonExecutablePath: "/daemon"
    )

    try ownershipStore.persist(initialRecord)
    try ownershipStore.persist(replacementRecord)

    XCTAssertEqual(try ownershipStore.load(), replacementRecord)
    XCTAssertFalse(
      FileManager.default.fileExists(atPath: ownershipRecordURL.appendingPathExtension("tmp").path))
  }

  func test_should_launch_an_owned_daemon_in_its_own_process_group() throws {
    let daemonProcess = try launchOwnedDaemonProcess(
      executableURL: URL(fileURLWithPath: "/bin/sleep"),
      arguments: ["30"]
    )
    defer {
      _ = kill(-daemonProcess.processIdentifier, SIGKILL)
      daemonProcess.waitUntilExit()
    }

    XCTAssertEqual(getpgid(daemonProcess.processIdentifier), daemonProcess.processIdentifier)
  }

  func test_should_parse_the_daemon_pid_from_the_instance_lock_note() {
    XCTAssertEqual(
      daemonProcessIdentifierFromInstanceLockNote("pid=4242 instance=astronomicald\n"), 4242)
    XCTAssertNil(daemonProcessIdentifierFromInstanceLockNote("instance=astronomicald\n"))
    XCTAssertNil(daemonProcessIdentifierFromInstanceLockNote("pid=not-a-number"))
    XCTAssertNil(daemonProcessIdentifierFromInstanceLockNote(nil))
  }

  @MainActor
  func test_should_stop_an_unowned_daemon_whose_executable_matches_the_bundle_on_quit() throws {
    let daemonProcess = try launchOwnedDaemonProcess(
      executableURL: URL(fileURLWithPath: "/bin/sleep"),
      arguments: ["30"]
    )
    defer {
      _ = kill(-daemonProcess.processIdentifier, SIGKILL)
      daemonProcess.waitUntilExit()
    }
    let lifecycleController = quitLifecycleController(
      daemonExecutableURL: URL(fileURLWithPath: "/bin/sleep"),
      lockHolderProcessIdentifier: daemonProcess.processIdentifier
    )

    lifecycleController.stopChannelDaemonOnApplicationQuit()

    XCTAssertTrue(
      waitForDaemonExit(daemonProcess),
      "quit must stop a healthy channel daemon the menu app did not spawn")
  }

  @MainActor
  func test_should_stop_an_unowned_daemon_whose_bundle_path_contains_a_space() throws {
    let spacedExecutableDirectory = FileManager.default.temporaryDirectory
      .appendingPathComponent("Astronomical Development.app", isDirectory: true)
      .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(
      at: spacedExecutableDirectory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: spacedExecutableDirectory) }
    let spacedDaemonExecutableURL = spacedExecutableDirectory.appendingPathComponent("astronomicald")
    try FileManager.default.copyItem(
      at: URL(fileURLWithPath: "/bin/sleep"), to: spacedDaemonExecutableURL)
    try FileManager.default.setAttributes(
      [.posixPermissions: 0o755], ofItemAtPath: spacedDaemonExecutableURL.path)
    let daemonProcess = try launchOwnedDaemonProcess(
      executableURL: spacedDaemonExecutableURL, arguments: ["30"])
    defer {
      _ = kill(-daemonProcess.processIdentifier, SIGKILL)
      daemonProcess.waitUntilExit()
    }
    let lifecycleController = quitLifecycleController(
      daemonExecutableURL: spacedDaemonExecutableURL,
      lockHolderProcessIdentifier: daemonProcess.processIdentifier
    )

    lifecycleController.stopChannelDaemonOnApplicationQuit()

    XCTAssertTrue(
      waitForDaemonExit(daemonProcess),
      "quit must stop a channel daemon whose bundle path contains a space")
  }

  @MainActor
  func test_should_leave_an_unowned_process_with_a_foreign_executable_running_on_quit() throws {
    let foreignProcess = try launchOwnedDaemonProcess(
      executableURL: URL(fileURLWithPath: "/bin/sleep"),
      arguments: ["30"]
    )
    defer {
      _ = kill(-foreignProcess.processIdentifier, SIGKILL)
      foreignProcess.waitUntilExit()
    }
    let lifecycleController = quitLifecycleController(
      daemonExecutableURL: URL(fileURLWithPath: "/opt/elsewhere/astronomicald"),
      lockHolderProcessIdentifier: foreignProcess.processIdentifier
    )

    lifecycleController.stopChannelDaemonOnApplicationQuit()

    XCTAssertFalse(
      waitForDaemonExit(foreignProcess),
      "quit must never stop a process that is not this bundle's daemon")
  }
}

/// Wires a lifecycle controller whose state directory holds an instance lock
/// naming `lockHolderProcessIdentifier`, with the daemon executable the quit
/// path will compare against fully injected, so no real daemon is needed.
@MainActor
private func quitLifecycleController(
  daemonExecutableURL: URL,
  lockHolderProcessIdentifier: pid_t
) -> DaemonLifecycleController {
  let stateDirectoryURL = FileManager.default.temporaryDirectory
    .appendingPathComponent(UUID().uuidString, isDirectory: true)
  let instanceLockNote = "pid=\(lockHolderProcessIdentifier) instance=astronomicald\n"
  try? FileManager.default.createDirectory(
    at: stateDirectoryURL, withIntermediateDirectories: true)
  FileManager.default.createFile(
    atPath: stateDirectoryURL.appendingPathComponent("instance.lock").path,
    contents: Data(instanceLockNote.utf8))
  return DaemonLifecycleController(
    daemonClient: NonInteractingDaemonClient(),
    daemonExecutableURL: daemonExecutableURL,
    ownershipRecordURL: stateDirectoryURL.appendingPathComponent("menu-owned-daemon.json")
  )
}

/// Polls for the spawned helper's exit for a bounded window instead of
/// blocking forever when the quit path fails to stop it.
private func waitForDaemonExit(_ daemonProcess: OwnedDaemonProcess) -> Bool {
  let waitClock = ContinuousClock()
  let waitDeadline = waitClock.now.advanced(by: .seconds(5))
  while waitClock.now < waitDeadline {
    if daemonProcess.reapIfExited() { return true }
    usleep(50_000)
  }
  return daemonProcess.reapIfExited()
}

private struct NonInteractingDaemonClient: DaemonClient {
  func fetchStatus() async throws -> DaemonStatusDocument { .unavailable }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    ConfigurationReloadResult(message: "Config reloaded")
  }

  func requestShutdown() async throws {}

  func healthIsAvailable() async -> Bool { false }
}
