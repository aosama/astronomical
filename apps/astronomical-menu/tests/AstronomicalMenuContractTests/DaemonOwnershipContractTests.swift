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
}
