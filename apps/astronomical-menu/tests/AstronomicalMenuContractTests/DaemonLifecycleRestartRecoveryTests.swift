import XCTest

@testable import AstronomicalMenuCore

final class DaemonLifecycleRestartRecoveryTests: XCTestCase {
  func test_should_detect_the_lock_conflict_in_a_captured_explanation() {
    XCTAssertTrue(
      lockConflictIsIndicated(
        by: "server said: astronomicald startup failed: Astronomical is already running"))
    XCTAssertFalse(lockConflictIsIndicated(by: "server said: configuration failed"))
    XCTAssertFalse(lockConflictIsIndicated(by: nil))
  }

  func test_should_describe_an_exit_with_the_captured_failure_explanation() {
    let silentExit = DaemonLifecycleError.daemonExitedBeforeReady("~/.astronomical/config.json", nil)
    XCTAssertEqual(
      silentExit.errorDescription,
      "The server exited before becoming ready. Correct or reveal ~/.astronomical/config.json, then retry"
    )

    let explainedExit = DaemonLifecycleError.daemonExitedBeforeReady(
      "~/.astronomical/config.json",
      "server said: astronomicald startup failed: Astronomical is already running — last start failed [instance-lock]: Astronomical is already running (holder: pid=4242 instance=astronomicald)"
    )
    XCTAssertTrue(
      explainedExit.errorDescription?.contains("already running") == true)
    XCTAssertTrue(explainedExit.errorDescription?.contains("pid=4242") == true)
  }

  func test_should_decode_the_daemon_startup_failure_record() throws {
    let failureRecordJson = Data(
      """
      {"timestamp_unix_millis":1779000000000,"stage":"instance-lock","message":"Astronomical is already running for the selected instance","context":"pid=4242 instance=astronomicald","cause_chain":["Astronomical is already running for the selected instance"]}
      """.utf8)

    let failureRecord = try JSONDecoder().decode(StartupFailureSnapshot.self, from: failureRecordJson)
    XCTAssertEqual(failureRecord.stage, "instance-lock")
    XCTAssertEqual(
      failureRecord.message, "Astronomical is already running for the selected instance")
    XCTAssertEqual(failureRecord.context, "pid=4242 instance=astronomicald")
  }

  func test_should_report_a_free_instance_lock_when_no_daemon_holds_it() {
    let lockURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("astronomical-restart-lock-absent-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: lockURL) }

    // No daemon has ever run here: the lock file is absent, so the state is free.
    XCTAssertTrue(instanceLockIsFree(lockPath: lockURL.path))
  }

  func test_should_detect_a_held_instance_lock_while_another_process_flocks_it() {
    let lockURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("astronomical-restart-lock-\(UUID().uuidString)")
    FileManager.default.createFile(atPath: lockURL.path, contents: nil)
    defer { try? FileManager.default.removeItem(at: lockURL) }

    let lockFileDescriptor = open(lockURL.path, O_RDWR)
    XCTAssertGreaterThanOrEqual(lockFileDescriptor, 0)
    defer { close(lockFileDescriptor) }

    XCTAssertEqual(flock(lockFileDescriptor, LOCK_EX | LOCK_NB), 0)
    XCTAssertFalse(instanceLockIsFree(lockPath: lockURL.path))
    XCTAssertEqual(flock(lockFileDescriptor, LOCK_UN), 0)
    XCTAssertTrue(instanceLockIsFree(lockPath: lockURL.path))
  }
}

private final class UnavailableDaemonClientStub: DaemonClient {
  func fetchStatus() async throws -> DaemonStatusDocument {
    throw DaemonClientError.serverRejected("unavailable in test")
  }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    throw DaemonClientError.serverRejected("unavailable in test")
  }

  func requestShutdown() async throws {
    throw DaemonClientError.serverRejected("unavailable in test")
  }

  func updateMaximumMlxMemoryGigabytes(_ maximumMlxMemoryGigabytes: UInt64?) async throws -> String {
    throw DaemonClientError.serverRejected("unavailable in test")
  }

  func healthIsAvailable() async -> Bool { false }

  func expectedInstanceIsHealthy() async -> Bool { false }
}
