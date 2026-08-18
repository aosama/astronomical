// Proves the user-triggered update journey and channel isolation without contacting the public feed.

import XCTest

@testable import AstronomicalMenu

@MainActor
final class ApplicationUpdateControllerTests: XCTestCase {
  func test_should_start_an_update_check_when_the_user_requests_one() {
    let updateChecker = RecordingApplicationUpdateChecker(canCheckForUpdates: true)

    requestManualApplicationUpdateCheck(using: updateChecker)

    XCTAssertEqual(updateChecker.updateCheckRequestCount, 1)
  }

  func test_should_not_start_another_update_check_while_one_is_unavailable() {
    let updateChecker = RecordingApplicationUpdateChecker(canCheckForUpdates: false)

    requestManualApplicationUpdateCheck(using: updateChecker)

    XCTAssertEqual(updateChecker.updateCheckRequestCount, 0)
  }

  func test_should_limit_stable_builds_to_the_default_update_channel() {
    XCTAssertEqual(sparkleUpdateChannels(for: .stable), [])
  }

  func test_should_allow_development_builds_to_receive_development_updates() {
    XCTAssertEqual(
      sparkleUpdateChannels(for: .development),
      ["release-candidate", "development"]
    )
  }

  func test_should_allow_release_candidate_users_to_receive_release_candidates() {
    XCTAssertEqual(sparkleUpdateChannels(for: .releaseCandidate), ["release-candidate"])
  }

  func test_should_change_the_persisted_automatic_check_preference() {
    let updateChecker = RecordingApplicationUpdateChecker(canCheckForUpdates: true)

    setAutomaticApplicationUpdateChecks(false, using: updateChecker)

    XCTAssertFalse(updateChecker.automaticallyChecksForUpdates)
  }
}

@MainActor
private final class RecordingApplicationUpdateChecker: ApplicationUpdateChecking {
  let canCheckForUpdates: Bool
  var automaticallyChecksForUpdates = true
  private(set) var updateCheckRequestCount = 0

  init(canCheckForUpdates: Bool) {
    self.canCheckForUpdates = canCheckForUpdates
  }

  func checkForUpdates() {
    updateCheckRequestCount += 1
  }
}
