// Proves the user-triggered update journey and channel isolation without contacting the public feed.

import XCTest

@testable import AstronomicalMenuCore
@testable import AstronomicalMenuSparkleUpdateController

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
final class AppStoreUpdateControllerTests: XCTestCase {
  func test_should_report_that_store_builds_do_not_surface_update_controls() {
    let updateController = AppStoreUpdateController()

    XCTAssertFalse(updateController.supportsUserUpdateControls)
  }

  func test_should_never_offer_an_update_check_on_store_builds() {
    let updateController = AppStoreUpdateController()

    XCTAssertFalse(updateController.canCheckForUpdates)
  }

  func test_should_ignore_manual_and_automatic_update_actions_on_store_builds() {
    let updateController = AppStoreUpdateController()

    requestManualApplicationUpdateCheck(using: updateController)
    setAutomaticApplicationUpdateChecks(true, using: updateController)

    XCTAssertFalse(updateController.automaticallyChecksForUpdates)
  }

  func test_should_keep_the_default_update_channel_fixed_to_stable_on_store_builds() {
    let updateController = AppStoreUpdateController()

    updateController.selectUpdateChannel(.development)

    XCTAssertEqual(updateController.selectedChannel, .stable)
  }

  func test_should_default_the_controller_to_store_semantics_without_an_installer() {
    let updateController = ApplicationUpdateControllerInstaller.makeController(
      applicationChannel: .stable)

    XCTAssertFalse(updateController.supportsUserUpdateControls)
  }
}

@MainActor
private final class RecordingApplicationUpdateChecker: ApplicationUpdateChecking {
  let supportsUserUpdateControls = true
  let canCheckForUpdates: Bool
  var automaticallyChecksForUpdates = true
  private(set) var selectedChannel = ApplicationUpdateChannel.stable
  private(set) var updateCheckRequestCount = 0

  init(canCheckForUpdates: Bool) {
    self.canCheckForUpdates = canCheckForUpdates
  }

  func checkForUpdates() {
    updateCheckRequestCount += 1
  }

  func start() {}

  func selectUpdateChannel(_: ApplicationUpdateChannel) {}
}
