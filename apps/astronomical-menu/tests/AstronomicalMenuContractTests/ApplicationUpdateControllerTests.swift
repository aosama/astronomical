// Proves the user-triggered update journey and channel isolation without contacting the public feed.

import XCTest

@testable import AstronomicalMenuCore
@testable import AstronomicalAppCore
@testable import AstronomicalSparkleUpdateController

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
final class SparkleUnbundledUpdateControllerTests: XCTestCase {
  // XCTest's tearDown override runs nonisolated, so the immutable Sendable
  // suite name must be nonisolated to stay reachable from it without an
  // actor-isolation warning (which the strict Swift build treats as an error).
  private nonisolated static let userDefaultsSuiteName = "AstronomicalMenuContractTests.unbundledSparkle"

  override func tearDown() {
    UserDefaults.standard.removePersistentDomain(forName: Self.userDefaultsSuiteName)
    super.tearDown()
  }

  private func makeUnbundledUpdateController() throws -> SparkleApplicationUpdateController {
    let ephemeralUserDefaults = try XCTUnwrap(
      UserDefaults(suiteName: Self.userDefaultsSuiteName))
    ephemeralUserDefaults.removePersistentDomain(forName: Self.userDefaultsSuiteName)
    return SparkleApplicationUpdateController(
      applicationChannel: .development,
      userDefaults: ephemeralUserDefaults,
      isHostPackagedApplication: false
    )
  }

  func test_should_keep_update_checks_unavailable_for_an_unbundled_host() throws {
    let updateController = try makeUnbundledUpdateController()

    updateController.start()

    XCTAssertFalse(updateController.canCheckForUpdates)
  }

  func test_should_ignore_manual_and_automatic_update_actions_for_an_unbundled_host() throws {
    let updateController = try makeUnbundledUpdateController()

    updateController.start()
    updateController.checkForUpdates()
    updateController.automaticallyChecksForUpdates = true

    XCTAssertFalse(updateController.automaticallyChecksForUpdates)
  }

  func test_should_still_persist_the_selected_channel_for_an_unbundled_host() throws {
    let updateController = try makeUnbundledUpdateController()

    updateController.selectUpdateChannel(.releaseCandidate)

    XCTAssertEqual(updateController.selectedChannel, .releaseCandidate)
    XCTAssertEqual(
      ephemeralPersistedChannelValue(),
      ApplicationUpdateChannel.releaseCandidate.rawValue)
  }

  private func ephemeralPersistedChannelValue() -> String? {
    UserDefaults(suiteName: Self.userDefaultsSuiteName)?.string(
      forKey: "AstronomicalSelectedUpdateChannel")
  }

  /// The store installs updates itself, so a store build never has one
  /// waiting for quit and the main app never stops its agent for it.
  func test_should_never_report_an_update_waiting_for_quit_in_a_store_build() {
    let storeController = AppStoreUpdateController()

    XCTAssertFalse(storeController.installsUpdateOnQuit)
    XCTAssertNil(storeController.prepareForRelaunch)
  }

  /// A direct build has nothing waiting until Sparkle says so.
  func test_should_report_no_update_waiting_for_quit_before_sparkle_announces_one() {
    let sparkleController = SparkleApplicationUpdateController(
      applicationChannel: .stable, isHostPackagedApplication: false)

    XCTAssertFalse(sparkleController.installsUpdateOnQuit)
  }

  /// An update that installs at quit has no relaunch to hook, so the main app
  /// must stop its agent as it terminates. This structural test pins that
  /// wiring, which no behavioral test reaches without a real update.
  func test_should_stop_the_agent_at_quit_when_an_update_waits_for_quit() throws {
    let packageDirectoryURL = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let applicationSource = try String(
      contentsOf: packageDirectoryURL.appendingPathComponent(
        "Sources/AstronomicalAppCore/AstronomicalApplication.swift"),
      encoding: .utf8)
    let willTerminateStart = try XCTUnwrap(applicationSource.range(of: "func applicationWillTerminate"))
    let stopOnQuit = try XCTUnwrap(
      applicationSource.range(
        of: "if applicationUpdateController.installsUpdateOnQuit { stopAgentBeforeUpdate() }"))

    XCTAssertLessThan(willTerminateStart.lowerBound, stopOnQuit.lowerBound)
  }
}


@MainActor
private final class RecordingApplicationUpdateChecker: ApplicationUpdateChecking {
  let supportsUserUpdateControls = true
  let canCheckForUpdates: Bool
  var automaticallyChecksForUpdates = true
  private(set) var selectedChannel = ApplicationUpdateChannel.stable
  private(set) var updateCheckRequestCount = 0
  var prepareForRelaunch: (@MainActor () -> Void)?
  let installsUpdateOnQuit = false

  init(canCheckForUpdates: Bool) {
    self.canCheckForUpdates = canCheckForUpdates
  }

  func checkForUpdates() {
    updateCheckRequestCount += 1
  }

  func start() {}

  func selectUpdateChannel(_: ApplicationUpdateChannel) {}
}
