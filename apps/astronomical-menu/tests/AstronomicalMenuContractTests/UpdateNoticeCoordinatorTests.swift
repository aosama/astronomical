import XCTest

@testable import AstronomicalMenuCore

/// When the agent posts its one notification per version.
@MainActor
final class UpdateNoticeCoordinatorTests: XCTestCase {
  private static let recordSuiteName = "UpdateNoticeCoordinatorTests.record"
  private var recordDefaults: UserDefaults!
  private var feedFetcher: StubUpdateFeedFetcher!
  private var presenter: StubUpdateNoticePresenter!
  private var mainApplicationPresence: StubMainApplicationPresence!

  override func setUp() async throws {
    recordDefaults = try XCTUnwrap(UserDefaults(suiteName: Self.recordSuiteName))
    recordDefaults.removePersistentDomain(forName: Self.recordSuiteName)
    feedFetcher = StubUpdateFeedFetcher(feedText: updateFeedText(buildNumber: 2300, displayVersion: "0.3.0"))
    presenter = StubUpdateNoticePresenter()
    mainApplicationPresence = StubMainApplicationPresence()
  }

  override func tearDown() async throws {
    recordDefaults.removePersistentDomain(forName: Self.recordSuiteName)
  }

  private func makeCoordinator() -> UpdateNoticeCoordinator {
    UpdateNoticeCoordinator(
      store: UpdateNoticeStore(
        installedBuildNumber: 2213, settingsReader: StubUpdateSettingsReader(), feedFetcher: feedFetcher,
        operatingSystemVersion: OperatingSystemVersion(majorVersion: 14, minorVersion: 0, patchVersion: 0)),
      presenter: presenter,
      announcementRecord: UpdateAnnouncementRecord(userDefaults: recordDefaults),
      mainApplicationPresence: mainApplicationPresence,
      activityIdentifier: "UpdateNoticeCoordinatorTests")
  }

  func test_should_announce_a_new_version_once_however_many_checks_see_it() async {
    let coordinator = makeCoordinator()

    await coordinator.checkAndAnnounce()
    await coordinator.checkAndAnnounce()

    XCTAssertEqual(presenter.presentedUpdates.map(\.displayVersion), ["0.3.0"])
    XCTAssertEqual(coordinator.store.availableUpdate?.displayVersion, "0.3.0", "the popover row stays")
  }

  func test_should_not_announce_the_same_version_again_after_the_agent_restarts() async {
    await makeCoordinator().checkAndAnnounce()

    await makeCoordinator().checkAndAnnounce()

    XCTAssertEqual(presenter.presentedUpdates.count, 1)
  }

  func test_should_announce_a_later_version_again() async {
    let coordinator = makeCoordinator()
    await coordinator.checkAndAnnounce()

    feedFetcher.serve(updateFeedText(buildNumber: 2400, displayVersion: "0.4.0"))
    await coordinator.checkAndAnnounce()

    XCTAssertEqual(presenter.presentedUpdates.map(\.displayVersion), ["0.3.0", "0.4.0"])
  }

  func test_should_announce_nothing_when_the_feed_holds_no_newer_version() async {
    feedFetcher.serve(updateFeedText(buildNumber: 2213, displayVersion: "0.2.99"))

    await makeCoordinator().checkAndAnnounce()

    XCTAssertTrue(presenter.presentedUpdates.isEmpty)
  }

  func test_should_wait_with_the_notification_while_the_main_app_is_open() async {
    let coordinator = makeCoordinator()
    mainApplicationPresence.isMainApplicationRunning = true

    await coordinator.checkAndAnnounce()
    XCTAssertTrue(presenter.presentedUpdates.isEmpty, "the main app's own updater speaks while it is open")
    XCTAssertNotNil(coordinator.store.availableUpdate, "the popover row shows regardless")

    mainApplicationPresence.isMainApplicationRunning = false
    await coordinator.checkAndAnnounce()

    XCTAssertEqual(presenter.presentedUpdates.count, 1, "announced at the next look once the app is closed")
  }

  func test_should_look_once_when_two_looks_fall_together() async {
    let coordinator = makeCoordinator()

    async let firstLook: Void = coordinator.checkAndAnnounce()
    async let secondLook: Void = coordinator.checkAndAnnounce()
    _ = await (firstLook, secondLook)

    XCTAssertEqual(feedFetcher.fetchCount, 1)
    XCTAssertEqual(presenter.presentedUpdates.count, 1)
  }

  func test_should_try_again_at_the_next_look_when_macos_did_not_take_the_notification() async {
    let coordinator = makeCoordinator()
    presenter.isNotificationAccepted = false
    await coordinator.checkAndAnnounce()

    presenter.isNotificationAccepted = true
    await coordinator.checkAndAnnounce()

    XCTAssertEqual(presenter.presentedUpdates.count, 1)
  }
}

final class UpdateAnnouncementRecordTests: XCTestCase {
  private static let recordSuiteName = "UpdateAnnouncementRecordTests"

  func test_should_need_an_announcement_only_for_builds_newer_than_the_last_announced() throws {
    let recordDefaults = try XCTUnwrap(UserDefaults(suiteName: Self.recordSuiteName))
    recordDefaults.removePersistentDomain(forName: Self.recordSuiteName)
    defer { recordDefaults.removePersistentDomain(forName: Self.recordSuiteName) }
    let announcementRecord = UpdateAnnouncementRecord(userDefaults: recordDefaults)
    let announcedUpdate = AvailableUpdate(displayVersion: "0.3.0", buildNumber: 2300)
    XCTAssertTrue(announcementRecord.needsAnnouncement(of: announcedUpdate), "nothing announced yet")

    announcementRecord.recordAnnouncement(of: announcedUpdate)

    XCTAssertFalse(announcementRecord.needsAnnouncement(of: announcedUpdate))
    XCTAssertFalse(
      announcementRecord.needsAnnouncement(of: AvailableUpdate(displayVersion: "0.2.99", buildNumber: 2213)))
    XCTAssertTrue(
      announcementRecord.needsAnnouncement(of: AvailableUpdate(displayVersion: "0.3.1", buildNumber: 2301)))
  }
}
