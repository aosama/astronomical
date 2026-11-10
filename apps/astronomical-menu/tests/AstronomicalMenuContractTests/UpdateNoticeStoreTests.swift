import XCTest

@testable import AstronomicalMenuCore

/// What the popover is told about an update.
@MainActor
final class UpdateNoticeStoreTests: XCTestCase {
  private static let installedBuildNumber = 2213

  private func makeStore(
    feedFetcher: StubUpdateFeedFetcher, settingsReader: StubUpdateSettingsReader
  ) -> UpdateNoticeStore {
    UpdateNoticeStore(
      installedBuildNumber: Self.installedBuildNumber, settingsReader: settingsReader,
      feedFetcher: feedFetcher,
      operatingSystemVersion: OperatingSystemVersion(majorVersion: 14, minorVersion: 0, patchVersion: 0))
  }

  func test_should_show_a_newer_version_found_in_the_feed() async {
    let feedFetcher = StubUpdateFeedFetcher(feedText: updateFeedText(buildNumber: 2300, displayVersion: "0.3.0"))
    let store = makeStore(feedFetcher: feedFetcher, settingsReader: StubUpdateSettingsReader())

    await store.checkFeed()

    XCTAssertEqual(store.availableUpdate, AvailableUpdate(displayVersion: "0.3.0", buildNumber: 2300))
  }

  func test_should_show_nothing_when_the_feed_holds_only_the_installed_build() async {
    let feedFetcher = StubUpdateFeedFetcher(feedText: updateFeedText(buildNumber: 2213, displayVersion: "0.2.99"))
    let store = makeStore(feedFetcher: feedFetcher, settingsReader: StubUpdateSettingsReader())

    await store.checkFeed()

    XCTAssertNil(store.availableUpdate)
  }

  func test_should_keep_a_known_update_when_a_later_download_fails() async {
    let feedFetcher = StubUpdateFeedFetcher(feedText: updateFeedText(buildNumber: 2300, displayVersion: "0.3.0"))
    let store = makeStore(feedFetcher: feedFetcher, settingsReader: StubUpdateSettingsReader())
    await store.checkFeed()

    feedFetcher.serve(nil)
    await store.checkFeed()
    feedFetcher.serve("<html>Bad gateway")
    await store.checkFeed()

    XCTAssertEqual(store.availableUpdate?.displayVersion, "0.3.0")
  }

  func test_should_show_nothing_when_the_first_download_fails() async {
    let store = makeStore(
      feedFetcher: StubUpdateFeedFetcher(feedText: nil), settingsReader: StubUpdateSettingsReader())

    await store.checkFeed()

    XCTAssertNil(store.availableUpdate)
  }

  func test_should_not_ask_the_network_while_automatic_checks_are_off() async {
    let feedFetcher = StubUpdateFeedFetcher(feedText: updateFeedText(buildNumber: 2300, displayVersion: "0.3.0"))
    let store = makeStore(
      feedFetcher: feedFetcher, settingsReader: StubUpdateSettingsReader(checksAutomatically: false))

    await store.checkFeed()

    XCTAssertEqual(feedFetcher.fetchCount, 0)
    XCTAssertNil(store.availableUpdate)
  }

  func test_should_not_ask_the_network_when_the_build_has_no_update_feed() async {
    let feedFetcher = StubUpdateFeedFetcher(feedText: updateFeedText(buildNumber: 2300, displayVersion: "0.3.0"))
    let settingsReader = StubUpdateSettingsReader()
    settingsReader.updateSettings = nil
    let store = makeStore(feedFetcher: feedFetcher, settingsReader: settingsReader)

    await store.checkFeed()
    store.applyCurrentSettings()

    XCTAssertEqual(feedFetcher.fetchCount, 0)
    XCTAssertNil(store.availableUpdate)
  }

  func test_should_drop_a_version_skipped_since_the_last_download_without_another_request() async {
    let feedFetcher = StubUpdateFeedFetcher(feedText: updateFeedText(buildNumber: 2300, displayVersion: "0.3.0"))
    let settingsReader = StubUpdateSettingsReader()
    let store = makeStore(feedFetcher: feedFetcher, settingsReader: settingsReader)
    await store.checkFeed()

    settingsReader.setPreferences(skippedBuildNumber: 2300)
    store.applyCurrentSettings()

    XCTAssertNil(store.availableUpdate)
    XCTAssertEqual(feedFetcher.fetchCount, 1)
  }

  func test_should_drop_a_known_update_once_automatic_checks_are_turned_off() async {
    let feedFetcher = StubUpdateFeedFetcher(feedText: updateFeedText(buildNumber: 2300, displayVersion: "0.3.0"))
    let settingsReader = StubUpdateSettingsReader()
    let store = makeStore(feedFetcher: feedFetcher, settingsReader: settingsReader)
    await store.checkFeed()

    settingsReader.setPreferences(checksAutomatically: false)
    store.applyCurrentSettings()

    XCTAssertNil(store.availableUpdate)
  }
}

final class UpdateAvailableRowTests: XCTestCase {
  func test_should_name_the_available_version_in_plain_words() {
    XCTAssertEqual(
      UpdateAvailableRow.title(for: AvailableUpdate(displayVersion: "0.3.0", buildNumber: 2300)),
      "Version 0.3.0 is available")
  }
}
