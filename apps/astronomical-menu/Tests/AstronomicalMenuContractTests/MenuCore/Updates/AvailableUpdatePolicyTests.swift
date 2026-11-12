import XCTest

@testable import AstronomicalMenuCore

/// The agent may only mention a version the main app's updater would then
/// offer, so these rules mirror the updater's own filters.
final class AvailableUpdatePolicyTests: XCTestCase {
  private static let installedBuildNumber = 2213
  private static let sonoma = OperatingSystemVersion(majorVersion: 14, minorVersion: 4, patchVersion: 1)

  private func preferences(
    checksAutomatically: Bool = true,
    allowedFeedChannels: Set<String> = [],
    skippedBuildNumber: Int? = nil
  ) -> UpdateNoticePreferences {
    UpdateNoticePreferences(
      checksAutomatically: checksAutomatically,
      allowedFeedChannels: allowedFeedChannels,
      skippedBuildNumber: skippedBuildNumber)
  }

  private func release(
    _ buildNumber: Int, _ displayVersion: String = "0.3.0", channelName: String? = nil,
    minimumSystemVersion: String? = nil
  ) -> UpdateFeedItem {
    UpdateFeedItem(
      buildNumber: buildNumber, displayVersion: displayVersion, channelName: channelName,
      minimumSystemVersion: minimumSystemVersion)
  }

  private func newestUpdate(
    in feedItems: [UpdateFeedItem],
    preferences: UpdateNoticePreferences? = nil,
    operatingSystemVersion: OperatingSystemVersion = AvailableUpdatePolicyTests.sonoma
  ) -> AvailableUpdate? {
    AvailableUpdatePolicy.newestApplicableUpdate(
      feedItems: feedItems,
      installedBuildNumber: Self.installedBuildNumber,
      preferences: preferences ?? self.preferences(),
      operatingSystemVersion: operatingSystemVersion)
  }

  func test_should_offer_a_build_newer_than_the_installed_one() {
    XCTAssertEqual(
      newestUpdate(in: [release(2214, "0.3.0")]),
      AvailableUpdate(displayVersion: "0.3.0", buildNumber: 2214))
  }

  func test_should_offer_nothing_for_the_installed_build_or_an_older_one() {
    XCTAssertNil(newestUpdate(in: [release(2213)]), "the installed build is not an update")
    XCTAssertNil(newestUpdate(in: [release(2212), release(1)]))
  }

  func test_should_offer_nothing_from_an_empty_feed() {
    XCTAssertNil(newestUpdate(in: []))
  }

  func test_should_offer_the_newest_of_several_newer_builds_whatever_their_order() {
    let feedItems = [release(2220, "0.3.1"), release(2230, "0.3.2"), release(2215, "0.3.0")]

    XCTAssertEqual(newestUpdate(in: feedItems)?.displayVersion, "0.3.2")
  }

  func test_should_offer_nothing_while_automatic_checks_are_off() {
    XCTAssertNil(newestUpdate(in: [release(2300)], preferences: preferences(checksAutomatically: false)))
  }

  func test_should_offer_only_releases_of_the_selected_channel() {
    let feedItems = [
      release(2300, "0.4.0-dev", channelName: "development"),
      release(2250, "0.3.5-rc", channelName: "release-candidate"),
      release(2220, "0.3.1"),
    ]

    XCTAssertEqual(newestUpdate(in: feedItems)?.displayVersion, "0.3.1", "Stable sees Stable releases only")
    XCTAssertEqual(
      newestUpdate(in: feedItems, preferences: preferences(allowedFeedChannels: ["release-candidate"]))?
        .displayVersion, "0.3.5-rc")
    XCTAssertEqual(
      newestUpdate(
        in: feedItems,
        preferences: preferences(allowedFeedChannels: ["release-candidate", "development"]))?
        .displayVersion, "0.4.0-dev")
  }

  func test_should_not_mention_a_skipped_build_or_anything_below_it() {
    let feedItems = [release(2220, "0.3.1"), release(2230, "0.3.2")]

    XCTAssertNil(newestUpdate(in: feedItems, preferences: preferences(skippedBuildNumber: 2230)))
    XCTAssertNil(newestUpdate(in: feedItems, preferences: preferences(skippedBuildNumber: 9999)))
  }

  func test_should_offer_a_build_newer_than_the_skipped_one() {
    let feedItems = [release(2220, "0.3.1"), release(2230, "0.3.2")]

    XCTAssertEqual(
      newestUpdate(in: feedItems, preferences: preferences(skippedBuildNumber: 2220))?.displayVersion,
      "0.3.2")
  }

  func test_should_offer_only_releases_this_macos_can_run() {
    let requiresNewerSystem = [release(2300, minimumSystemVersion: "15.0")]
    XCTAssertNil(newestUpdate(in: requiresNewerSystem))
    XCTAssertNil(newestUpdate(in: [release(2300, minimumSystemVersion: "14.4.2")]), "one patch too new")

    XCTAssertNotNil(newestUpdate(in: [release(2300, minimumSystemVersion: "14.4.1")]), "exactly this system")
    XCTAssertNotNil(newestUpdate(in: [release(2300, minimumSystemVersion: "14.4")]))
    XCTAssertNotNil(newestUpdate(in: [release(2300, minimumSystemVersion: "14")]))
    XCTAssertNotNil(newestUpdate(in: [release(2300, minimumSystemVersion: "13.6.9")]))
  }

  func test_should_fall_back_to_an_older_release_when_the_newest_needs_a_newer_macos() {
    let feedItems = [release(2300, "0.4.0", minimumSystemVersion: "16.0"), release(2220, "0.3.1")]

    XCTAssertEqual(newestUpdate(in: feedItems)?.displayVersion, "0.3.1")
  }

  func test_should_leave_an_unreadable_minimum_system_to_the_updater() {
    XCTAssertNotNil(newestUpdate(in: [release(2300, minimumSystemVersion: "fourteen")]))
    XCTAssertNotNil(newestUpdate(in: [release(2300, minimumSystemVersion: "")]))
  }

  /// The store build shows no update control of its own; every direct build
  /// offers the check, which its main app carries out.
  func test_should_offer_the_update_check_on_direct_channels_only() {
    XCTAssertTrue(updateCheckSupported(applicationChannel: .stable))
    XCTAssertTrue(updateCheckSupported(applicationChannel: .development))
    XCTAssertFalse(updateCheckSupported(applicationChannel: .appStore))
  }
}
