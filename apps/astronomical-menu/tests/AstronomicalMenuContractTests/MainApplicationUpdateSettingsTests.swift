import XCTest

@testable import AstronomicalMenuCore

/// The agent keeps no update settings of its own; it must read the main
/// app's the way the updater does.
final class MainApplicationUpdateSettingsTests: XCTestCase {
  private static let userDefaultsSuiteName = "MainApplicationUpdateSettingsTests"
  /// What a direct build's main app bundle ships with.
  private static var bundledInfo: [String: Any] {
    ["SUFeedURL": "https://example.invalid/appcast.xml", "SUEnableAutomaticChecks": true]
  }
  private var mainApplicationDefaults: UserDefaults!

  override func setUpWithError() throws {
    mainApplicationDefaults = try XCTUnwrap(UserDefaults(suiteName: Self.userDefaultsSuiteName))
    mainApplicationDefaults.removePersistentDomain(forName: Self.userDefaultsSuiteName)
  }

  override func tearDown() {
    mainApplicationDefaults.removePersistentDomain(forName: Self.userDefaultsSuiteName)
  }

  private func settings(
    info: [String: Any] = MainApplicationUpdateSettingsTests.bundledInfo,
    applicationChannel: ApplicationChannel = .stable
  ) -> MainApplicationUpdateSettings? {
    MainApplicationUpdateSettings(
      mainApplicationInfo: info, mainApplicationDefaults: mainApplicationDefaults,
      applicationChannel: applicationChannel)
  }

  func test_should_use_what_the_bundle_ships_with_before_anyone_has_chosen() throws {
    let updateSettings = try XCTUnwrap(settings())

    XCTAssertEqual(updateSettings.feedURL.absoluteString, "https://example.invalid/appcast.xml")
    XCTAssertEqual(
      updateSettings.preferences,
      UpdateNoticePreferences(checksAutomatically: true, allowedFeedChannels: [], skippedBuildNumber: nil))
  }

  func test_should_have_no_settings_without_an_update_feed() {
    XCTAssertNil(settings(info: [:]), "a store build ships no feed")
    XCTAssertNil(settings(info: ["SUFeedURL": 42]))
    XCTAssertNil(settings(info: ["SUFeedURL": ""]))
  }

  func test_should_follow_the_choice_to_stop_checking_automatically() throws {
    mainApplicationDefaults.set(false, forKey: "SUEnableAutomaticChecks")

    XCTAssertFalse(try XCTUnwrap(settings()).preferences.checksAutomatically)
  }

  func test_should_follow_the_choice_to_check_when_the_bundle_ships_with_checks_off() throws {
    var infoWithChecksOff = Self.bundledInfo
    infoWithChecksOff["SUEnableAutomaticChecks"] = false
    XCTAssertFalse(try XCTUnwrap(settings(info: infoWithChecksOff)).preferences.checksAutomatically)

    mainApplicationDefaults.set(true, forKey: "SUEnableAutomaticChecks")

    XCTAssertTrue(try XCTUnwrap(settings(info: infoWithChecksOff)).preferences.checksAutomatically)
  }

  func test_should_not_check_when_neither_the_bundle_nor_the_person_says_to() throws {
    let infoWithoutTheKey: [String: Any] = ["SUFeedURL": "https://example.invalid/appcast.xml"]

    XCTAssertFalse(try XCTUnwrap(settings(info: infoWithoutTheKey)).preferences.checksAutomatically)
  }

  func test_should_admit_the_feed_channels_of_the_selected_update_channel() throws {
    XCTAssertEqual(
      try XCTUnwrap(settings(applicationChannel: .development)).preferences.allowedFeedChannels,
      ["release-candidate", "development"], "a Development build follows every channel by default")

    mainApplicationDefaults.set("release-candidate", forKey: UpdateChannelSelection.userDefaultsKey)

    XCTAssertEqual(try XCTUnwrap(settings()).preferences.allowedFeedChannels, ["release-candidate"])
  }

  func test_should_read_the_build_skipped_in_the_updater() throws {
    mainApplicationDefaults.set("2230", forKey: "SUSkippedVersion")

    XCTAssertEqual(try XCTUnwrap(settings()).preferences.skippedBuildNumber, 2230)
  }

  func test_should_ignore_a_skipped_build_that_is_not_a_number() throws {
    mainApplicationDefaults.set("0.3.2-beta", forKey: "SUSkippedVersion")

    XCTAssertNil(try XCTUnwrap(settings()).preferences.skippedBuildNumber)
  }
}
