import XCTest

@testable import AstronomicalShared

/// Both processes resolve the selected update channel through these.
final class UpdateChannelSelectionTests: XCTestCase {
  private static let userDefaultsSuiteName = "UpdateChannelSelectionTests"
  private var userDefaults: UserDefaults!

  override func setUpWithError() throws {
    userDefaults = try XCTUnwrap(UserDefaults(suiteName: Self.userDefaultsSuiteName))
    userDefaults.removePersistentDomain(forName: Self.userDefaultsSuiteName)
  }

  override func tearDown() {
    userDefaults.removePersistentDomain(forName: Self.userDefaultsSuiteName)
  }

  func test_should_default_to_the_channel_the_build_belongs_to() {
    XCTAssertEqual(UpdateChannelSelection.storedChannel(in: userDefaults, applicationChannel: .stable), .stable)
    XCTAssertEqual(
      UpdateChannelSelection.storedChannel(in: userDefaults, applicationChannel: .development), .development)
  }

  func test_should_return_the_stored_choice() {
    userDefaults.set("release-candidate", forKey: UpdateChannelSelection.userDefaultsKey)

    XCTAssertEqual(
      UpdateChannelSelection.storedChannel(in: userDefaults, applicationChannel: .stable), .releaseCandidate)
  }

  func test_should_fall_back_to_the_default_for_a_stored_value_it_does_not_know() {
    userDefaults.set("nightly", forKey: UpdateChannelSelection.userDefaultsKey)

    XCTAssertEqual(UpdateChannelSelection.storedChannel(in: userDefaults, applicationChannel: .stable), .stable)
  }
}
