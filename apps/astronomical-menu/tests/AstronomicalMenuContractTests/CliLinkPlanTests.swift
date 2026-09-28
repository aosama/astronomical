import XCTest

@testable import AstronomicalMenuCore

final class CliLinkPlanTests: XCTestCase {
  private let bundleCliPath =
    "/Applications/Astronomical.app/Contents/MacOS/astronomical"

  private func status(
    exists: Bool,
    isSymlink: Bool = false,
    destination: String? = nil,
    destinationExists: Bool = false
  ) -> CliLinkStatus {
    CliLinkStatus(
      exists: exists,
      isSymlink: isSymlink,
      symlinkDestinationPath: destination,
      symlinkDestinationExists: destinationExists)
  }

  func test_development_channel_never_installs_the_public_command() {
    let action = CliLinkPlan.decide(
      status: status(exists: false),
      channel: .development,
      bundleCliPath: bundleCliPath)
    XCTAssertEqual(action, .notSupportedForChannel)
  }

  func test_stable_and_app_store_channels_own_the_public_command_name() {
    XCTAssertEqual(CliLinkPlan.linkName(for: .stable), "astronomical")
    XCTAssertEqual(CliLinkPlan.linkName(for: .appStore), "astronomical")
    XCTAssertNil(CliLinkPlan.linkName(for: .development))
  }

  func test_missing_destination_is_a_fresh_install() {
    let action = CliLinkPlan.decide(
      status: status(exists: false),
      channel: .stable,
      bundleCliPath: bundleCliPath)
    XCTAssertEqual(action, .installMissing)
  }

  func test_link_already_pointing_at_this_bundle_is_current() {
    let action = CliLinkPlan.decide(
      status: status(exists: true, isSymlink: true, destination: bundleCliPath, destinationExists: true),
      channel: .stable,
      bundleCliPath: bundleCliPath)
    XCTAssertEqual(action, .alreadyCurrent)
  }

  func test_dangling_link_into_a_stale_bundle_is_repaired() {
    let staleBundlePath =
      "/Applications/Astronomical 2.app/Contents/MacOS/astronomical"
    let action = CliLinkPlan.decide(
      status: status(
        exists: true, isSymlink: true, destination: staleBundlePath, destinationExists: false),
      channel: .stable,
      bundleCliPath: bundleCliPath)
    XCTAssertEqual(action, .repairOwnedDanglingLink)
  }

  func test_live_link_into_a_different_bundle_is_repaired() {
    let movedBundlePath = "/Users/someone/Applications/Astronomical.app/Contents/MacOS/astronomical"
    let action = CliLinkPlan.decide(
      status: status(
        exists: true, isSymlink: true, destination: movedBundlePath, destinationExists: true),
      channel: .stable,
      bundleCliPath: bundleCliPath)
    XCTAssertEqual(action, .repairOwnedDanglingLink)
  }

  func test_regular_file_at_destination_is_refused() {
    let action = CliLinkPlan.decide(
      status: status(exists: true, isSymlink: false),
      channel: .stable,
      bundleCliPath: bundleCliPath)
    XCTAssertEqual(action, .refuseForeignDestination)
  }

  func test_symlink_to_unrelated_binary_is_refused() {
    let action = CliLinkPlan.decide(
      status: status(
        exists: true, isSymlink: true, destination: "/usr/local/bin/other-tool",
        destinationExists: true),
      channel: .stable,
      bundleCliPath: bundleCliPath)
    XCTAssertEqual(action, .refuseForeignDestination)
  }

  func test_dangling_symlink_outside_owned_bundles_is_refused() {
    let action = CliLinkPlan.decide(
      status: status(
        exists: true, isSymlink: true, destination: "/Volumes/USB/Tool.app/Contents/MacOS/astronomical",
        destinationExists: false),
      channel: .stable,
      bundleCliPath: bundleCliPath)
    XCTAssertEqual(action, .refuseForeignDestination)
  }
}
