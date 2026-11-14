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

  func test_should_never_install_the_public_command_on_the_development_channel() {
    let action = CliLinkPlan.decide(
      status: status(exists: false),
      channel: .development,
      bundleCliPath: bundleCliPath)
    XCTAssertEqual(action, .notSupportedForChannel)
  }

  func test_should_give_only_the_stable_channel_the_public_command_name() {
    XCTAssertEqual(CliLinkPlan.linkName(for: .stable), "astronomical")
    XCTAssertNil(CliLinkPlan.linkName(for: .appStore))
    XCTAssertNil(CliLinkPlan.linkName(for: .development))
  }

  /// The store bundle carries no terminal command, so the store build must
  /// neither install a link nor repair one a direct install left behind,
  /// even when that link dangles into an Astronomical bundle.
  func test_should_never_touch_the_public_command_on_the_app_store_channel() {
    let missingLinkAction = CliLinkPlan.decide(
      status: status(exists: false),
      channel: .appStore,
      bundleCliPath: bundleCliPath)
    let danglingOwnedLinkAction = CliLinkPlan.decide(
      status: status(
        exists: true, isSymlink: true,
        destination: "/Applications/Astronomical.app/Contents/MacOS/astronomical",
        destinationExists: false),
      channel: .appStore,
      bundleCliPath: bundleCliPath)

    XCTAssertEqual(missingLinkAction, .notSupportedForChannel)
    XCTAssertEqual(danglingOwnedLinkAction, .notSupportedForChannel)
  }

  func test_should_plan_a_fresh_install_for_a_missing_destination() {
    let action = CliLinkPlan.decide(
      status: status(exists: false),
      channel: .stable,
      bundleCliPath: bundleCliPath)
    XCTAssertEqual(action, .installMissing)
  }

  func test_should_leave_a_link_already_pointing_at_this_bundle() {
    let action = CliLinkPlan.decide(
      status: status(exists: true, isSymlink: true, destination: bundleCliPath, destinationExists: true),
      channel: .stable,
      bundleCliPath: bundleCliPath)
    XCTAssertEqual(action, .alreadyCurrent)
  }

  func test_should_repair_a_dangling_link_into_a_stale_bundle() {
    let staleBundlePath =
      "/Applications/Astronomical 2.app/Contents/MacOS/astronomical"
    let action = CliLinkPlan.decide(
      status: status(
        exists: true, isSymlink: true, destination: staleBundlePath, destinationExists: false),
      channel: .stable,
      bundleCliPath: bundleCliPath)
    XCTAssertEqual(action, .repairOwnedDanglingLink)
  }

  func test_should_repair_a_live_link_into_a_different_bundle() {
    let movedBundlePath = "/Users/someone/Applications/Astronomical.app/Contents/MacOS/astronomical"
    let action = CliLinkPlan.decide(
      status: status(
        exists: true, isSymlink: true, destination: movedBundlePath, destinationExists: true),
      channel: .stable,
      bundleCliPath: bundleCliPath)
    XCTAssertEqual(action, .repairOwnedDanglingLink)
  }

  func test_should_refuse_a_regular_file_at_the_destination() {
    let action = CliLinkPlan.decide(
      status: status(exists: true, isSymlink: false),
      channel: .stable,
      bundleCliPath: bundleCliPath)
    XCTAssertEqual(action, .refuseForeignDestination)
  }

  func test_should_refuse_a_symlink_to_an_unrelated_binary() {
    let action = CliLinkPlan.decide(
      status: status(
        exists: true, isSymlink: true, destination: "/usr/local/bin/other-tool",
        destinationExists: true),
      channel: .stable,
      bundleCliPath: bundleCliPath)
    XCTAssertEqual(action, .refuseForeignDestination)
  }

  func test_should_refuse_a_dangling_symlink_outside_owned_bundles() {
    let action = CliLinkPlan.decide(
      status: status(
        exists: true, isSymlink: true, destination: "/Volumes/USB/Tool.app/Contents/MacOS/astronomical",
        destinationExists: false),
      channel: .stable,
      bundleCliPath: bundleCliPath)
    XCTAssertEqual(action, .refuseForeignDestination)
  }
}
