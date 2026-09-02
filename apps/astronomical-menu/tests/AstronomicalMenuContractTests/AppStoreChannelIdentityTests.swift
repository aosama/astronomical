// Pins the App Store channel identity: Stable product presentation, Stable
// runtime instance, and Application Support state roots that the sandbox maps
// into the app container.

import Foundation
import XCTest

@testable import AstronomicalMenuCore

@MainActor
final class AppStoreChannelIdentityTests: XCTestCase {
  private func appStoreIdentity() -> ApplicationIdentity {
    ApplicationIdentity(
      channel: .appStore,
      supervisorPort: 6732,
      stateDirectoryName: nil,
      version: "0.3.0",
      buildNumber: "500",
      commit: "fixturecommit",
      isDirty: false
    )
  }

  func test_should_present_the_stable_product_identity_for_app_store_builds() {
    XCTAssertEqual(appStoreIdentity().channel.displayName, "Stable")
  }

  func test_should_run_the_stable_runtime_instance_on_app_store_builds() {
    XCTAssertEqual(appStoreIdentity().daemonArguments, ["--instance", "stable"])
  }

  func test_should_derive_app_store_state_from_the_application_support_directory() {
    let fictionalHomeDirectory = URL(fileURLWithPath: "/Users/example", isDirectory: true)

    let stateDirectoryURL = appStoreIdentity().stateDirectoryURL(
      homeDirectoryURL: fictionalHomeDirectory)

    XCTAssertEqual(
      stateDirectoryURL,
      fictionalHomeDirectory
        .appendingPathComponent("Library/Application Support", isDirectory: true)
        .appendingPathComponent("Astronomical", isDirectory: true)
    )
    XCTAssertEqual(
      appStoreIdentity().configFileURL(homeDirectoryURL: fictionalHomeDirectory),
      stateDirectoryURL.appendingPathComponent("config.json")
    )
    XCTAssertEqual(
      appStoreIdentity().daemonOwnershipURL(homeDirectoryURL: fictionalHomeDirectory),
      stateDirectoryURL.appendingPathComponent("menu-owned-daemon.json")
    )
  }

  func test_should_keep_direct_channels_on_their_home_dot_folders() {
    let fictionalHomeDirectory = URL(fileURLWithPath: "/Users/example", isDirectory: true)

    XCTAssertEqual(ApplicationChannel.stable.stateDirectoryName, ".astronomical")
    XCTAssertEqual(ApplicationChannel.development.stateDirectoryName, ".astronomical-dev")
    XCTAssertNil(ApplicationChannel.appStore.stateDirectoryName)
    XCTAssertEqual(
      ApplicationChannel.stable.stateDirectoryName,
      fictionalHomeDirectory
        .appendingPathComponent(".astronomical", isDirectory: true).lastPathComponent
    )
  }

  func test_should_resolve_the_app_store_channel_from_its_bundle_value() {
    XCTAssertEqual(ApplicationChannel(rawValue: "app-store"), .appStore)
  }
}