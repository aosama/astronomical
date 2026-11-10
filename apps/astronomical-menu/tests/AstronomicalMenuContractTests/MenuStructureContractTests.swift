import Foundation
import XCTest

@testable import AstronomicalMenuCore

final class MenuStructureContractTests: XCTestCase {
  func test_should_expose_open_observatory_as_a_direct_popover_action() throws {
    let packageDirectoryURL = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
    let popoverSourceURL =
      packageDirectoryURL
      .appendingPathComponent("Sources/AstronomicalMenuCore/OrbitalTelemetryPopover.swift")
    let popoverSource = try String(contentsOf: popoverSourceURL, encoding: .utf8)

    let openObservatoryButtonStart = try XCTUnwrap(
      popoverSource.range(
        of: "Button(action: telemetryStore.hasDiscoveredModels ? openObservatory : openDownloadModels)"))
    let overflowMenuStart = try XCTUnwrap(popoverSource.range(of: "Menu {"))

    XCTAssertLessThan(openObservatoryButtonStart.lowerBound, overflowMenuStart.lowerBound)
  }

  func test_should_never_disable_the_restart_server_button() throws {
    let packageDirectoryURL = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
    let popoverSourceURL =
      packageDirectoryURL
      .appendingPathComponent("Sources/AstronomicalMenuCore/OrbitalTelemetryPopover.swift")
    let popoverSource = try String(contentsOf: popoverSourceURL, encoding: .utf8)
    let restartButtonStart = try XCTUnwrap(
      popoverSource.range(of: "Button(\"Restart server\", action: restartServer)"))
    let trailingActionsStart = try XCTUnwrap(
      popoverSource.range(
        of: "Spacer()", range: restartButtonStart.upperBound..<popoverSource.endIndex))
    let restartButtonSource = popoverSource[
      restartButtonStart.lowerBound..<trailingActionsStart.lowerBound]

    XCTAssertFalse(restartButtonSource.contains(".disabled("))
  }

  func test_should_expose_download_a_model_as_a_direct_popover_action() throws {
    let packageDirectoryURL = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let popoverSource = try String(
      contentsOf: packageDirectoryURL.appendingPathComponent(
        "Sources/AstronomicalMenuCore/OrbitalTelemetryPopover.swift"),
      encoding: .utf8)
    let downloadActionStart = try XCTUnwrap(popoverSource.range(of: "\"Download a model\""))
    let overflowMenuStart = try XCTUnwrap(popoverSource.range(of: "Menu {"))

    XCTAssertLessThan(downloadActionStart.lowerBound, overflowMenuStart.lowerBound)
  }

  /// The main app owns the updater and registers the agent at login, so their
  /// settings stay in its menu. The popover keeps the one-shot update check
  /// people reach for in the menu bar, which it hands to the main app.
  func test_should_keep_the_update_check_in_the_popover_and_leave_settings_to_the_main_app() throws {
    let packageDirectoryURL = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let popoverSource = try String(
      contentsOf: packageDirectoryURL.appendingPathComponent(
        "Sources/AstronomicalMenuCore/OrbitalTelemetryPopover.swift"),
      encoding: .utf8)
    let settingsMenuSource = try String(
      contentsOf: packageDirectoryURL.appendingPathComponent(
        "Sources/AstronomicalAppCore/ApplicationSettingsMenu.swift"),
      encoding: .utf8)

    for mainApplicationSetting in ["Launch at Login", "Check automatically", "Update Channel"] {
      XCTAssertFalse(popoverSource.contains("\"\(mainApplicationSetting)"), mainApplicationSetting)
    }
    XCTAssertTrue(popoverSource.contains("\"Check for Updates…\""))
    XCTAssertTrue(settingsMenuSource.contains("\"Launch at Login\""))
    XCTAssertTrue(settingsMenuSource.contains("\"Update Channel\""))
  }
}
