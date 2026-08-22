import Foundation
import XCTest

@testable import AstronomicalMenu

final class MenuStructureContractTests: XCTestCase {
  func test_should_expose_open_observatory_as_a_direct_popover_action() throws {
    let packageDirectoryURL = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
    let popoverSourceURL =
      packageDirectoryURL
      .appendingPathComponent("Sources/AstronomicalMenu/OrbitalTelemetryPopover.swift")
    let popoverSource = try String(contentsOf: popoverSourceURL, encoding: .utf8)

    let openObservatoryButtonStart = try XCTUnwrap(
      popoverSource.range(
        of: "Button(action: telemetryStore.hasDiscoveredModels ? openObservatory : openLibrary)"))
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
      .appendingPathComponent("Sources/AstronomicalMenu/OrbitalTelemetryPopover.swift")
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
        "Sources/AstronomicalMenu/OrbitalTelemetryPopover.swift"),
      encoding: .utf8)
    let downloadActionStart = try XCTUnwrap(popoverSource.range(of: "\"Download a model\""))
    let overflowMenuStart = try XCTUnwrap(popoverSource.range(of: "Menu {"))

    XCTAssertLessThan(downloadActionStart.lowerBound, overflowMenuStart.lowerBound)
  }
}
