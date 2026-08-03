import Foundation
import XCTest

@testable import AstronomicalMenu

final class MenuStructureContractTests: XCTestCase {
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
}
