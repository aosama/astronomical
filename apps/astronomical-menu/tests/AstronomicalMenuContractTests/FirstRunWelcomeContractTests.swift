import Foundation
import XCTest

@testable import AstronomicalMenuCore

final class FirstRunWelcomeContractTests: XCTestCase {
  private var temporaryDirectoryURL: URL!

  override func setUpWithError() throws {
    try super.setUpWithError()
    temporaryDirectoryURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("first-run-welcome-contract-\(UUID().uuidString)", isDirectory: true)
  }

  override func tearDownWithError() throws {
    if FileManager.default.fileExists(atPath: temporaryDirectoryURL.path) {
      try FileManager.default.removeItem(at: temporaryDirectoryURL)
    }
    try super.tearDownWithError()
  }

  func test_fresh_install_without_state_directory_shows_the_welcome() {
    let decision = makeStore().welcomeDecision()
    XCTAssertEqual(decision, .showWelcome)
  }

  func test_preexisting_configuration_is_migrated_to_acknowledged_without_showing() throws {
    try FileManager.default.createDirectory(at: temporaryDirectoryURL, withIntermediateDirectories: true)
    try Data("{}".utf8).write(
      to: temporaryDirectoryURL.appendingPathComponent("config.json"))

    let decision = makeStore().welcomeDecision()

    XCTAssertEqual(decision, .alreadyAcknowledged)
    // The migration must persist, so a later evaluation never re-shows.
    XCTAssertEqual(makeStore().welcomeDecision(), .alreadyAcknowledged)
    XCTAssertTrue(
      FileManager.default.fileExists(
        atPath: temporaryDirectoryURL.appendingPathComponent(
          FirstRunWelcomeAcknowledgmentStore.markerFileName).path))
  }

  func test_acknowledged_marker_suppresses_the_welcome() {
    let store = makeStore()
    store.acknowledgeWelcome()
    XCTAssertEqual(store.welcomeDecision(), .alreadyAcknowledged)
  }

  func test_acknowledgment_survives_a_new_store_instance() {
    makeStore().acknowledgeWelcome()
    XCTAssertEqual(makeStore().welcomeDecision(), .alreadyAcknowledged)
  }

  func test_welcome_stays_reopenable_from_the_popover_overflow_menu() throws {
    let packageDirectoryURL = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let popoverSource = try String(
      contentsOf: packageDirectoryURL.appendingPathComponent(
        "Sources/AstronomicalMenuCore/OrbitalTelemetryPopover.swift"),
      encoding: .utf8)

    XCTAssertTrue(popoverSource.contains("Button(\"Welcome…\", action: showWelcome)"))
  }

  private func makeStore() -> FirstRunWelcomeAcknowledgmentStore {
    FirstRunWelcomeAcknowledgmentStore(stateDirectoryURL: temporaryDirectoryURL)
  }
}
