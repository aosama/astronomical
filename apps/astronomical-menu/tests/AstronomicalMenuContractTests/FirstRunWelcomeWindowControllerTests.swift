import AppKit
import XCTest

@testable import AstronomicalMenuCore

/// Journey contracts for the first-run welcome window presentation itself.
///
/// The acknowledgment store tests cover the decision state machine; these
/// tests cover the wiring a new user actually experiences: a fresh install
/// presents a visible welcome window, closing it by any path acknowledges
/// exactly once, and the window can be reopened from the menu afterwards.
/// These are the first AppKit-level contracts in this package, so they
/// bootstrap a shared test application instead of assuming one exists.
/// Setup and cleanup live in MainActor-isolated helpers invoked from each
/// test body because the XCTest setUp/tearDown overrides are nonisolated.
@MainActor
final class FirstRunWelcomeWindowControllerTests: XCTestCase {
  private var temporaryDirectoryURL: URL!
  private var controller: FirstRunWelcomeWindowController!
  private var closeCount = 0

  func test_show_presents_a_visible_welcome_window() throws {
    try bootstrap()
    defer { cleanup() }

    showWelcome()

    let window = try XCTUnwrap(welcomeWindow(), "show must present the welcome window")
    XCTAssertTrue(window.isVisible, "the welcome window must actually be on screen")
  }

  func test_closing_the_welcome_window_invokes_the_close_handler_exactly_once() throws {
    try bootstrap()
    defer { cleanup() }

    showWelcome()
    let window = try XCTUnwrap(welcomeWindow())

    window.close()

    XCTAssertEqual(closeCount, 1, "closing by any path must acknowledge exactly once")
    XCTAssertFalse(window.isVisible)
  }

  func test_reopening_after_close_presents_the_welcome_again_and_acknowledges_again() throws {
    try bootstrap()
    defer { cleanup() }

    showWelcome()
    try XCTUnwrap(welcomeWindow()).close()
    XCTAssertEqual(closeCount, 1)

    showWelcome()
    let reopenedWindow = try XCTUnwrap(
      welcomeWindow(), "the menu Welcome item must reopen the window after a close")
    XCTAssertTrue(reopenedWindow.isVisible)

    reopenedWindow.close()
    XCTAssertEqual(closeCount, 2, "each presented welcome acknowledges on its own close")
  }

  func test_close_acknowledgment_keeps_a_subsequent_launch_silent() throws {
    try bootstrap()
    defer { cleanup() }

    let store = FirstRunWelcomeAcknowledgmentStore(stateDirectoryURL: temporaryDirectoryURL)
    XCTAssertEqual(
      store.welcomeDecision(), .showWelcome,
      "a fresh state directory must ask for the welcome")

    // Wire the close path the way the application delegate does: closing the
    // window acknowledges through the same store the next launch will read.
    controller.showWelcomeWindow(
      telemetryStore: TelemetryStore(),
      applicationIdentity: testIdentity,
      openObservatory: {},
      openLibrary: {},
      restartServer: {},
      handleClose: { store.acknowledgeWelcome() })
    try XCTUnwrap(welcomeWindow()).close()

    XCTAssertEqual(
      FirstRunWelcomeAcknowledgmentStore(stateDirectoryURL: temporaryDirectoryURL)
        .welcomeDecision(),
      .alreadyAcknowledged,
      "after acknowledging, the next launch must stay silent")
  }

  private func bootstrap() throws {
    _ = NSApplication.shared
    temporaryDirectoryURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("first-run-welcome-window-\(UUID().uuidString)", isDirectory: true)
    controller = FirstRunWelcomeWindowController()
    closeCount = 0
  }

  private func cleanup() {
    welcomeWindow()?.close()
    controller = nil
    try? FileManager.default.removeItem(at: temporaryDirectoryURL)
  }

  private var testIdentity: ApplicationIdentity {
    ApplicationIdentity(
      channel: .development,
      supervisorPort: 0,
      stateDirectoryName: nil,
      version: "test",
      buildNumber: "0",
      commit: "test",
      isDirty: false)
  }

  private func showWelcome() {
    controller.showWelcomeWindow(
      telemetryStore: TelemetryStore(),
      applicationIdentity: testIdentity,
      openObservatory: {},
      openLibrary: {},
      restartServer: {},
      handleClose: { [weak self] in self?.closeCount += 1 })
  }

  // Windows created with isReleasedWhenClosed=false stay in NSApp.windows
  // after close, so earlier tests leave dead windows behind; the window this
  // test just presented is always the most recently created match.
  private func welcomeWindow() -> NSWindow? {
    NSApp.windows.last { $0.title == "Welcome to Astronomical" }
  }
}
