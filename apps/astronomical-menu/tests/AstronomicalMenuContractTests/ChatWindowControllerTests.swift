import AppKit
import ThinTalkCanvas
import XCTest
import ThinTalkCore

@testable import AstronomicalMenuCore

/// Journey contracts for the Chat menu item's window: opening chat presents a
/// visible conversation window, closing it restores the accessory posture
/// through the application handler exactly once, and reopening reuses the
/// same persistent window. These follow the first-run welcome window contracts.
@MainActor
final class ChatWindowControllerTests: XCTestCase {
  private var controller: ChatWindowController!
  private var closeCount = 0

  func test_show_chat_presents_a_visible_conversation_window() throws {
    try bootstrap()
    defer { cleanup() }

    showChat()

    let window = try XCTUnwrap(chatWindow(), "the Chat item must present the chat window")
    XCTAssertTrue(window.isVisible, "the chat window must actually be on screen")
    XCTAssertEqual(window.title, "Thin Talk Development")
  }

  /// The hosted window has no View menu, so without a zoom controller on its
  /// canvas every zoom key silently does nothing.
  func test_should_answer_the_zoom_keys_in_the_hosted_chat_window() throws {
    try bootstrap()
    defer { cleanup() }

    showChat()

    let window = try XCTUnwrap(chatWindow())
    window.layoutIfNeeded()
    let canvas = try XCTUnwrap(
      findCanvas(in: try XCTUnwrap(window.contentView)), "the chat window must host a canvas")
    for zoomKey in ["=", "+", "-", "0"] {
      XCTAssertTrue(canvas.performKeyEquivalent(with: try commandKeyEvent(zoomKey)), zoomKey)
    }
    XCTAssertFalse(canvas.performKeyEquivalent(with: try commandKeyEvent("j")))
  }

  private func findCanvas(in view: NSView) -> ConversationCanvasView? {
    if let canvas = view as? ConversationCanvasView { return canvas }
    for childView in view.subviews {
      if let canvas = findCanvas(in: childView) { return canvas }
    }
    return nil
  }

  private func commandKeyEvent(_ keyCharacter: String) throws -> NSEvent {
    try XCTUnwrap(
      NSEvent.keyEvent(
        with: .keyDown, location: .zero, modifierFlags: [.command], timestamp: 0, windowNumber: 0,
        context: nil, characters: keyCharacter, charactersIgnoringModifiers: keyCharacter,
        isARepeat: false, keyCode: 0))
  }

  func test_closing_the_chat_window_invokes_the_close_handler_exactly_once() throws {
    try bootstrap()
    defer { cleanup() }

    showChat()
    let window = try XCTUnwrap(chatWindow())

    window.close()

    XCTAssertEqual(closeCount, 1, "closing the chat window must notify the app exactly once")
    XCTAssertFalse(window.isVisible)
  }

  func test_reopening_after_close_presents_the_chat_window_again() throws {
    try bootstrap()
    defer { cleanup() }

    showChat()
    try XCTUnwrap(chatWindow()).close()

    showChat()
    let reopenedWindow = try XCTUnwrap(
      chatWindow(), "the Chat item must reopen the window after a close")
    XCTAssertTrue(reopenedWindow.isVisible)

    reopenedWindow.close()
    XCTAssertEqual(closeCount, 2, "each presented chat window closes on its own")
  }

  private func bootstrap() throws {
    _ = NSApplication.shared
    // The identity points at an isolated port, so the client behind the
    // window never touches a real daemon during these journeys. Port 0
    // answers immediately with a connection-refused failure the surface
    // classifies; the window presentation itself does not depend on the reply.
    controller = ChatWindowController(thinTalkIdentity: ThinTalkApplicationIdentity(
      channel: .development,
      daemonPort: 0,
      stateDirectoryName: ".astronomical-tests",
      version: "test",
      buildNumber: "0",
      commit: "test",
      isDirty: false))
    closeCount = 0
  }

  private func cleanup() {
    chatWindow()?.close()
    controller = nil
  }

  private func showChat() {
    controller.showChatWindow(handleClose: { [weak self] in self?.closeCount += 1 })
  }

  // Windows created with isReleasedWhenClosed=false stay in NSApp.windows
  // after close, so earlier tests leave dead windows behind; the window this
  // test just presented is always the most recently created match.
  private func chatWindow() -> NSWindow? {
    NSApp.windows.last { $0.title == "Thin Talk Development" }
  }
}
