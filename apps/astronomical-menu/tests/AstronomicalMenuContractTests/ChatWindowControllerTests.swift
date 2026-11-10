import AppKit
import ThinTalkCanvas
import XCTest
import ThinTalkCore

@testable import AstronomicalAppCore

/// Journey contracts for the main app's conversation window: showing it puts a
/// real window on screen, closing only takes it off screen, and reopening
/// brings the same window back. The controller never touches the activation
/// policy or the menu bar; those are fixed for the life of the main app.
@MainActor
final class ChatWindowControllerTests: XCTestCase {
  private var controller: ChatWindowController!

  func test_should_present_a_visible_conversation_window() throws {
    try bootstrap()
    defer { cleanup() }

    controller.showChatWindow()

    let window = try XCTUnwrap(chatWindow(), "showing chat must present the chat window")
    XCTAssertTrue(window.isVisible, "the chat window must actually be on screen")
    XCTAssertEqual(window.title, "Thin Talk Development")
  }

  /// "Your Astro…" opens the same chat window; the page then opens Your Astro
  /// once it has loaded.
  func test_should_present_the_chat_window_for_your_astro() throws {
    try bootstrap()
    defer { cleanup() }

    controller.showYourAstro()

    XCTAssertTrue(try XCTUnwrap(chatWindow(), "Your Astro… must present the chat window").isVisible)
  }

  /// Without a zoom controller on its canvas every zoom key silently does nothing.
  func test_should_answer_the_zoom_keys_in_the_chat_window() throws {
    try bootstrap()
    defer { cleanup() }

    controller.showChatWindow()

    let window = try XCTUnwrap(chatWindow())
    window.layoutIfNeeded()
    let canvas = try XCTUnwrap(
      findCanvas(in: try XCTUnwrap(window.contentView)), "the chat window must host a canvas")
    for zoomKey in ["=", "+", "-", "0"] {
      XCTAssertTrue(canvas.performKeyEquivalent(with: try commandKeyEvent(zoomKey)), zoomKey)
    }
    XCTAssertFalse(canvas.performKeyEquivalent(with: try commandKeyEvent("j")))
  }

  func test_should_frame_the_chat_window_like_a_native_mac_window() throws {
    try bootstrap()
    defer { cleanup() }

    controller.showChatWindow()

    let window = try XCTUnwrap(chatWindow())
    XCTAssertTrue(window.styleMask.contains(.fullSizeContentView))
    XCTAssertEqual(window.titleVisibility, .hidden)
    XCTAssertNotNil(window.toolbar)
    XCTAssertEqual(window.tabbingMode, .disallowed)
  }

  func test_should_report_chat_showing_only_while_the_window_is_on_screen() throws {
    try bootstrap()
    defer { cleanup() }
    XCTAssertFalse(controller.isShowingChat, "a window that was never shown is not showing")

    controller.showChatWindow()
    XCTAssertTrue(controller.isShowingChat)

    try XCTUnwrap(chatWindow()).close()
    XCTAssertFalse(controller.isShowingChat)
  }

  /// A reply still arriving must survive a close, so the window is kept and
  /// shown again rather than rebuilt.
  func test_should_bring_back_the_same_window_after_a_close() throws {
    try bootstrap()
    defer { cleanup() }
    controller.showChatWindow()
    let firstWindow = try XCTUnwrap(chatWindow())
    firstWindow.close()

    controller.showChatWindow()

    let reopenedWindow = try XCTUnwrap(chatWindow())
    XCTAssertTrue(reopenedWindow === firstWindow, "reopening must reuse the one conversation window")
    XCTAssertTrue(reopenedWindow.isVisible)
  }

  /// The menu bar's commands and the toolbar's must be the same object as the
  /// window's, or a menu item would act on a window that is not this one.
  func test_should_offer_the_actions_of_its_own_window_for_the_menu_bar() throws {
    try bootstrap()
    defer { cleanup() }
    controller.showChatWindow()

    let newChatItem = NSMenuItem(title: "New Chat", action: nil, keyEquivalent: "")

    XCTAssertTrue(controller.conversationActions.validateMenuItem(newChatItem))
    try XCTUnwrap(chatWindow()).close()
    XCTAssertFalse(
      controller.conversationActions.validateMenuItem(newChatItem),
      "conversation commands are disabled while the window is closed")
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
  }

  private func cleanup() {
    chatWindow()?.close()
    controller = nil
  }

  // Windows created with isReleasedWhenClosed=false stay in NSApp.windows
  // after close, so earlier tests leave dead windows behind; the window this
  // test just presented is always the most recently created match.
  private func chatWindow() -> NSWindow? {
    NSApp.windows.last { $0.title == "Thin Talk Development" }
  }
}
