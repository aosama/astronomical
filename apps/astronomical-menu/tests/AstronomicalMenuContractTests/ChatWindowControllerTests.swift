import AppKit
import ThinTalkCanvas
import XCTest
import ThinTalkCore

@testable import AstronomicalMenuCore

/// Journey contracts for the Chat menu item's window: opening chat presents a
/// visible conversation window, closing it restores the accessory posture
/// through the application handler exactly once, and reopening reuses the
/// same persistent window.
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

  /// The popover's "Your Astro…" opens the same chat window (the page then
  /// opens Your Astro once it has loaded), with the conversation's menus.
  func test_show_your_astro_presents_the_chat_window_with_its_menus() throws {
    try bootstrap()
    defer { cleanup() }

    controller.showYourAstro(handleClose: { [weak self] in self?.closeCount += 1 })

    let window = try XCTUnwrap(chatWindow(), "Your Astro… must present the chat window")
    XCTAssertTrue(window.isVisible)
    XCTAssertNotNil(NSApp.mainMenu?.item(withTitle: "Conversation"), "the conversation's menus are installed")
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

  /// While chat is open the app is a regular app with a menu bar; without the
  /// conversation's menus the bar held only the Apple menu, and ⌘V had no Edit
  /// menu to reach the message field through.
  func test_should_install_the_conversation_menu_bar_when_chat_opens() throws {
    try bootstrap()
    defer { cleanup() }

    showChat()

    let mainMenu = try XCTUnwrap(NSApp.mainMenu)
    XCTAssertEqual(
      mainMenu.items.map(\.title),
      ["Astronomical", "File", "Edit", "View", "Conversation", "Window", "Help"])
    let pasteItem = try XCTUnwrap(mainMenu.item(withTitle: "Edit")?.submenu?.item(withTitle: "Paste"))
    XCTAssertEqual(pasteItem.keyEquivalent, "v")
  }

  /// The menu bar outlives the chat window, so closing chat puts back the
  /// menu from before; another window must never show the conversation menus.
  func test_should_put_the_previous_menu_bar_back_when_chat_closes() throws {
    try bootstrap()
    defer { cleanup() }
    let menuBeforeChat = NSMenu(title: "Before chat")
    NSApp.mainMenu = menuBeforeChat

    showChat()
    XCTAssertNotNil(NSApp.mainMenu?.item(withTitle: "Conversation"))
    try XCTUnwrap(chatWindow()).close()
    XCTAssertNotNil(
      NSApp.mainMenu?.item(withTitle: "Conversation"),
      "the swap waits until menu handling ends, so an open menu is never orphaned")
    finishPendingCloseWork()

    XCTAssertTrue(NSApp.mainMenu === menuBeforeChat)
  }

  /// Chat reopened before the deferred close work ran must keep its menus,
  /// and the superseded close must not send the app back to accessory.
  func test_should_keep_the_conversation_menus_when_chat_reopens_before_the_close_settles() throws {
    try bootstrap()
    defer { cleanup() }
    let menuBeforeChat = NSMenu(title: "Before chat")
    NSApp.mainMenu = menuBeforeChat

    showChat()
    try XCTUnwrap(chatWindow()).close()
    showChat()
    finishPendingCloseWork()

    XCTAssertNotNil(NSApp.mainMenu?.item(withTitle: "Conversation"))
    XCTAssertEqual(closeCount, 0, "the superseded close never reaches the app")
    try XCTUnwrap(chatWindow()).close()
    finishPendingCloseWork()
    XCTAssertTrue(NSApp.mainMenu === menuBeforeChat, "the menu from before chat is still the one put back")
    XCTAssertEqual(closeCount, 1)
  }

  func test_should_frame_the_chat_window_like_a_native_mac_window() throws {
    try bootstrap()
    defer { cleanup() }

    showChat()

    let window = try XCTUnwrap(chatWindow())
    XCTAssertTrue(window.styleMask.contains(.fullSizeContentView))
    XCTAssertEqual(window.titleVisibility, .hidden)
    XCTAssertNotNil(window.toolbar)
    XCTAssertEqual(window.tabbingMode, .disallowed)
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

  /// Two closes before the deferred work runs must still put back the menu
  /// from before chat, not an empty menu bar.
  func test_should_put_the_previous_menu_back_once_after_two_quick_closes() throws {
    try bootstrap()
    defer { cleanup() }
    let menuBeforeChat = NSMenu(title: "Before chat")
    NSApp.mainMenu = menuBeforeChat

    showChat()
    try XCTUnwrap(chatWindow()).close()
    showChat()
    try XCTUnwrap(chatWindow()).close()
    finishPendingCloseWork()

    XCTAssertTrue(NSApp.mainMenu === menuBeforeChat)
    XCTAssertEqual(closeCount, 1, "the app hears one close for the window that stayed closed")
  }

  /// The app keeps its menu bar only while it stays a regular app, so it asks
  /// this before going back to menu-bar-only.
  func test_should_report_chat_showing_only_while_the_window_is_on_screen() throws {
    try bootstrap()
    defer { cleanup() }
    XCTAssertFalse(controller.isShowingChat, "no window yet")

    showChat()
    XCTAssertTrue(controller.isShowingChat)
    try XCTUnwrap(chatWindow()).close()
    finishPendingCloseWork()

    XCTAssertFalse(controller.isShowingChat)
  }

  func test_closing_the_chat_window_invokes_the_close_handler_exactly_once() throws {
    try bootstrap()
    defer { cleanup() }

    showChat()
    let window = try XCTUnwrap(chatWindow())

    window.close()
    finishPendingCloseWork()

    XCTAssertEqual(closeCount, 1, "closing the chat window must notify the app exactly once")
    XCTAssertFalse(window.isVisible)
  }

  func test_reopening_after_close_presents_the_chat_window_again() throws {
    try bootstrap()
    defer { cleanup() }

    showChat()
    try XCTUnwrap(chatWindow()).close()
    finishPendingCloseWork()

    showChat()
    let reopenedWindow = try XCTUnwrap(
      chatWindow(), "the Chat item must reopen the window after a close")
    XCTAssertTrue(reopenedWindow.isVisible)

    reopenedWindow.close()
    finishPendingCloseWork()
    XCTAssertEqual(closeCount, 2, "each presented chat window closes on its own")
  }

  /// Runs the close work the controller defers to the default run-loop mode,
  /// as the app's run loop does once menu handling ends.
  private func finishPendingCloseWork() {
    RunLoop.main.run(mode: .default, before: Date().addingTimeInterval(0.05))
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
    // A close's deferred work must not leak into the next test.
    finishPendingCloseWork()
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
