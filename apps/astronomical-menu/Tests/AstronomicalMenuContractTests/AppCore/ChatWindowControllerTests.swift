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
    XCTAssertEqual(window.title, "Astronomical Development")
  }

  /// A rebuild in the middle of other work must not take the screen: the
  /// window is on screen, so it can be looked at, but it is not the key window.
  func test_should_show_the_window_at_launch_without_taking_the_keyboard_when_launched_behind() throws {
    try bootstrap()
    defer { cleanup() }

    controller.showChatWindowAtLaunch(.behindOtherWindows)

    let window = try XCTUnwrap(chatWindow())
    XCTAssertTrue(window.isVisible)
    XCTAssertFalse(window.isKeyWindow, "a background launch must not take the keyboard")
  }

  /// Someone who clicks the Dock icon during the invisible first draw asked
  /// for the window: it must become visible and stay in front.
  func test_should_show_the_window_fully_when_asked_for_during_a_background_launch() throws {
    try bootstrap()
    defer { cleanup() }

    controller.showChatWindowAtLaunch(.behindOtherWindows)
    let window = try XCTUnwrap(chatWindow())
    XCTAssertEqual(window.alphaValue, 0, "invisible while the page draws its first frame")
    XCTAssertTrue(window.ignoresMouseEvents)
    controller.showChatWindow()

    XCTAssertEqual(window.alphaValue, 1)
    XCTAssertFalse(window.ignoresMouseEvents)
  }

  /// A rebuild that finishes on another Space or with the display asleep has
  /// no screen to draw on; only a window macOS reports as showing may start
  /// the wait for its first frame.
  func test_should_wait_for_a_visible_screen_before_counting_the_first_draw() {
    XCTAssertTrue(ChatWindowController.isOnAVisibleScreen([.visible]))
    XCTAssertFalse(ChatWindowController.isOnAVisibleScreen([]))
  }

  func test_should_read_the_launch_placement_from_the_launch_arguments() {
    XCTAssertEqual(ChatLaunchPlacement.fromLaunchArguments(["/app/astronomical-app"]), .inFront)
    XCTAssertEqual(
      ChatLaunchPlacement.fromLaunchArguments(["/app/astronomical-app", "--launch-in-background"]), .behindOtherWindows)
    XCTAssertEqual(ChatLaunchPlacement.fromLaunchArguments([]), .inFront)
    XCTAssertEqual(
      ChatLaunchPlacement.fromLaunchArguments(["--launch-in-background"]), .inFront,
      "the first argument is the program's own name, never an option")
    XCTAssertEqual(ChatLaunchPlacement.fromLaunchArguments(["/app/astronomical-app", "--background"]), .inFront)
  }

  /// "Settings…" opens the same chat window; the page then opens Settings
  /// once it has loaded.
  func test_should_present_the_chat_window_for_settings() throws {
    try bootstrap()
    defer { cleanup() }

    controller.showSettings()

    XCTAssertTrue(try XCTUnwrap(chatWindow(), "Settings… must present the chat window").isVisible)
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
    NSApp.windows.last { $0.title == "Astronomical Development" }
  }
}
