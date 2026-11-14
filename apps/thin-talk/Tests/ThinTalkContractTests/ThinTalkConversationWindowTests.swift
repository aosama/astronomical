import AppKit
import ThinTalkCore
import XCTest

@testable import ThinTalkUI

/// The conversation window's frame: content under a transparent, title-less
/// title bar, a unified toolbar with the sidebar toggle and New Chat, and no
/// window tabs. These are what make it read as a native Mac window rather than
/// a web page in a box.
@MainActor
final class ThinTalkConversationWindowTests: XCTestCase {
  private func conversationWindow() -> ThinTalkConversationWindow {
    _ = NSApplication.shared
    return ThinTalkConversationWindow(
      applicationIdentity: ThinTalkApplicationIdentity(channel: .development, daemonPort: 0))
  }

  func test_should_run_the_content_under_a_transparent_title_less_title_bar() {
    let window = conversationWindow().window
    XCTAssertTrue(window.styleMask.contains(.fullSizeContentView))
    XCTAssertTrue(window.titlebarAppearsTransparent)
    XCTAssertEqual(window.titleVisibility, .hidden)
    XCTAssertEqual(window.title, "Astronomical Development", "the title still names the window for the Window menu")
  }

  func test_should_frame_the_window_with_a_unified_toolbar_and_no_tabs() throws {
    let window = conversationWindow().window
    let toolbar = try XCTUnwrap(window.toolbar)
    XCTAssertEqual(window.toolbarStyle, .unified)
    XCTAssertEqual(
      toolbar.items.map(\.itemIdentifier), [.thinTalkToggleSidebar, .thinTalkNewChat, .flexibleSpace, .thinTalkExport])
    XCTAssertFalse(toolbar.allowsUserCustomization)
    XCTAssertEqual(window.tabbingMode, .disallowed)
  }

  func test_should_aim_the_toolbar_buttons_at_the_conversation_actions() throws {
    let framedWindow = conversationWindow()
    let toolbar = try XCTUnwrap(framedWindow.window.toolbar)
    for toolbarItem in toolbar.items where toolbarItem.itemIdentifier != .flexibleSpace {
      XCTAssertTrue(toolbarItem.target === framedWindow.conversationActions, toolbarItem.label)
      XCTAssertNotNil(toolbarItem.action, toolbarItem.label)
      XCTAssertNotNil(toolbarItem.image, toolbarItem.label)
    }
  }

  func test_should_export_from_the_far_end_of_the_toolbar_with_the_share_symbol() throws {
    let framedWindow = conversationWindow()
    let toolbar = try XCTUnwrap(framedWindow.window.toolbar)
    let exportItem = try XCTUnwrap(toolbar.items.last)
    XCTAssertEqual(exportItem.itemIdentifier, .thinTalkExport)
    XCTAssertEqual(exportItem.label, "Export")
    XCTAssertEqual(exportItem.action, #selector(ThinTalkConversationActions.exportConversation(_:)))
  }

  func test_should_measure_a_titlebar_inset_that_clears_the_toolbar_but_not_the_page() {
    let window = conversationWindow().window
    let titlebarInset = ThinTalkConversationWindow.titlebarInset(of: window)
    XCTAssertGreaterThan(titlebarInset, 20, "the toolbar strip has real height")
    XCTAssertLessThanOrEqual(titlebarInset, 200, "the page accepts insets up to 200 points")
  }

  func test_should_report_no_inset_for_a_window_without_a_title_bar_overlay() {
    let plainWindow = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 400, height: 300), styleMask: [.borderless],
      backing: .buffered, defer: true)
    XCTAssertEqual(ThinTalkConversationWindow.titlebarInset(of: plainWindow), 0)
  }

  func test_should_never_shrink_below_the_size_the_layout_needs() {
    let window = conversationWindow().window
    XCTAssertEqual(window.contentMinSize, ThinTalkConversationWindow.minimumContentSize)
  }
}
