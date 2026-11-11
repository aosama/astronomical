import AppKit
import XCTest

@testable import ThinTalkCanvas

/// The chat window's top strip has to act as a title bar even where the
/// conversation scrolls under it: a double-click there zoomed the window until
/// the web view started taking the clicks and selecting words instead.
final class TitlebarStripTests: XCTestCase {
  func test_should_zoom_on_a_mac_that_never_changed_the_setting() {
    XCTAssertEqual(TitlebarDoubleClickAction.from(actionOnDoubleClick: nil, minimizesOnDoubleClick: false), .zoom)
  }

  func test_should_follow_each_double_click_choice_in_system_settings() {
    for (storedChoice, expectedAction) in [
      ("Maximize", TitlebarDoubleClickAction.zoom), ("Fill", .zoom), ("Minimize", .minimize), ("None", .doNothing),
    ] {
      XCTAssertEqual(
        TitlebarDoubleClickAction.from(actionOnDoubleClick: storedChoice, minimizesOnDoubleClick: false), expectedAction,
        storedChoice)
    }
  }

  func test_should_read_the_older_minimise_switch_only_without_the_newer_choice() {
    XCTAssertEqual(TitlebarDoubleClickAction.from(actionOnDoubleClick: nil, minimizesOnDoubleClick: true), .minimize)
    XCTAssertEqual(
      TitlebarDoubleClickAction.from(actionOnDoubleClick: "Maximize", minimizesOnDoubleClick: true), .zoom,
      "the newer choice wins")
  }

  func test_should_treat_an_unknown_future_choice_as_zoom() {
    XCTAssertEqual(
      TitlebarDoubleClickAction.from(actionOnDoubleClick: "Something New", minimizesOnDoubleClick: false), .zoom)
  }

  func test_should_pin_the_strip_to_the_top_edge_at_the_title_bar_height() {
    let canvasBounds = NSRect(x: 0, y: 0, width: 1080, height: 720)

    XCTAssertEqual(
      ConversationCanvasView.titlebarStripFrame(in: canvasBounds, stripHeight: 52, isFlipped: false),
      NSRect(x: 0, y: 668, width: 1080, height: 52))
    XCTAssertEqual(
      ConversationCanvasView.titlebarStripFrame(in: canvasBounds, stripHeight: 52, isFlipped: true),
      NSRect(x: 0, y: 0, width: 1080, height: 52))
  }

  func test_should_have_no_strip_without_a_title_bar_and_never_exceed_the_canvas() {
    let canvasBounds = NSRect(x: 0, y: 0, width: 1080, height: 40)

    XCTAssertEqual(
      ConversationCanvasView.titlebarStripFrame(in: canvasBounds, stripHeight: 0, isFlipped: false).height, 0)
    XCTAssertEqual(
      ConversationCanvasView.titlebarStripFrame(in: canvasBounds, stripHeight: -5, isFlipped: false).height, 0)
    XCTAssertEqual(
      ConversationCanvasView.titlebarStripFrame(in: canvasBounds, stripHeight: 52, isFlipped: false),
      NSRect(x: 0, y: 0, width: 1080, height: 40))
  }

  @MainActor
  func test_should_take_the_first_click_so_it_never_reaches_the_page() {
    XCTAssertTrue(TitlebarStripView().acceptsFirstMouse(for: nil))
  }
}
