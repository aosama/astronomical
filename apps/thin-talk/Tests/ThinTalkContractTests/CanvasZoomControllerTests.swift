import XCTest
@testable import ThinTalkCanvas

/// Zoom arithmetic for the conversation canvas.
///
/// The clamp is the part of zoom that can silently drift out of spec (a stuck
/// ceiling, a negative zoom), so it is pure arithmetic under test; the web view
/// wiring itself needs a live window and is exercised by the journey suites.
@MainActor
final class CanvasZoomControllerTests: XCTestCase {
  func test_should_clamp_zoom_in_at_the_maximum() {
    let zoom = CanvasZoomController.clampedPageZoom(
      current: CanvasZoomController.maximumPageZoom, factor: CanvasZoomController.zoomInStep)
    XCTAssertEqual(zoom, CanvasZoomController.maximumPageZoom, accuracy: 0.0001)
  }

  func test_should_clamp_zoom_out_at_the_minimum() {
    let zoom = CanvasZoomController.clampedPageZoom(
      current: CanvasZoomController.minimumPageZoom, factor: 1 / CanvasZoomController.zoomInStep)
    XCTAssertEqual(zoom, CanvasZoomController.minimumPageZoom, accuracy: 0.0001)
  }

  func test_should_scale_an_ordinary_zoom_step_without_clamping() {
    let zoom = CanvasZoomController.clampedPageZoom(current: 1.0, factor: CanvasZoomController.zoomInStep)
    XCTAssertEqual(zoom, 1.1, accuracy: 0.0001)
  }

  func test_should_keep_zoom_in_range_across_a_long_zoom_in_run() {
    var zoom = CanvasZoomController.defaultPageZoom
    for _ in 0..<64 {
      zoom = CanvasZoomController.clampedPageZoom(current: zoom, factor: CanvasZoomController.zoomInStep)
    }
    XCTAssertEqual(zoom, CanvasZoomController.maximumPageZoom, accuracy: 0.0001)
  }

  func test_should_keep_zoom_in_range_across_a_long_zoom_out_run() {
    var zoom = CanvasZoomController.defaultPageZoom
    for _ in 0..<64 {
      zoom = CanvasZoomController.clampedPageZoom(current: zoom, factor: 1 / CanvasZoomController.zoomInStep)
    }
    XCTAssertEqual(zoom, CanvasZoomController.minimumPageZoom, accuracy: 0.0001)
  }

  func test_should_reject_a_non_positive_zoom_factor() {
    let zoomedToFloor = CanvasZoomController.clampedPageZoom(current: 1.0, factor: 0)
    XCTAssertEqual(zoomedToFloor, CanvasZoomController.minimumPageZoom, accuracy: 0.0001)
    let zoomedNegative = CanvasZoomController.clampedPageZoom(current: 1.0, factor: -2)
    XCTAssertEqual(zoomedNegative, CanvasZoomController.minimumPageZoom, accuracy: 0.0001)
  }

  func test_should_accept_a_bare_command_equals_as_the_zoom_in_alias() {
    let isAlias = CanvasZoomController.isZoomInKeyAlias(
      keyCharacters: "=", modifierFlags: .command)
    XCTAssertTrue(isAlias)
  }

  func test_should_accept_the_shifted_command_plus_keystroke_users_press() {
    let isAlias = CanvasZoomController.isZoomInKeyAlias(
      keyCharacters: "+", modifierFlags: [.command, .shift])
    XCTAssertTrue(isAlias, "⌘+ is how zoom-in is actually typed; the alias must answer it")
  }

  func test_should_accept_the_numpad_command_plus_keystroke() {
    let isAlias = CanvasZoomController.isZoomInKeyAlias(
      keyCharacters: "+", modifierFlags: [.command, .numericPad])
    XCTAssertTrue(isAlias, "the full-keyboard numpad plus must zoom in like the shifted plus")
  }

  func test_should_reject_zoom_in_keys_carrying_option_or_control() {
    let withOption = CanvasZoomController.isZoomInKeyAlias(
      keyCharacters: "=", modifierFlags: [.command, .option])
    let withControl = CanvasZoomController.isZoomInKeyAlias(
      keyCharacters: "+", modifierFlags: [.command, .control])
    XCTAssertFalse(withOption)
    XCTAssertFalse(withControl)
  }

  func test_should_reject_other_keys_and_missing_command_modifier() {
    let zoomOutKey = CanvasZoomController.isZoomInKeyAlias(
      keyCharacters: "-", modifierFlags: .command)
    let actualSizeKey = CanvasZoomController.isZoomInKeyAlias(
      keyCharacters: "0", modifierFlags: .command)
    let plusWithoutCommand = CanvasZoomController.isZoomInKeyAlias(
      keyCharacters: "+", modifierFlags: [.shift])
    let equalsWithoutCommand = CanvasZoomController.isZoomInKeyAlias(
      keyCharacters: "=", modifierFlags: [])
    let missingCharacters = CanvasZoomController.isZoomInKeyAlias(
      keyCharacters: nil, modifierFlags: .command)
    XCTAssertFalse(zoomOutKey)
    XCTAssertFalse(actualSizeKey)
    XCTAssertFalse(plusWithoutCommand)
    XCTAssertFalse(equalsWithoutCommand)
    XCTAssertFalse(missingCharacters)
  }
}
