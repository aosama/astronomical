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
    let isAlias = CanvasZoomController.isUnshiftedZoomInAlias(
      keyCharacters: "=", modifierFlags: .command)
    XCTAssertTrue(isAlias)
  }

  func test_should_reject_the_shifted_variant_the_view_menu_already_owns() {
    let isAlias = CanvasZoomController.isUnshiftedZoomInAlias(
      keyCharacters: "+", modifierFlags: [.command, .shift])
    XCTAssertFalse(isAlias, "the View menu answers ⌘+ itself; the alias must leave it alone")
  }

  func test_should_reject_command_equals_carrying_extra_modifiers() {
    let withOption = CanvasZoomController.isUnshiftedZoomInAlias(
      keyCharacters: "=", modifierFlags: [.command, .option])
    let withControl = CanvasZoomController.isUnshiftedZoomInAlias(
      keyCharacters: "=", modifierFlags: [.command, .control])
    XCTAssertFalse(withOption)
    XCTAssertFalse(withControl)
  }

  func test_should_reject_other_keys_and_missing_command_modifier() {
    let zoomOutKey = CanvasZoomController.isUnshiftedZoomInAlias(
      keyCharacters: "-", modifierFlags: .command)
    let equalsWithoutCommand = CanvasZoomController.isUnshiftedZoomInAlias(
      keyCharacters: "=", modifierFlags: [])
    let missingCharacters = CanvasZoomController.isUnshiftedZoomInAlias(
      keyCharacters: nil, modifierFlags: .command)
    XCTAssertFalse(zoomOutKey)
    XCTAssertFalse(equalsWithoutCommand)
    XCTAssertFalse(missingCharacters)
  }
}
