import Foundation
import ThinTalkCore
import XCTest

/// Journey contracts for the all-HTML conversation surface: the composer, the
/// thinking-effort control, and the failure banner live inside the canvas page,
/// while Swift keeps every behaviour. Each case drives the real bundled shell in
/// a real web view, so what is proven is what a reader can actually do; every
/// wait is bounded, so a page that never renders or never reports fails the case
/// instead of hanging it.
@MainActor
final class CanvasComposerTests: CanvasTestCase {
  // MARK: - Sending an ask

  func test_pressing_return_reports_a_send_action_carrying_the_draft() async throws {
    let harness = try XCTUnwrap(harness)
    try await harness.send(.composer(Self.state()))
    try await harness.waitForElement("[data-testid='composer-input']")

    let ask = "how much wired memory does the model need?"
    try await harness.setJS(
      """
      var field = document.querySelector('[data-testid="composer-input"]');
      field.value = '\(ask)';
      field.focus();
      """)
    try await pressReturn()

    try await harness.waitForReportedActionCount(1)
    let sends = try reportedActions(ofKind: "send")
    XCTAssertEqual(sends.count, 1, "Return in the composer must report exactly one send")
    XCTAssertEqual(sends.first?["detail"] as? String, ask)
  }

  func test_an_empty_draft_cannot_send() async throws {
    let harness = try XCTUnwrap(harness)
    try await harness.send(.composer(Self.state()))
    try await harness.waitForElement("[data-testid='composer-input']")

    let sendDisabled = try await harness.bool(
      of: "document.querySelector('[data-testid=\"composer-send\"]').disabled")
    XCTAssertTrue(sendDisabled, "the send control must be disabled while the draft is empty")

    try await pressReturn()
    try await Task.sleep(nanoseconds: 250_000_000)
    XCTAssertTrue(
      try reportedActions(ofKind: "send").isEmpty,
      "Return on an empty draft must never report a send")
  }

  func test_reporting_a_send_clears_the_field() async throws {
    let harness = try XCTUnwrap(harness)
    try await harness.send(.composer(Self.state()))
    try await harness.waitForElement("[data-testid='composer-input']")

    try await harness.setJS(
      """
      var field = document.querySelector('[data-testid="composer-input"]');
      field.value = 'an ask worth sending';
      field.focus();
      """)
    try await pressReturn()

    try await harness.waitForReportedActionCount(1)
    let cleared = try await harness.bool(
      of: "document.querySelector('[data-testid=\"composer-input\"]').value === ''")
    XCTAssertTrue(
      cleared,
      "a turn the page reported as sent must leave the field ready for the next ask")
  }

  // MARK: - Chrome that must never paint

  func test_a_hidden_stop_control_never_paints() async throws {
    let harness = try XCTUnwrap(harness)
    try await harness.send(.composer(Self.state()))
    try await harness.waitForElement("[data-testid='composer-input']")

    let stopPaints = try await harness.bool(
      of:
        "getComputedStyle(document.querySelector('[data-testid=\"composer-stop\"]')).display !== 'none'"
    )
    XCTAssertFalse(
      stopPaints,
      "the hidden stop control must not paint; a class display rule beats the hidden attribute otherwise")
  }

  func test_the_ask_field_never_offers_horizontal_scrolling() async throws {
    let harness = try XCTUnwrap(harness)
    try await harness.send(.composer(Self.state()))
    try await harness.waitForElement("[data-testid='composer-input']")

    let scrollsSideways = try await harness.bool(
      of:
        "getComputedStyle(document.querySelector('[data-testid=\"composer-input\"]')).overflowX !== 'hidden'"
    )
    XCTAssertFalse(
      scrollsSideways,
      "a wrapping ask never scrolls sideways; WebKit's zoomed overflow must not paint a phantom bar")
  }

  func test_the_ask_field_scrolls_vertically_only_once_growth_is_clamped() async throws {
    let harness = try XCTUnwrap(harness)
    try await harness.send(.composer(Self.state()))
    try await harness.waitForElement("[data-testid='composer-input']")

    let shortAskPaintsNoScrollbar = try await harness.bool(
      of:
        "getComputedStyle(document.querySelector('[data-testid=\"composer-input\"]')).overflowY === 'hidden'"
    )
    XCTAssertTrue(
      shortAskPaintsNoScrollbar,
      "a short ask must fill its field without showing a vertical scrollbar")

    try await harness.setJS(
      """
      var field = document.querySelector('[data-testid="composer-input"]');
      field.value = Array.from({length: 30}, function (_, line) { return 'line ' + line; }).join('\\n');
      field.dispatchEvent(new Event('input', { bubbles: true }));
      """)
    let clampedAskScrolls = try await harness.bool(
      of:
        "getComputedStyle(document.querySelector('[data-testid=\"composer-input\"]')).overflowY === 'auto'"
    )
    XCTAssertTrue(
      clampedAskScrolls,
      "an ask clamped by the growth ceiling must scroll its own overflow")
  }

  // MARK: - Streaming state

  func test_streaming_swaps_send_for_stop_and_stop_reports_stop() async throws {
    let harness = try XCTUnwrap(harness)
    try await harness.send(.composer(Self.state(isStreaming: true)))
    try await harness.waitForElement("[data-testid='composer-stop']")
    let sendHidden = try await harness.bool(
      of: "document.querySelector('[data-testid=\"composer-send\"]').hidden")
    XCTAssertTrue(sendHidden, "a streaming turn must not offer a second send")

    try await harness.click("[data-testid='composer-stop']")
    try await harness.waitForReportedActionCount(1)
    XCTAssertEqual(try reportedActions(ofKind: "stop").count, 1)

    try await harness.send(.composer(Self.state()))
    try await harness.waitForElement("[data-testid='composer-send']")
  }

  // MARK: - Thinking effort

  func test_the_effort_pill_lists_every_level_and_reports_the_choice() async throws {
    let harness = try XCTUnwrap(harness)
    try await harness.send(.composer(Self.state()))
    try await harness.waitForElement("[data-testid='composer-effort']")

    try await harness.click("[data-testid='composer-effort']")
    try await harness.waitForElement("[data-testid='composer-effort-balanced']")
    try await harness.click("[data-testid='composer-effort-balanced']")

    try await harness.waitForReportedActionCount(1)
    let choices = try reportedActions(ofKind: "setEffort")
    XCTAssertEqual(choices.first?["detail"] as? String, "balanced")

    try await harness.send(.composer(Self.state(effort: .balanced)))
    let label = try await harness.text(of: "[data-testid='composer-effort-label']")
    XCTAssertEqual(label, "Balanced", "the pill must show the level Swift pushed")
  }

  // MARK: - Failure banner

  func test_a_failure_renders_its_banner_and_retry_reports_retry() async throws {
    let harness = try XCTUnwrap(harness)
    try await harness.send(
      .composer(Self.state(
        failureMessage: "The supervisor stopped responding.",
        failureNextAction: "Check that Astronomical is running, then retry.")))
    try await harness.waitForElement("[data-testid='composer-banner']")

    let message = try await harness.text(of: "[data-testid='banner-message']")
    XCTAssertTrue(message.contains("stopped responding"))
    let nextAction = try await harness.text(of: "[data-testid='banner-next-action']")
    XCTAssertTrue(nextAction.contains("Check that Astronomical is running"))

    try await harness.click("[data-testid='composer-retry']")
    try await harness.waitForReportedActionCount(1)
    XCTAssertEqual(try reportedActions(ofKind: "retry").count, 1)

    try await harness.send(.composer(Self.state()))
    let bannerHidden = try await harness.bool(
      of: "document.querySelector('[data-testid=\"composer-banner\"]').hidden")
    XCTAssertTrue(bannerHidden, "clearing the failure must hide the banner")
  }

  // MARK: - Startup states

  func test_a_loading_surface_cannot_type_or_send() async throws {
    let harness = try XCTUnwrap(harness)
    try await harness.send(.composer(Self.state(acceptsInput: false, isReady: false)))
    try await harness.waitForElement("[data-testid='composer-input']")

    let disabled = try await harness.bool(
      of: "document.querySelector('[data-testid=\"composer-input\"]').disabled")
    XCTAssertTrue(disabled, "the field must reject typing while the surface is loading")

    try await pressReturn()
    try await Task.sleep(nanoseconds: 250_000_000)
    XCTAssertTrue(
      try reportedActions(ofKind: "send").isEmpty,
      "a loading surface must never report a send")
  }

  // MARK: - Accessibility

  func test_the_composer_and_banner_carry_accessible_names() async throws {
    let harness = try XCTUnwrap(harness)
    try await harness.send(
      .composer(Self.state(failureMessage: "boom", failureNextAction: "Retry it.")))
    try await harness.waitForElement("[data-testid='composer-banner']")

    let inputLabelled = try await harness.bool(
      of:
        "document.querySelector('[data-testid=\"composer-input\"]').getAttribute('aria-label') === 'Message Thin Talk'"
    )
    XCTAssertTrue(inputLabelled, "the ask field must be named for VoiceOver")
    let sendLabelled = try await harness.bool(
      of: "document.querySelector('[data-testid=\"composer-send\"]').getAttribute('aria-label') !== null")
    XCTAssertTrue(sendLabelled, "the send control must be named")
    let effortHasMenu = try await harness.bool(
      of: "document.querySelector('[data-testid=\"composer-effort\"]').getAttribute('aria-haspopup') === 'menu'")
    XCTAssertTrue(effortHasMenu, "the effort pill must expose its menu")
    let bannerAnnounced = try await harness.bool(
      of: "document.querySelector('[data-testid=\"composer-banner\"]').getAttribute('role') === 'alert'")
    XCTAssertTrue(bannerAnnounced, "a failure must be announced, not painted silently")
  }

  // MARK: - Support

  private static func state(
    isStreaming: Bool = false,
    acceptsInput: Bool = true,
    isReady: Bool = true,
    effort: ThinkingEffort = .quick,
    failureMessage: String? = nil,
    failureNextAction: String = ""
  ) -> CanvasComposerState {
    CanvasComposerState(
      isStreaming: isStreaming,
      acceptsInput: acceptsInput,
      isReady: isReady,
      effort: effort,
      failure: failureMessage.map {
        CanvasComposerFailure(message: $0, nextAction: failureNextAction)
      })
  }

  private func pressReturn() async throws {
    try await harness?.setJS(
      """
      var field = document.querySelector('[data-testid="composer-input"]');
      field.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
      """)
  }

  private func reportedActions(ofKind action: String) throws -> [[String: Any]] {
    let harness = try XCTUnwrap(harness)
    return harness.reportedActions.filter { $0["action"] as? String == action }
  }
}
