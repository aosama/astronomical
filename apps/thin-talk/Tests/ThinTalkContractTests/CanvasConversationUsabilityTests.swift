import Foundation
import ThinTalkCore
import XCTest

/// Journey contracts for conversation readability: the user's own turns must be
/// recognisably theirs and the transcript must stay on the newest content.
/// These capture the usability failures seen in live use — user turns styled
/// like compose fields, and new turns arriving off-screen.
@MainActor
final class CanvasConversationUsabilityTests: CanvasTestCase {
  func test_a_user_turn_offers_copy_but_not_regenerate() async throws {
    let harness = try XCTUnwrap(harness)
    let turn = TranscriptMessage(
      id: UUID(), role: .user, markdown: "how is life?", state: .complete)
    try await harness.send(.snapshot(TranscriptSnapshot(messages: [turn], channel: "Development")))
    try await harness.waitForElement("[data-testid='action-copy']")

    let regenerateCount = try await harness.count(of: "[data-testid='action-regenerate']")
    XCTAssertEqual(
      regenerateCount, 0,
      "regenerating belongs to what the model produced, never to the user's own words")
  }

  func test_an_assistant_turn_still_offers_both_actions() async throws {
    let harness = try XCTUnwrap(harness)
    let turn = TranscriptMessage(
      id: UUID(), role: .assistant, markdown: "Answered.", state: .complete)
    try await harness.send(.snapshot(TranscriptSnapshot(messages: [turn], channel: "Development")))

    try await harness.waitForElement("[data-testid='action-regenerate']")
  }

  func test_a_user_turn_is_scrolled_out_of_the_compose_surface_look() async throws {
    let harness = try XCTUnwrap(harness)
    let turn = TranscriptMessage(
      id: UUID(), role: .user, markdown: "what can you help me with?", state: .complete)
    try await harness.send(.snapshot(TranscriptSnapshot(messages: [turn], channel: "Development")))
    try await harness.waitForElement("article.message--user")

    let isRightAligned = try await harness.bool(
      of: "getComputedStyle(document.querySelector('article.message--user')).alignItems === 'flex-end'"
    )
    XCTAssertTrue(
      isRightAligned,
      "the user's turn must read as something they said, not as another input field")
  }

  func test_a_new_user_turn_pulls_the_transcript_to_the_newest_content() async throws {
    let harness = try XCTUnwrap(harness)
    // Content tall enough to overflow the viewport, so reaching the newest turn
    // requires an actual scroll rather than a no-op.
    let filler = Array(repeating: "Filler paragraph so the transcript overflows.", count: 60)
      .joined(separator: "\n\n")
    let earlier = TranscriptMessage(
      id: UUID(), role: .assistant, markdown: filler, state: .complete)
    try await harness.send(
      .snapshot(TranscriptSnapshot(messages: [earlier], channel: "Development")))
    try await harness.waitForElement("article.message")

    // The reader scrolled up, away from the newest content.
    try await harness.setJS("document.getElementById('transcript').scrollTop = 0")

    let newest = TranscriptMessage(
      id: UUID(), role: .user, markdown: "a fresh question", state: .complete)
    try await harness.send(.snapshot(TranscriptSnapshot(messages: [earlier, newest], channel: "Development")))
    try await harness.waitForElement("article.message--user")

    let isPinnedToBottom = try await harness.bool(
      of: """
      document.getElementById('transcript').scrollHeight
        - document.getElementById('transcript').scrollTop
        - document.getElementById('transcript').clientHeight < 60
      """)
    XCTAssertTrue(
      isPinnedToBottom,
      "sending a turn must bring the newest content back into view")
  }

  func test_turns_render_in_conversation_order() async throws {
    let harness = try XCTUnwrap(harness)
    let first = TranscriptMessage(id: UUID(), role: .user, markdown: "first", state: .complete)
    let second = TranscriptMessage(id: UUID(), role: .assistant, markdown: "second", state: .complete)
    let third = TranscriptMessage(id: UUID(), role: .user, markdown: "third", state: .complete)
    try await harness.send(
      .snapshot(TranscriptSnapshot(messages: [first, second, third], channel: "Development")))
    try await harness.waitForMessageCount(3)

    let classes = try await harness.bool(
      of: """
      (function () {
        var articles = document.querySelectorAll('article.message');
        if (articles.length < 3) { return false; }
        return articles[0].classList.contains('message--user')
          && articles[1].classList.contains('message--assistant')
          && articles[2].classList.contains('message--user');
      })()
      """)
    XCTAssertTrue(
      classes,
      "transcript order must follow conversation order: user, assistant, user")
  }

  func test_the_you_label_lives_inside_the_user_bubble_not_a_meta_row() async throws {
    let harness = try XCTUnwrap(harness)
    let turn = TranscriptMessage(
      id: UUID(), role: .user, markdown: "hi", state: .complete)
    try await harness.send(.snapshot(TranscriptSnapshot(messages: [turn], channel: "Development")))
    try await harness.waitForElement("article.message--user .message__answer--user")

    // The "You" label must be inside .message__answer--user (the bubble),
    // not in a separate .message__meta row above it.
    let insideBubble = try await harness.bool(
      of: """
      document.querySelector('article.message--user .message__answer--user .message__role') !== null
      """.trimmingCharacters(in: .whitespacesAndNewlines))
    XCTAssertTrue(insideBubble, "'You' label must live inside the user bubble")

    let noSeparateMetaRow = try await harness.bool(
      of: """
      document.querySelector('article.message--user .message__meta') === null
      """.trimmingCharacters(in: .whitespacesAndNewlines))
    XCTAssertTrue(noSeparateMetaRow, "user turns must not carry a separate meta row above the bubble")
  }
}
