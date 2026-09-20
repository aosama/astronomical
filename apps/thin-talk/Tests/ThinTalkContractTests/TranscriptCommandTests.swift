import Foundation
import ThinTalkCore
import XCTest

/// How the bridge chooses what to send. Streaming a token at a time is only cheap
/// if a token-sized change produces a token-sized message, so these cases pin the
/// planner's decisions rather than the canvas implementation.
final class TranscriptCommandTests: XCTestCase {
  private let messageID = UUID()

  private func snapshot(
    markdown: String,
    reasoning: String = "",
    state: TranscriptMessageState = .complete,
    model: String? = "Astronomical 3B Chat",
    channel: String = "Development",
    notice: String? = nil
  ) -> TranscriptSnapshot {
    TranscriptSnapshot(
      messages: [
        TranscriptMessage(
          id: messageID, role: .assistant, markdown: markdown, reasoning: reasoning, state: state)
      ],
      model: model,
      channel: channel,
      notice: notice
    )
  }

  func test_should_send_a_full_snapshot_when_nothing_was_sent_before() {
    let next = snapshot(markdown: "Hello")
    XCTAssertEqual(TranscriptCommandPlanner.plan(previous: nil, next: next), [.snapshot(next)])
  }

  func test_should_send_nothing_when_the_conversation_did_not_change() {
    let current = snapshot(markdown: "Hello")
    XCTAssertEqual(TranscriptCommandPlanner.plan(previous: current, next: current), [])
  }

  func test_should_send_only_the_growing_message_while_streaming() {
    let previous = snapshot(markdown: "The ceiling", state: .streaming)
    let next = snapshot(markdown: "The ceiling applies to wired memory.", state: .streaming)
    let commands = TranscriptCommandPlanner.plan(previous: previous, next: next)
    XCTAssertEqual(commands.count, 1, "a delta in one answer must not resend the conversation")
    guard case .message(let message) = commands.first else {
      return XCTFail("expected a single message command, found \(commands)")
    }
    XCTAssertEqual(message.id, messageID)
    XCTAssertEqual(message.markdown, "The ceiling applies to wired memory.")
  }

  func test_should_send_a_full_snapshot_when_a_message_is_added() {
    let previous = snapshot(markdown: "First")
    let next = TranscriptSnapshot(
      messages: previous.messages + [TranscriptMessage(role: .user, markdown: "Second")],
      model: previous.model,
      channel: previous.channel
    )
    XCTAssertEqual(TranscriptCommandPlanner.plan(previous: previous, next: next), [.snapshot(next)])
  }

  func test_should_send_a_full_snapshot_when_two_messages_change() {
    let first = UUID()
    let second = UUID()
    let previous = TranscriptSnapshot(
      messages: [
        TranscriptMessage(id: first, role: .assistant, markdown: "One"),
        TranscriptMessage(id: second, role: .assistant, markdown: "Two"),
      ], channel: "Development")
    let next = TranscriptSnapshot(
      messages: [
        TranscriptMessage(id: first, role: .assistant, markdown: "One changed"),
        TranscriptMessage(id: second, role: .assistant, markdown: "Two changed"),
      ], channel: "Development")
    XCTAssertEqual(TranscriptCommandPlanner.plan(previous: previous, next: next), [.snapshot(next)])
  }

  func test_should_send_a_full_snapshot_when_the_selected_model_changes() {
    let previous = snapshot(markdown: "Hello", model: "Astronomical 3B Chat")
    let next = snapshot(markdown: "Hello", model: "Astronomical 7B Chat")
    XCTAssertEqual(TranscriptCommandPlanner.plan(previous: previous, next: next), [.snapshot(next)])
  }

  func test_should_send_a_full_snapshot_when_only_the_notice_changes() {
    let previous = snapshot(markdown: "Hello", notice: "Looking for a chat model on this Mac.")
    let next = snapshot(markdown: "Hello", notice: nil)
    XCTAssertEqual(TranscriptCommandPlanner.plan(previous: previous, next: next), [.snapshot(next)])
  }

  func test_should_treat_a_message_state_change_as_a_single_message_update() {
    let previous = snapshot(markdown: "Done", state: .streaming)
    let next = snapshot(markdown: "Done", state: .complete)
    guard case .message(let message)? = TranscriptCommandPlanner.plan(previous: previous, next: next).first
    else {
      return XCTFail("a state change must update the one message that changed")
    }
    XCTAssertEqual(message.state, .complete)
  }
}

/// The wire form the canvas receives.
final class TranscriptCommandCodecTests: XCTestCase {
  func test_should_invoke_the_single_canvas_entry_point() throws {
    let invocation = try TranscriptCommandCodec.invocation(for: .appearance(dark: true))
    XCTAssertTrue(invocation.hasPrefix("window.__thintalk.receive(\""))
    XCTAssertTrue(invocation.hasSuffix("\")"))
  }

  func test_should_encode_model_output_that_cannot_escape_the_call() throws {
    // A hostile prompt or answer can contain the characters that would break
    // interpolated source: a closing quote, a closing script tag, a newline.
    let hostile = "He said \"stop\" and wrote </script>\nand then `code`."
    let command = TranscriptCommand.message(
      TranscriptMessage(id: UUID(), role: .assistant, markdown: hostile))
    let invocation = try TranscriptCommandCodec.invocation(for: command)

    XCTAssertEqual(
      invocation.filter { $0 == "\"" }.count, 2,
      "the payload must contain no quote that could end the call")
    XCTAssertFalse(invocation.contains("</script>"))
    XCTAssertFalse(invocation.contains("\n"))
    let payload = try XCTUnwrap(
      invocation.split(separator: "\"").dropFirst().first.map(String.init))
    XCTAssertNotNil(
      Data(base64Encoded: payload), "the payload must be plain Base64")
  }

  func test_should_round_trip_a_snapshot_through_the_wire_form() throws {
    let attachment = TranscriptAttachment(
      assetURL: "thintalk-asset://attachment/abc", label: "chart")
    let snapshot = TranscriptSnapshot(
      messages: [
        TranscriptMessage(id: UUID(), role: .user, markdown: "What is the ceiling?"),
        TranscriptMessage(
          id: UUID(),
          role: .assistant,
          markdown: "**20 GB**",
          reasoning: "Read the configured ceiling.",
          attachments: [attachment],
          state: .streaming),
      ],
      model: "Astronomical 3B Chat",
      channel: "Development",
      notice: nil
    )
    let decoded = try decodePayload(of: .snapshot(snapshot))

    XCTAssertEqual(decoded["v"] as? Int, 1)
    XCTAssertEqual(decoded["kind"] as? String, "snapshot")
    XCTAssertEqual(decoded["channel"] as? String, "Development")
    XCTAssertEqual(decoded["model"] as? String, "Astronomical 3B Chat")
    let messages = try XCTUnwrap(decoded["messages"] as? [[String: Any]])
    XCTAssertEqual(messages.count, 2)
    XCTAssertEqual(messages[0]["role"] as? String, "user")
    XCTAssertEqual(messages[1]["markdown"] as? String, "**20 GB**")
    XCTAssertEqual(messages[1]["reasoning"] as? String, "Read the configured ceiling.")
    XCTAssertEqual(messages[1]["state"] as? String, "streaming")
    let attachments = try XCTUnwrap(messages[1]["attachments"] as? [[String: Any]])
    XCTAssertEqual(attachments.first?["assetURL"] as? String, "thintalk-asset://attachment/abc")
    XCTAssertEqual(attachments.first?["label"] as? String, "chart")
  }

  func test_should_encode_a_single_message_command_with_its_identifier() throws {
    let id = UUID()
    let message = TranscriptMessage(
      id: id, role: .assistant, markdown: "Growing", state: .streaming)
    let decoded = try decodePayload(of: .message(message))

    XCTAssertEqual(decoded["kind"] as? String, "message")
    let encodedMessage = try XCTUnwrap(decoded["message"] as? [String: Any])
    XCTAssertEqual(encodedMessage["id"] as? String, id.uuidString)
    XCTAssertEqual(encodedMessage["state"] as? String, "streaming")
  }

  func test_should_encode_appearance_as_its_own_command() throws {
    let decoded = try decodePayload(of: .appearance(dark: false))
    XCTAssertEqual(decoded["kind"] as? String, "appearance")
    XCTAssertEqual(decoded["dark"] as? Bool, false)
  }

  private func decodePayload(of command: TranscriptCommand) throws -> [String: Any] {
    let invocation = try TranscriptCommandCodec.invocation(for: command)
    let payload = try XCTUnwrap(invocation.split(separator: "\"").dropFirst().first.map(String.init))
    let data = try XCTUnwrap(Data(base64Encoded: payload))
    return try XCTUnwrap(
      JSONSerialization.jsonObject(with: data) as? [String: Any])
  }
}