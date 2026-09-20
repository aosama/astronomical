import XCTest

@testable import ThinTalkCore

/// Proves the thinking-budget control is real: that the levels mean the token
/// counts the product promises, that a fresh install starts on Quick, and that the
/// chosen level actually leaves the process on the REST request rather than only
/// changing what the window draws.
final class ThinkingEffortTests: XCTestCase {
  override func tearDown() {
    StubSupervisorURLProtocol.chatResponse = nil
    StubSupervisorURLProtocol.receivedRequestPaths = []
    StubSupervisorURLProtocol.receivedRequestBodies = []
    super.tearDown()
  }

  // MARK: - Levels

  func test_should_bound_each_level_at_the_documented_token_budget() {
    XCTAssertEqual(ThinkingEffort.quick.thinkingBudgetTokens, 256)
    XCTAssertEqual(ThinkingEffort.balanced.thinkingBudgetTokens, 512)
    XCTAssertEqual(ThinkingEffort.high.thinkingBudgetTokens, 1024)
  }

  func test_should_offer_the_three_levels_in_order_and_start_on_quick() {
    XCTAssertEqual(ThinkingEffort.allCases, [.quick, .balanced, .high])
    XCTAssertEqual(ThinkingEffort.default, .quick)
  }

  func test_should_state_the_token_cost_in_the_label_a_reader_picks() {
    XCTAssertEqual(ThinkingEffort.quick.budgetSummary, "Quick · 256 tokens")
    XCTAssertEqual(ThinkingEffort.high.budgetSummary, "High · 1024 tokens")
  }

  // MARK: - Remembering the choice

  func test_should_start_a_fresh_install_on_quick() {
    XCTAssertEqual(ThinkingEffortPreference.load(from: isolatedDefaults()), .quick)
  }

  func test_should_remember_a_deliberate_choice_across_launches() {
    let defaults = isolatedDefaults()
    ThinkingEffortPreference.save(.high, to: defaults)
    XCTAssertEqual(ThinkingEffortPreference.load(from: defaults), .high)
  }

  func test_should_fall_back_to_quick_when_a_stored_level_is_unrecognised() {
    let defaults = isolatedDefaults()
    defaults.set("turbo", forKey: ThinkingEffortPreference.storageKey)
    XCTAssertEqual(
      ThinkingEffortPreference.load(from: defaults), .quick,
      "a value written by a different release must not leave the control unusable")
  }

  // MARK: - Propagation to the REST call

  func test_should_send_the_selected_budget_on_the_rest_request() async throws {
    StubSupervisorURLProtocol.chatResponse = StubSupervisorURLProtocol.stubbedChat(
      sseBody: Data("data: [DONE]\n\n".utf8))

    for level in ThinkingEffort.allCases {
      StubSupervisorURLProtocol.receivedRequestBodies = []
      for await _ in StubbedClientFactory.stableClient().chatStream(
        modelID: "fictional/chat-7b",
        messages: [ChatMessage(role: .user, content: "Explain expert paging.")],
        thinkingEffort: level
      ) {}

      let body = try XCTUnwrap(
        StubSupervisorURLProtocol.receivedRequestBodies.first,
        "the request that left the process must be observable, not assumed")
      let document = try XCTUnwrap(
        try JSONSerialization.jsonObject(with: body) as? [String: Any],
        "the chat request body must be a JSON object")
      XCTAssertEqual(
        document["thinking_budget"] as? Int, level.thinkingBudgetTokens,
        "the \(level.displayName) level must put its own budget on the wire")
      XCTAssertEqual(document["stream"] as? Bool, true)
    }
  }

  func test_should_put_the_default_budget_on_the_wire_when_the_reader_changes_nothing() async throws {
    StubSupervisorURLProtocol.chatResponse = StubSupervisorURLProtocol.stubbedChat(
      sseBody: Data("data: [DONE]\n\n".utf8))

    for await _ in StubbedClientFactory.stableClient().chatStream(
      modelID: "fictional/chat-7b",
      messages: [ChatMessage(role: .user, content: "Hi")],
      thinkingEffort: ThinkingEffortPreference.load(from: isolatedDefaults())
    ) {}

    let body = try XCTUnwrap(StubSupervisorURLProtocol.receivedRequestBodies.first)
    let document = try XCTUnwrap(try JSONSerialization.jsonObject(with: body) as? [String: Any])
    XCTAssertEqual(
      document["thinking_budget"] as? Int, 256,
      "an untouched install must ask for the quick budget, not for no budget at all")
  }

  // MARK: - Support

  /// A defaults store with no trace of any earlier choice, so a pass cannot depend
  /// on what a previous test run happened to leave behind.
  private func isolatedDefaults() -> UserDefaults {
    let suiteName = "thin-talk.thinking-effort.tests.\(UUID().uuidString)"
    let defaults = UserDefaults(suiteName: suiteName) ?? .standard
    defaults.removePersistentDomain(forName: suiteName)
    return defaults
  }
}