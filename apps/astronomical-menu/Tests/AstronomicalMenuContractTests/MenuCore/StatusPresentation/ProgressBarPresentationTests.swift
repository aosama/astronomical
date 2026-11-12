// Contract for `showsProgressBar`, which alone decides whether the popover's
// progress bar is drawn. Cases where it is hidden (model load, prompt-cache
// restore, the default generation budget) are pinned beside their journeys in
// ModelLoadInFlightPresentationTests and StatusPresentationContractTests.

import XCTest

@testable import AstronomicalMenuCore

final class ProgressBarPresentationTests: XCTestCase {
  private func generationStatus(processedTokens: Int, totalTokens: Int) throws -> DaemonStatusDocument {
    try JSONDecoder().decode(
      DaemonStatusDocument.self,
      from: Data(
        """
        {"status":"ready","activity":"generating","ready_model_id":"Ornith-9B","progress":{"phase":"generation","processed_tokens":\(processedTokens),"total_tokens":\(totalTokens),"elapsed_ms":1000},"expert_memory_mode":"resident","serving_session":{"completed_request_count":1,"total_prompt_token_count":4096,"total_reused_prompt_token_count":0,"average_prefill_tok_per_second":1000,"average_generation_tok_per_second":27.0}}
        """.utf8)
    )
  }

  func test_should_draw_the_bar_for_a_generation_with_an_explicit_cap() throws {
    let statusDocument = try generationStatus(processedTokens: 27, totalTokens: 512)

    XCTAssertTrue(statusDocument.showsProgressBar)
    XCTAssertEqual(statusDocument.progressCompletedUnitCount, 27)
    XCTAssertEqual(statusDocument.progressTotalUnitCount, 512)
  }

  func test_should_draw_the_bar_just_below_the_default_budget_threshold() throws {
    let statusDocument = try generationStatus(processedTokens: 27, totalTokens: 8_191)

    XCTAssertTrue(statusDocument.showsProgressBar)
    XCTAssertEqual(statusDocument.progressTitle, "27 / 8191 tokens")
  }

  func test_should_hide_the_bar_from_the_default_budget_threshold_up() throws {
    let statusDocument = try generationStatus(processedTokens: 27, totalTokens: 8_192)

    XCTAssertFalse(statusDocument.showsProgressBar)
    XCTAssertEqual(statusDocument.progressTitle, "27 tokens")
  }

  func test_should_fill_the_bar_by_the_completed_fraction() throws {
    let statusDocument = try generationStatus(processedTokens: 128, totalTokens: 512)

    XCTAssertEqual(statusDocument.progressBarFillFraction, 0.25)
    XCTAssertEqual(statusDocument.progressBarAccessibilityValue, "25 percent")
  }

  func test_should_fill_the_bar_fully_when_completed_equals_total() throws {
    let statusDocument = try generationStatus(processedTokens: 512, totalTokens: 512)

    XCTAssertEqual(statusDocument.progressBarFillFraction, 1)
    XCTAssertEqual(statusDocument.progressBarAccessibilityValue, "100 percent")
  }

  func test_should_clamp_the_fill_when_completed_exceeds_total() throws {
    let statusDocument = try generationStatus(processedTokens: 600, totalTokens: 512)

    XCTAssertEqual(statusDocument.progressBarFillFraction, 1)
  }

  func test_should_fill_nothing_when_the_total_is_zero() throws {
    let statusDocument = try generationStatus(processedTokens: 0, totalTokens: 0)

    XCTAssertEqual(statusDocument.progressBarFillFraction, 0)
    XCTAssertEqual(statusDocument.progressBarAccessibilityValue, "0 percent")
  }

  func test_should_draw_an_empty_bar_while_idle() throws {
    let statusDocument = try JSONDecoder().decode(
      DaemonStatusDocument.self,
      from: Data(#"{"status":"ready","activity":"idle","ready_model_id":null,"progress":null}"#.utf8)
    )

    XCTAssertTrue(statusDocument.showsProgressBar)
    XCTAssertEqual(statusDocument.progressCompletedUnitCount, 0)
    XCTAssertEqual(statusDocument.progressTotalUnitCount, 1)
  }
}
