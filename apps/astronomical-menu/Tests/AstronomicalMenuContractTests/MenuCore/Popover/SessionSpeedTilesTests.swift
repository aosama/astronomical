import XCTest

@testable import AstronomicalMenuCore

final class SessionSpeedTilesTests: XCTestCase {
  private func statusDocument(
    activity: String = "idle", progressJSON: String? = nil, requestCount: Int = 12,
    prefillSpeed: Double = 2290.4, generationSpeed: Double = 62.6
  ) throws -> DaemonStatusDocument {
    let servingSession =
      "{\"completed_request_count\":\(requestCount),\"total_prompt_token_count\":0,\"total_reused_prompt_token_count\":0,\"target_prompt_work_token_count\":0,\"target_reused_prompt_work_token_count\":0,\"average_prefill_tok_per_second\":\(prefillSpeed),\"average_generation_tok_per_second\":\(generationSpeed)}"
    let progressField = progressJSON.map { ",\"progress\":\($0)" } ?? ""
    let statusJSON =
      "{\"status\":\"ready\",\"activity\":\"\(activity)\",\"ready_model_id\":\"model\"\(progressField),\"serving_session\":\(servingSession)}"
    return try JSONDecoder().decode(DaemonStatusDocument.self, from: Data(statusJSON.utf8))
  }

  func test_should_show_both_session_averages_as_whole_grouped_numbers_when_idle() throws {
    let tilesPresentation = SessionSpeedTilesPresentation(statusDocument: try statusDocument())
    XCTAssertEqual(tilesPresentation.readingTile.title, "Reading speed")
    XCTAssertEqual(tilesPresentation.readingTile.speedTitle, "2,290 tok/s")
    XCTAssertEqual(tilesPresentation.writingTile.title, "Writing speed")
    XCTAssertEqual(tilesPresentation.writingTile.speedTitle, "63 tok/s")
    XCTAssertFalse(tilesPresentation.readingTile.isLive)
    XCTAssertFalse(tilesPresentation.writingTile.isLive)
    XCTAssertEqual(tilesPresentation.caption, "Average this session · 12 requests")
  }

  func test_should_show_the_running_request_speed_on_the_tile_for_its_phase_only() throws {
    let reading = SessionSpeedTilesPresentation(
      statusDocument: try statusDocument(
        activity: "prompt_processing",
        progressJSON: "{\"phase\":\"target\",\"processed_tokens\":4096,\"total_tokens\":10179,\"elapsed_ms\":2000}"))
    XCTAssertEqual(reading.readingTile.speedTitle, "2,048 tok/s")
    XCTAssertTrue(reading.readingTile.isLive)
    XCTAssertEqual(reading.writingTile.speedTitle, "63 tok/s")
    XCTAssertFalse(reading.writingTile.isLive)

    let writing = SessionSpeedTilesPresentation(
      statusDocument: try statusDocument(
        activity: "generating",
        progressJSON: "{\"phase\":\"generation\",\"processed_tokens\":156,\"total_tokens\":300,\"elapsed_ms\":2000}"))
    XCTAssertEqual(writing.writingTile.speedTitle, "78 tok/s")
    XCTAssertTrue(writing.writingTile.isLive)
    XCTAssertEqual(writing.readingTile.speedTitle, "2,290 tok/s")
    XCTAssertFalse(writing.readingTile.isLive)
  }

  func test_should_show_a_dash_not_zero_before_a_phase_has_been_measured() throws {
    let nothingMeasured = SessionSpeedTilesPresentation(
      statusDocument: try statusDocument(requestCount: 0, prefillSpeed: 0, generationSpeed: 0))
    XCTAssertEqual(nothingMeasured.readingTile.speedTitle, "—")
    XCTAssertEqual(nothingMeasured.writingTile.speedTitle, "—")
    XCTAssertEqual(nothingMeasured.caption, "Averages appear after the first request")
    XCTAssertEqual(
      SessionSpeedTilesPresentation(statusDocument: .unavailable).readingTile.speedTitle, "—")
    // A request under way with no token finished yet has no live speed to show either.
    let justStarted = SessionSpeedTilesPresentation(
      statusDocument: try statusDocument(
        activity: "prompt_processing",
        progressJSON: "{\"phase\":\"target\",\"processed_tokens\":0,\"total_tokens\":10179,\"elapsed_ms\":900}",
        requestCount: 0, prefillSpeed: 0, generationSpeed: 0))
    XCTAssertEqual(justStarted.readingTile.speedTitle, "—")
    XCTAssertFalse(justStarted.readingTile.isLive)
  }

  func test_should_name_prefill_and_decode_in_the_explanations_and_use_singular_grammar() throws {
    let oneRequest = SessionSpeedTilesPresentation(statusDocument: try statusDocument(requestCount: 1))
    XCTAssertEqual(oneRequest.caption, "Average this session · 1 request")
    XCTAssertTrue(oneRequest.readingTile.explanation.hasPrefix("Prefill:"))
    XCTAssertTrue(oneRequest.writingTile.explanation.hasPrefix("Decode:"))
    XCTAssertEqual(sessionSpeedTitle(tokensPerSecond: 999.5), "1,000 tok/s")
  }
}
