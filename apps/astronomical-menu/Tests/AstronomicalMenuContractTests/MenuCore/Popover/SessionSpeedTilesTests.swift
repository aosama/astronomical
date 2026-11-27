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
    XCTAssertEqual(tilesPresentation.caption, "Average this session · 12 requests")
  }

  func test_should_keep_showing_the_session_averages_while_a_request_runs() throws {
    // The daemon's averages already include the running request, so a tile never switches to the
    // request's own speed: 4,096 tokens in 2 s would be 2,048 tok/s, but the tile shows the session.
    let reading = SessionSpeedTilesPresentation(
      statusDocument: try statusDocument(
        activity: "prompt_processing",
        progressJSON: "{\"phase\":\"target\",\"processed_tokens\":4096,\"total_tokens\":10179,\"elapsed_ms\":2000}"))
    XCTAssertEqual(reading.readingTile.speedTitle, "2,290 tok/s")
    XCTAssertEqual(reading.writingTile.speedTitle, "63 tok/s")
    XCTAssertEqual(reading.caption, "Average this session · 12 requests, 1 running")

    let writing = SessionSpeedTilesPresentation(
      statusDocument: try statusDocument(activity: "generating", requestCount: 1))
    XCTAssertEqual(writing.caption, "Average this session · 1 request, 1 running")
  }

  func test_should_show_the_first_running_request_as_soon_as_it_is_measured() throws {
    let firstRequest = SessionSpeedTilesPresentation(
      statusDocument: try statusDocument(
        activity: "generating", requestCount: 0, prefillSpeed: 1500, generationSpeed: 40))
    XCTAssertEqual(firstRequest.readingTile.speedTitle, "1,500 tok/s")
    XCTAssertEqual(firstRequest.writingTile.speedTitle, "40 tok/s")
    XCTAssertEqual(firstRequest.caption, "Average this session · 1 running")
  }

  func test_should_show_a_dash_not_zero_before_a_phase_has_been_measured() throws {
    let nothingMeasured = SessionSpeedTilesPresentation(
      statusDocument: try statusDocument(requestCount: 0, prefillSpeed: 0, generationSpeed: 0))
    XCTAssertEqual(nothingMeasured.readingTile.speedTitle, "—")
    XCTAssertEqual(nothingMeasured.writingTile.speedTitle, "—")
    XCTAssertEqual(nothingMeasured.caption, "Averages appear after the first request")
    XCTAssertEqual(
      SessionSpeedTilesPresentation(statusDocument: .unavailable).readingTile.speedTitle, "—")
    let justStarted = SessionSpeedTilesPresentation(
      statusDocument: try statusDocument(
        activity: "prompt_processing", requestCount: 0, prefillSpeed: 0, generationSpeed: 0))
    XCTAssertEqual(justStarted.readingTile.speedTitle, "—")
  }

  func test_should_name_prefill_and_decode_in_the_explanations_and_use_singular_grammar() throws {
    let oneRequest = SessionSpeedTilesPresentation(statusDocument: try statusDocument(requestCount: 1))
    XCTAssertEqual(oneRequest.caption, "Average this session · 1 request")
    XCTAssertTrue(oneRequest.readingTile.explanation.hasPrefix("Prefill:"))
    XCTAssertTrue(oneRequest.writingTile.explanation.hasPrefix("Decode:"))
    XCTAssertEqual(sessionSpeedTitle(tokensPerSecond: 999.5), "1,000 tok/s")
  }

  func test_should_refresh_the_shown_tiles_at_most_once_a_second() throws {
    var tilesHold = SessionSpeedTilesHold()
    let first = SessionSpeedTilesPresentation(statusDocument: try statusDocument(generationSpeed: 40))
    let second = SessionSpeedTilesPresentation(statusDocument: try statusDocument(generationSpeed: 41))
    let third = SessionSpeedTilesPresentation(statusDocument: try statusDocument(generationSpeed: 42))
    let start = ContinuousClock.now

    XCTAssertEqual(tilesHold.presentation(for: first, at: start), first, "the first figures show at once")
    XCTAssertEqual(
      tilesHold.presentation(for: second, at: start.advanced(by: .milliseconds(250))), first)
    XCTAssertEqual(
      tilesHold.presentation(for: second, at: start.advanced(by: .milliseconds(999))), first)
    XCTAssertEqual(
      tilesHold.presentation(for: third, at: start.advanced(by: .seconds(1))), third,
      "a second later the latest figures show")
  }

  func test_should_show_the_latest_tiles_at_once_when_asked() throws {
    var tilesHold = SessionSpeedTilesHold()
    let first = SessionSpeedTilesPresentation(statusDocument: try statusDocument(generationSpeed: 40))
    let latest = SessionSpeedTilesPresentation(statusDocument: try statusDocument(generationSpeed: 55))
    let start = ContinuousClock.now
    _ = tilesHold.presentation(for: first, at: start)
    XCTAssertEqual(
      tilesHold.presentation(
        for: latest, at: start.advanced(by: .milliseconds(100)), showsAtOnce: true), latest,
      "an opened popover never shows figures held from before it closed")
  }
}
