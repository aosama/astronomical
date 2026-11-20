import XCTest

@testable import AstronomicalMenuCore

final class MenuBarPresentationTests: XCTestCase {
  private func statusDocument(
    activity: String, progressJSON: String?, showsProgressAndSpeed: Bool = false
  ) throws -> DaemonStatusDocument {
    let servingSession =
      "{\"completed_request_count\":0,\"total_prompt_token_count\":0,\"total_reused_prompt_token_count\":0,\"target_prompt_work_token_count\":0,\"target_reused_prompt_work_token_count\":0,\"average_prefill_tok_per_second\":0,\"average_generation_tok_per_second\":0}"
    let progressField = progressJSON.map { ",\"progress\":\($0)" } ?? ""
    let statusJSON =
      "{\"status\":\"ready\",\"activity\":\"\(activity)\",\"ready_model_id\":\"qwen-35b\",\"menu_bar_shows_progress_and_speed\":\(showsProgressAndSpeed)\(progressField),\"serving_session\":\(servingSession)}"
    return try JSONDecoder().decode(DaemonStatusDocument.self, from: Data(statusJSON.utf8))
  }

  private func generatingPresentation(showsProgressAndSpeed: Bool = false) throws
    -> MenuBarPresentation
  {
    MenuBarPresentation(
      statusDocument: try statusDocument(
        activity: "generating",
        progressJSON:
          "{\"phase\":\"generation\",\"processed_tokens\":688,\"total_tokens\":65535,\"elapsed_ms\":10000}",
        showsProgressAndSpeed: showsProgressAndSpeed),
      engineIsDown: false)
  }

  func test_should_show_only_the_icon_unless_the_person_asked_for_text() throws {
    XCTAssertEqual(
      menuBarTitleToShow(
        presentation: try generatingPresentation(), isDevelopmentChannel: false), "")
  }

  func test_should_show_a_whole_number_speed_when_the_person_asked_for_text() throws {
    XCTAssertEqual(
      menuBarTitleToShow(
        presentation: try generatingPresentation(showsProgressAndSpeed: true),
        isDevelopmentChannel: false), " 69 tok/s")
  }

  func test_should_show_no_text_when_idle_even_with_text_switched_on() throws {
    let idlePresentation = MenuBarPresentation(
      statusDocument: try statusDocument(
        activity: "idle", progressJSON: nil, showsProgressAndSpeed: true), engineIsDown: false)
    XCTAssertEqual(
      menuBarTitleToShow(presentation: idlePresentation, isDevelopmentChannel: false), "")
  }

  func test_should_always_label_the_development_build() throws {
    XCTAssertEqual(
      menuBarTitleToShow(
        presentation: try generatingPresentation(), isDevelopmentChannel: true), " DEV")
    XCTAssertEqual(
      menuBarTitleToShow(
        presentation: try generatingPresentation(showsProgressAndSpeed: true),
        isDevelopmentChannel: true), " DEV · 69 tok/s")
  }

  func test_should_name_a_stopped_engine_although_the_text_switch_is_off() {
    let stoppedPresentation = MenuBarPresentation(statusDocument: .unavailable, engineIsDown: true)
    XCTAssertEqual(
      menuBarTitleToShow(presentation: stoppedPresentation, isDevelopmentChannel: false),
      " Not running")
    XCTAssertEqual(stoppedPresentation.glyph, MenuBarGlyph(state: .engineNotRunning))
    XCTAssertEqual(
      stoppedPresentation.toolTip(buildTitle: "Astronomical 1.0"),
      "The engine is not running. Click, then choose Restart server.\nAstronomical 1.0")
  }

  func test_should_pad_short_counts_to_a_steady_width() {
    XCTAssertEqual(steadyWidthCount(7, minimumDigitCount: 2), "\u{2007}7")
    XCTAssertEqual(steadyWidthCount(68, minimumDigitCount: 2), "68")
    XCTAssertEqual(steadyWidthCount(1076, minimumDigitCount: 3), "1076")
    XCTAssertEqual(steadyWidthCount(0, minimumDigitCount: 0), "0")
  }

  func test_should_say_what_is_happening_in_the_tooltip() throws {
    XCTAssertEqual(
      try statusDocument(activity: "idle", progressJSON: nil)
        .menuBarStateSentence(engineIsDown: false), "qwen-35b · All in RAM")
    XCTAssertEqual(
      try generatingPresentation().stateSentence, "Generating · 68.8 tok/s")
    XCTAssertEqual(
      DaemonStatusDocument.unavailable.menuBarStateSentence(engineIsDown: false), "Starting")
    XCTAssertEqual(MenuBarPresentation.launching.glyph, MenuBarGlyph(state: .idle))
  }
}
