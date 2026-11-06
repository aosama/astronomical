import XCTest

@testable import AstronomicalMenuCore

final class NeuralEngineStatusTests: XCTestCase {
  private let preparingJSON = """
    {"state":"preparing","completed_program_count":12,"total_program_count":56,\
    "elapsed_ms":12000,"estimated_remaining_ms":44000}
    """
  private let activeJSON = """
    {"state":"active","layer_program_count":32,"neural_engine_tokens_per_chunk":960,\
    "chunk_token_count":2048,"estimated_memory_bytes":1}
    """
  private let unavailableJSON = #"{"state":"unavailable","reason":"too small"}"#
  private let offJSON = #"{"state":"off"}"#
  private let startedAt = Date(timeIntervalSinceReferenceDate: 1_000)

  private func statusDocument(
    prefillJSON: String, status: String = "ready", activity: String = "idle",
    modelIdentifier: String = "first-model"
  ) throws -> DaemonStatusDocument {
    try JSONDecoder().decode(
      DaemonStatusDocument.self,
      from: Data(
        #"{"status":"\#(status)","activity":"\#(activity)","ready_model_id":"\#(modelIdentifier)","neural_engine_prefill":\#(prefillJSON)}"#
          .utf8))
  }

  private func after(_ seconds: TimeInterval) -> Date { startedAt.addingTimeInterval(seconds) }

  func test_should_count_programs_in_the_menu_bar_while_idle() throws {
    let statusDocument = try statusDocument(prefillJSON: preparingJSON)

    XCTAssertEqual(statusDocument.menuBarTitle, "Neural Engine 12/56")
    XCTAssertEqual(statusDocument.neuralEnginePrefill?.metricTitle(showsEstimate: true), "Preparing 12/56 · about 44 s")
  }

  func test_should_leave_the_menu_bar_to_a_running_generation() throws {
    let statusDocument = try statusDocument(prefillJSON: preparingJSON, activity: "generating")

    XCTAssertFalse(statusDocument.menuBarTitle.contains("Neural Engine"))
  }

  func test_should_not_show_a_count_once_the_programs_are_settled() throws {
    XCTAssertEqual(try statusDocument(prefillJSON: activeJSON).menuBarTitle, "")
    XCTAssertEqual(try statusDocument(prefillJSON: unavailableJSON).menuBarTitle, "")
  }

  func test_should_title_each_settled_state_in_the_popover() throws {
    XCTAssertEqual(try statusDocument(prefillJSON: activeJSON).neuralEnginePrefill?.metricTitle(showsEstimate: true), "Engaged")
    XCTAssertEqual(
      try statusDocument(prefillJSON: unavailableJSON).neuralEnginePrefill?.metricTitle(showsEstimate: true), "On the GPU")
    XCTAssertNil(try statusDocument(prefillJSON: offJSON).neuralEnginePrefill?.metricTitle(showsEstimate: true))
  }

  // The state is a string on the wire so a newer daemon cannot blank the whole menu.
  func test_should_show_nothing_for_an_unknown_state() throws {
    let statusDocument = try statusDocument(prefillJSON: #"{"state":"warming"}"#)

    XCTAssertEqual(statusDocument.menuBarTitle, "")
    XCTAssertNil(statusDocument.neuralEnginePrefill?.metricTitle(showsEstimate: true))
  }

  func test_should_announce_preparation_only_after_the_delay_and_only_once() throws {
    var policy = NeuralEngineBannerPolicy()
    let preparing = try statusDocument(prefillJSON: preparingJSON)

    XCTAssertNil(policy.banner(for: preparing, at: startedAt))
    XCTAssertNil(policy.banner(for: preparing, at: after(4.9)))
    XCTAssertEqual(
      policy.banner(for: preparing, at: after(5))?.title, "Prepping Apple Neural Engine")
    XCTAssertNil(policy.banner(for: preparing, at: after(30)))
  }

  func test_should_promise_the_second_banner_in_the_first() throws {
    var policy = NeuralEngineBannerPolicy()
    let preparing = try statusDocument(prefillJSON: preparingJSON)
    _ = policy.banner(for: preparing, at: startedAt)

    let body = try XCTUnwrap(policy.banner(for: preparing, at: after(5))?.body)

    XCTAssertTrue(body.contains("Will notify you when engaged"), body)
  }

  func test_should_announce_engagement_after_a_long_preparation() throws {
    var policy = NeuralEngineBannerPolicy()
    _ = policy.banner(for: try statusDocument(prefillJSON: preparingJSON), at: startedAt)

    let banner = policy.banner(for: try statusDocument(prefillJSON: activeJSON), at: after(60))

    XCTAssertEqual(banner?.title, "Apple Neural Engine ready and engaged")
    XCTAssertEqual(
      banner?.body, "first-model now prefills on the Neural Engine and the GPU.")
  }

  func test_should_only_announce_engagement_after_a_short_preparation() throws {
    var policy = NeuralEngineBannerPolicy()
    XCTAssertNil(policy.banner(for: try statusDocument(prefillJSON: preparingJSON), at: startedAt))

    let banner = policy.banner(for: try statusDocument(prefillJSON: activeJSON), at: after(2))

    XCTAssertEqual(banner?.title, "Apple Neural Engine ready and engaged")
  }

  func test_should_announce_a_new_model_engaged_between_two_polls() throws {
    var policy = NeuralEngineBannerPolicy()
    _ = policy.banner(for: try statusDocument(prefillJSON: activeJSON), at: startedAt)

    let banner = policy.banner(
      for: try statusDocument(prefillJSON: activeJSON, modelIdentifier: "second-model"),
      at: after(5))

    XCTAssertEqual(banner?.title, "Apple Neural Engine ready and engaged")
  }

  func test_should_stay_quiet_about_a_model_already_engaged_when_the_app_starts() throws {
    var policy = NeuralEngineBannerPolicy()
    let active = try statusDocument(prefillJSON: activeJSON)

    XCTAssertNil(policy.banner(for: active, at: startedAt))
    XCTAssertNil(policy.banner(for: active, at: after(5)))
  }

  func test_should_report_a_preparation_that_ends_unavailable() throws {
    var policy = NeuralEngineBannerPolicy()
    _ = policy.banner(for: try statusDocument(prefillJSON: preparingJSON), at: startedAt)

    let banner = policy.banner(for: try statusDocument(prefillJSON: unavailableJSON), at: after(9))

    XCTAssertEqual(banner?.title, "Apple Neural Engine unavailable")
  }

  func test_should_stay_quiet_about_a_model_refused_at_load() throws {
    var policy = NeuralEngineBannerPolicy()

    XCTAssertNil(policy.banner(for: try statusDocument(prefillJSON: unavailableJSON), at: startedAt))
  }

  func test_should_stay_quiet_when_the_setting_is_off() throws {
    var policy = NeuralEngineBannerPolicy()
    let off = try statusDocument(prefillJSON: offJSON)

    XCTAssertNil(policy.banner(for: off, at: startedAt))
    XCTAssertNil(policy.banner(for: off, at: after(60)))
  }

  func test_should_keep_the_wait_through_a_missed_poll() throws {
    var policy = NeuralEngineBannerPolicy()
    _ = policy.banner(for: try statusDocument(prefillJSON: preparingJSON), at: startedAt)

    XCTAssertNil(policy.banner(for: DaemonStatusDocument.unavailable, at: after(2)))

    XCTAssertEqual(
      policy.banner(for: try statusDocument(prefillJSON: preparingJSON), at: after(5))?.title,
      "Prepping Apple Neural Engine")
  }

  func test_should_restart_the_wait_for_a_different_model() throws {
    var policy = NeuralEngineBannerPolicy()
    _ = policy.banner(for: try statusDocument(prefillJSON: preparingJSON), at: startedAt)

    let secondModel = try statusDocument(prefillJSON: preparingJSON, modelIdentifier: "second-model")

    XCTAssertNil(policy.banner(for: secondModel, at: after(5)))
    XCTAssertEqual(
      policy.banner(for: secondModel, at: after(10))?.title, "Prepping Apple Neural Engine")
  }

  func test_should_hide_the_estimate_while_a_request_runs() throws {
    let prefill = try XCTUnwrap(statusDocument(prefillJSON: preparingJSON).neuralEnginePrefill)

    XCTAssertEqual(prefill.metricTitle(showsEstimate: false), "Preparing 12/56")
  }

  // During a swap the daemon still names the previous model, which is already gone.
  func test_should_say_nothing_about_a_model_being_replaced() throws {
    var policy = NeuralEngineBannerPolicy()
    _ = policy.banner(for: try statusDocument(prefillJSON: preparingJSON), at: startedAt)
    let swapping = try JSONDecoder().decode(
      DaemonStatusDocument.self,
      from: Data(
        #"""
        {"status":"ready","activity":"idle","ready_model_id":"first-model",
        "model_load_in_flight":{"model_id":"second-model"},
        "neural_engine_prefill":{"state":"preparing","completed_program_count":1,
        "total_program_count":2,"elapsed_ms":1,"estimated_remaining_ms":null}}
        """#.utf8))

    XCTAssertNil(policy.banner(for: swapping, at: after(5)))
    XCTAssertNil(policy.banner(for: try statusDocument(prefillJSON: preparingJSON), at: after(6)))
  }
}
