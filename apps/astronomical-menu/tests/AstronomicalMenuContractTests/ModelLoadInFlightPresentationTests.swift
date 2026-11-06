import XCTest

@testable import AstronomicalMenuCore

final class ModelLoadInFlightPresentationTests: XCTestCase {
  private func statusDocument(
    status: String = "ready", activity: String = "idle", modelLoadInFlightJSON: String
  ) throws -> DaemonStatusDocument {
    try JSONDecoder().decode(
      DaemonStatusDocument.self,
      from: Data(
        #"{"status":"\#(status)","activity":"\#(activity)","model_load_in_flight":\#(modelLoadInFlightJSON)}"#
          .utf8)
    )
  }

  private let namedLoadJSON = #"{"model_id":"second-model"}"#
  private let unnamedLoadJSON = #"{"model_id":null}"#

  // A model swap keeps reporting "ready" with "idle" activity, which used to render as nothing at
  // all: the reported silence this presentation exists to end.
  func test_should_describe_a_load_while_the_status_still_reads_ready() throws {
    let statusDocument = try statusDocument(modelLoadInFlightJSON: namedLoadJSON)

    XCTAssertEqual(statusDocument.menuBarTitle, "Loading model")
    XCTAssertEqual(statusDocument.phaseTitle, "Loading")
    XCTAssertEqual(statusDocument.flightTitle, "Loading the model")
    XCTAssertEqual(statusDocument.progressTitle, "Loading the model")
    XCTAssertEqual(statusDocument.elapsedTimeMetricTitle, "Elapsed")
    XCTAssertEqual(statusDocument.elapsedTimeTitle, "In progress")
    XCTAssertFalse(statusDocument.hasDeterminateProgress)
  }

  // During a swap the daemon still names the model the worker has already dropped.
  func test_should_stop_describing_the_previous_model_during_a_swap() throws {
    let statusDocument = try JSONDecoder().decode(
      DaemonStatusDocument.self,
      from: Data(
        #"""
        {"status":"ready","activity":"idle","ready_model_id":"first-model",
        "ready_model_size_bytes":1073741824,"model_load_in_flight":{"model_id":"second-model"}}
        """#.utf8)
    )

    XCTAssertEqual(statusDocument.modelLoadInFlight?.headlineTitle, "Loading second-model")
    XCTAssertEqual(statusDocument.modelFootprintTitle, "Being prepared")
    XCTAssertEqual(statusDocument.modelDiskSizeTitle, "Being prepared")
    XCTAssertEqual(statusDocument.replyReadTitle, "")
  }

  func test_should_name_a_load_generically_when_the_model_is_not_reported() throws {
    let statusDocument = try statusDocument(modelLoadInFlightJSON: unnamedLoadJSON)

    XCTAssertEqual(statusDocument.modelLoadInFlight?.headlineTitle, "Loading a model")
  }

  func test_should_treat_a_document_without_the_field_as_no_load() throws {
    let statusDocument = try JSONDecoder().decode(
      DaemonStatusDocument.self, from: Data(#"{"status":"ready","activity":"idle"}"#.utf8))

    XCTAssertFalse(statusDocument.isLoadingModel)
  }

  func test_should_fall_back_to_the_usual_titles_with_no_load_in_flight() throws {
    let statusDocument = try statusDocument(modelLoadInFlightJSON: "null")

    XCTAssertNil(statusDocument.modelLoadInFlight)
    XCTAssertEqual(statusDocument.menuBarTitle, "")
    XCTAssertEqual(statusDocument.flightTitle, "Standing by")
    XCTAssertEqual(statusDocument.phaseTitle, "Ready")
  }

  func test_should_not_treat_a_load_as_an_active_request() throws {
    let statusDocument = try statusDocument(modelLoadInFlightJSON: namedLoadJSON)

    XCTAssertFalse(statusDocument.isActive)
  }
}
