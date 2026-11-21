import XCTest

@testable import AstronomicalMenuCore

/// The engine reports the loaded model as a kind plus that kind's settings.
/// The agent has dedicated shapes for a few kinds; every other kind must
/// still decode, or the whole status reply is unreadable while it is loaded.
final class LoadedModelKindTests: XCTestCase {
  private func decodeLoadedModel(_ json: String) throws -> WorkerLoadedModelRuntimeConfiguration {
    try JSONDecoder().decode(WorkerLoadedModelRuntimeConfiguration.self, from: Data(json.utf8))
  }

  /// What the engine sent while a speech model was loaded, when the menu bar
  /// said "Not running" for a minute after every spoken sentence.
  func test_should_read_a_loaded_speech_model() throws {
    let loadedModel = try decodeLoadedModel(
      #"{"configuration":{"artifact_revision":"a71e4d38","model_family":"kokoro","model_id":"Kokoro-82M-bf16"},"kind":"speech"}"#
    )
    XCTAssertEqual(loadedModel, .other(kind: "speech", modelIdentifier: "Kokoro-82M-bf16"))
    XCTAssertNil(loadedModel.autoregressiveConfiguration)
  }

  func test_should_read_a_loaded_decision_model() throws {
    let loadedModel = try decodeLoadedModel(
      #"{"configuration":{"model_id":"laya-MLX-8bit","anything_else":[1,2,3]},"kind":"decisions"}"#)
    XCTAssertEqual(loadedModel, .other(kind: "decisions", modelIdentifier: "laya-MLX-8bit"))
  }

  func test_should_read_a_kind_a_newer_engine_adds_even_without_settings() throws {
    XCTAssertEqual(
      try decodeLoadedModel(#"{"kind":"video"}"#), .other(kind: "video", modelIdentifier: nil))
  }

  func test_should_still_refuse_a_loaded_model_without_a_kind() {
    XCTAssertThrowsError(try decodeLoadedModel(#"{"configuration":{"model_id":"x"}}"#))
  }

  func test_should_tell_two_models_of_the_same_other_kind_apart() throws {
    let firstModel = try decodeLoadedModel(#"{"configuration":{"model_id":"a"},"kind":"speech"}"#)
    let secondModel = try decodeLoadedModel(#"{"configuration":{"model_id":"b"},"kind":"speech"}"#)
    XCTAssertNotEqual(firstModel, secondModel)
  }

  func test_should_keep_the_kind_and_model_when_written_back() throws {
    let loadedModel = WorkerLoadedModelRuntimeConfiguration.other(
      kind: "speech", modelIdentifier: "Kokoro-82M-bf16")
    let writtenBytes = try JSONEncoder().encode(loadedModel)
    XCTAssertEqual(
      try JSONDecoder().decode(WorkerLoadedModelRuntimeConfiguration.self, from: writtenBytes),
      loadedModel)
  }

  /// The whole reply, as the agent polls it: an existing full status fixture
  /// with its loaded model swapped for a speech model must decode as ready.
  func test_should_read_a_whole_status_reply_while_a_speech_model_is_loaded() throws {
    let fixtureURL = try XCTUnwrap(
      Bundle.module.url(forResource: "full-embedding-status", withExtension: "json"))
    var statusObject = try XCTUnwrap(
      JSONSerialization.jsonObject(with: Data(contentsOf: fixtureURL)) as? [String: Any])
    var workerConfiguration = try XCTUnwrap(
      statusObject["worker_runtime_feature_configuration"] as? [String: Any])
    workerConfiguration["loaded_model"] = [
      "kind": "speech",
      "configuration": ["model_id": "Kokoro-82M-bf16", "model_family": "kokoro"],
    ]
    statusObject["worker_runtime_feature_configuration"] = workerConfiguration
    let statusBytes = try JSONSerialization.data(withJSONObject: statusObject)
    let statusDocument = try JSONDecoder().decode(DaemonStatusDocument.self, from: statusBytes)
    XCTAssertNotEqual(statusDocument.status, "unavailable")
  }
}
