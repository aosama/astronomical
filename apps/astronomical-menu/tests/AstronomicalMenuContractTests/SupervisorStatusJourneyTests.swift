// Proves complete Rust-produced status documents survive the production Swift client, telemetry
// store, and loaded-model family boundary before the menu renders them.

import Foundation
import XCTest

@testable import AstronomicalMenuCore

final class SupervisorStatusJourneyTests: XCTestCase {
  @MainActor
  func test_should_present_a_complete_autoregressive_status_response_in_the_menu_store() async throws {
    StubSupervisorURLProtocol.statusResponseConfiguration = .init(
      statusCode: 200,
      responseBody: try fixtureData(named: "full-autoregressive-status")
    )
    defer { StubSupervisorURLProtocol.statusResponseConfiguration = nil }
    let telemetryStore = TelemetryStore(supervisorClient: localDevelopmentSupervisorClient())

    await telemetryStore.refresh()

    XCTAssertEqual(telemetryStore.statusDocument.status, "ready")
    XCTAssertEqual(
      telemetryStore.statusDocument.readyModelIdentifier,
      "fictional/autoregressive-model")
    let loadedModelConfiguration = telemetryStore.statusDocument
      .workerRuntimeFeatureConfiguration?.loadedModel?.autoregressiveConfiguration
    XCTAssertEqual(loadedModelConfiguration?.modelIdentifier, "fictional/autoregressive-model")
    XCTAssertEqual(loadedModelConfiguration?.maximumContextTokens, 65_536)
    XCTAssertNil(telemetryStore.lastStatusRefreshErrorMessage)
  }

  @MainActor
  func test_should_present_a_complete_flux_status_response_in_the_menu_store() async throws {
    StubSupervisorURLProtocol.statusResponseConfiguration = .init(
      statusCode: 200,
      responseBody: try fixtureData(named: "full-flux-status")
    )
    defer { StubSupervisorURLProtocol.statusResponseConfiguration = nil }
    let telemetryStore = TelemetryStore(supervisorClient: localDevelopmentSupervisorClient())

    await telemetryStore.refresh()

    guard case let .flux2Klein(imageConfiguration) = telemetryStore.statusDocument
      .workerRuntimeFeatureConfiguration?.loadedModel
    else {
      return XCTFail("The tagged image configuration must retain its model family")
    }
    XCTAssertEqual(imageConfiguration.modelIdentifier, "fictional/image-model")
    XCTAssertEqual(imageConfiguration.modelFamily, .flux2Klein)
    XCTAssertNil(telemetryStore.lastStatusRefreshErrorMessage)
  }

  @MainActor
  func test_should_diagnose_worker_policy_drift_and_recover_on_the_next_valid_status() async throws {
    let validStatusResponse = try fixtureData(named: "full-autoregressive-status")
    var malformedStatusFixture = try jsonObject(from: validStatusResponse)
    var runtimeConfiguration = try XCTUnwrap(
      malformedStatusFixture["worker_runtime_feature_configuration"] as? [String: Any])
    var loadedModel = try XCTUnwrap(runtimeConfiguration["loaded_model"] as? [String: Any])
    var loadedModelConfiguration = try XCTUnwrap(
      loadedModel["configuration"] as? [String: Any])
    loadedModelConfiguration["unexpected_execution_policy"] = true
    loadedModel["configuration"] = loadedModelConfiguration
    runtimeConfiguration["loaded_model"] = loadedModel
    malformedStatusFixture["worker_runtime_feature_configuration"] = runtimeConfiguration
    StubSupervisorURLProtocol.statusResponseConfiguration = .init(
      statusCode: 200,
      responseBody: try JSONSerialization.data(withJSONObject: malformedStatusFixture)
    )
    defer { StubSupervisorURLProtocol.statusResponseConfiguration = nil }
    let telemetryStore = TelemetryStore(supervisorClient: localDevelopmentSupervisorClient())

    await telemetryStore.refresh()

    XCTAssertEqual(telemetryStore.statusDocument.status, "unavailable")
    XCTAssertTrue(
      telemetryStore.lastStatusRefreshErrorMessage?.contains("unexpected_execution_policy") == true)

    StubSupervisorURLProtocol.statusResponseConfiguration = .init(
      statusCode: 200,
      responseBody: validStatusResponse
    )
    await telemetryStore.refresh()

    XCTAssertEqual(telemetryStore.statusDocument.status, "ready")
    XCTAssertNil(telemetryStore.lastStatusRefreshErrorMessage)
  }

  @MainActor
  func test_should_reject_null_for_a_worker_policy_field_that_rust_omits() async throws {
    let validStatusResponse = try fixtureData(named: "full-autoregressive-status")
    var malformedStatusFixture = try jsonObject(from: validStatusResponse)
    var runtimeConfiguration = try XCTUnwrap(
      malformedStatusFixture["worker_runtime_feature_configuration"] as? [String: Any])
    var loadedModel = try XCTUnwrap(runtimeConfiguration["loaded_model"] as? [String: Any])
    var loadedModelConfiguration = try XCTUnwrap(
      loadedModel["configuration"] as? [String: Any])
    var chunkingConfiguration = try XCTUnwrap(
      loadedModelConfiguration["chunking"] as? [String: Any])
    chunkingConfiguration["fixed_ssd_streaming_prompt_processing_chunk_size_tokens"] = NSNull()
    loadedModelConfiguration["chunking"] = chunkingConfiguration
    loadedModel["configuration"] = loadedModelConfiguration
    runtimeConfiguration["loaded_model"] = loadedModel
    malformedStatusFixture["worker_runtime_feature_configuration"] = runtimeConfiguration
    StubSupervisorURLProtocol.statusResponseConfiguration = .init(
      statusCode: 200,
      responseBody: try JSONSerialization.data(withJSONObject: malformedStatusFixture)
    )
    defer { StubSupervisorURLProtocol.statusResponseConfiguration = nil }
    let telemetryStore = TelemetryStore(supervisorClient: localDevelopmentSupervisorClient())

    await telemetryStore.refresh()

    XCTAssertEqual(telemetryStore.statusDocument.status, "unavailable")
    XCTAssertTrue(
      telemetryStore.lastStatusRefreshErrorMessage?.contains(
        "fixed_ssd_streaming_prompt_processing_chunk_size_tokens") == true)
  }

  @MainActor
  func test_should_not_let_an_older_failure_replace_a_newer_success() async throws {
    let validStatusDocument = try decodedFixtureStatusDocument()
    let supervisorClient = OutOfOrderStatusSupervisorClient(
      successfulStatusDocument: validStatusDocument,
      firstRequestFails: true
    )
    let telemetryStore = TelemetryStore(supervisorClient: supervisorClient)
    let delayedRefreshTask = Task { _ = await telemetryStore.refresh() }
    await supervisorClient.waitUntilFirstRequestIsSuspended()

    let latestRefreshTask = Task { _ = await telemetryStore.refresh() }
    await Task.yield()
    await supervisorClient.resumeFirstRequest()
    await delayedRefreshTask.value
    await latestRefreshTask.value

    XCTAssertEqual(telemetryStore.statusDocument.status, "ready")
    XCTAssertNil(telemetryStore.lastStatusRefreshErrorMessage)
  }

  @MainActor
  func test_should_not_let_an_older_success_clear_a_newer_failure() async throws {
    let validStatusDocument = try decodedFixtureStatusDocument()
    let supervisorClient = OutOfOrderStatusSupervisorClient(
      successfulStatusDocument: validStatusDocument,
      firstRequestFails: false
    )
    let telemetryStore = TelemetryStore(supervisorClient: supervisorClient)
    let delayedRefreshTask = Task { _ = await telemetryStore.refresh() }
    await supervisorClient.waitUntilFirstRequestIsSuspended()

    let latestRefreshTask = Task { _ = await telemetryStore.refresh() }
    await Task.yield()
    await supervisorClient.resumeFirstRequest()
    await delayedRefreshTask.value
    await latestRefreshTask.value

    XCTAssertEqual(telemetryStore.statusDocument.status, "unavailable")
    XCTAssertTrue(
      telemetryStore.lastStatusRefreshErrorMessage?.contains("Controlled status failure") == true)
  }

  private func localDevelopmentSupervisorClient() -> LocalSupervisorClient {
    LocalSupervisorClient(
      applicationIdentity: ApplicationIdentity(
        channel: .development,
        supervisorPort: ApplicationChannel.development.defaultSupervisorPort,
        stateDirectoryName: ApplicationChannel.development.stateDirectoryName,
        version: "0.3.0",
        buildNumber: "300",
        commit: "abcdef1",
        isDirty: true
      ),
      urlSession: URLSession(configuration: StubSupervisorURLProtocol.urlSessionConfiguration())
    )
  }

  private func fixtureData(named fixtureName: String) throws -> Data {
    let fixtureURL = try XCTUnwrap(
      Bundle.module.url(forResource: fixtureName, withExtension: "json"))
    return try Data(contentsOf: fixtureURL)
  }

  private func decodedFixtureStatusDocument() throws -> SupervisorStatusDocument {
    try JSONDecoder().decode(
      SupervisorStatusDocument.self,
      from: fixtureData(named: "full-autoregressive-status")
    )
  }

  private func jsonObject(from jsonData: Data) throws -> [String: Any] {
    try XCTUnwrap(JSONSerialization.jsonObject(with: jsonData) as? [String: Any])
  }
}

private actor OutOfOrderStatusSupervisorClient: SupervisorClient {
  private let successfulStatusDocument: SupervisorStatusDocument
  private let firstRequestFails: Bool
  private var requestCount = 0
  private var firstRequestContinuation: CheckedContinuation<Void, Never>?

  init(successfulStatusDocument: SupervisorStatusDocument, firstRequestFails: Bool) {
    self.successfulStatusDocument = successfulStatusDocument
    self.firstRequestFails = firstRequestFails
  }

  func fetchStatus() async throws -> SupervisorStatusDocument {
    requestCount += 1
    let currentRequestCount = requestCount
    if currentRequestCount == 1 {
      await withCheckedContinuation { continuation in
        firstRequestContinuation = continuation
      }
    }
    let requestFails = currentRequestCount == 1 ? firstRequestFails : !firstRequestFails
    guard !requestFails else { throw ControlledStatusFailure.failed }
    return successfulStatusDocument
  }

  func waitUntilFirstRequestIsSuspended() async {
    while firstRequestContinuation == nil { await Task.yield() }
  }

  func resumeFirstRequest() {
    firstRequestContinuation?.resume()
    firstRequestContinuation = nil
  }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    ConfigurationReloadResult(message: "Configuration reloaded")
  }

  func requestShutdown() async throws {}

  func healthIsAvailable() async -> Bool { true }
}

private enum ControlledStatusFailure: LocalizedError {
  case failed

  var errorDescription: String? { "Controlled status failure" }
}
