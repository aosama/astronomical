import Foundation
import XCTest

@testable import AstronomicalMenu

final class SupervisorClientContractTests: XCTestCase {
  func test_should_build_observatory_and_status_urls_from_the_same_local_supervisor_origin() throws {
    let observatoryURL = try localSupervisorEndpointURL(path: "/")
    let statusURL = try localSupervisorEndpointURL(path: "/v1/status")

    XCTAssertEqual(observatoryURL.absoluteString, "http://127.0.0.1:6732/")
    XCTAssertEqual(statusURL.absoluteString, "http://127.0.0.1:6732/v1/status")
  }

  func test_should_decode_a_worker_applied_configuration_reload_response() async throws {
    StubSupervisorURLProtocol.responseConfiguration = .init(
      statusCode: 202,
      responseBody: Data(
        #"{"status":"reloaded","message":"Config reloaded and applied by the worker","worker_restart_completed":true,"worker_runtime_feature_configuration":{"persistent_prompt_cache_enabled":true,"mtp_enabled":true,"speculative_prefill_enabled":false}}"#.utf8)
    )
    defer { StubSupervisorURLProtocol.responseConfiguration = nil }

    let localSupervisorClient = LocalSupervisorClient(
      urlSession: URLSession(configuration: StubSupervisorURLProtocol.urlSessionConfiguration())
    )

    let configurationReloadResult = try await localSupervisorClient.reloadConfiguration()

    XCTAssertEqual(configurationReloadResult.message, "Config reloaded and applied by the worker")
    XCTAssertTrue(configurationReloadResult.workerRestartCompleted)
    XCTAssertEqual(
      configurationReloadResult.workerRuntimeFeatureConfiguration?.speculativePrefillEnabled,
      false
    )
  }

  func test_should_preserve_the_server_message_when_configuration_reload_is_rejected() async {
    StubSupervisorURLProtocol.responseConfiguration = .init(
      statusCode: 409,
      responseBody: Data(
        #"{"status":"busy","message":"A generation is active or queued; reload aborted"}"#.utf8)
    )
    defer { StubSupervisorURLProtocol.responseConfiguration = nil }

    let localSupervisorClient = LocalSupervisorClient(
      urlSession: URLSession(configuration: StubSupervisorURLProtocol.urlSessionConfiguration())
    )

    do {
      _ = try await localSupervisorClient.reloadConfiguration()
      XCTFail("A rejected reload must throw")
    } catch {
      XCTAssertEqual(error.localizedDescription, "A generation is active or queued; reload aborted")
    }
  }

  func test_should_present_invalid_configuration_feedback_when_reload_is_rejected() async {
    StubSupervisorURLProtocol.responseConfiguration = .init(
      statusCode: 400,
      responseBody: Data(#"{"status":"invalid_config","message":"invalid Astronomical configuration"}"#.utf8)
    )
    defer { StubSupervisorURLProtocol.responseConfiguration = nil }

    let localSupervisorClient = LocalSupervisorClient(
      urlSession: URLSession(configuration: StubSupervisorURLProtocol.urlSessionConfiguration())
    )

    do {
      _ = try await localSupervisorClient.reloadConfiguration()
      XCTFail("An invalid configuration reload must throw")
    } catch {
      XCTAssertEqual(error.localizedDescription, "invalid Astronomical configuration")
    }
  }

  func test_should_allow_a_live_memory_update_to_use_the_bounded_reclamation_timeout() async throws {
    StubSupervisorURLProtocol.responseConfiguration = .init(
      statusCode: 200,
      responseBody: Data(#"{"message":"MLX memory setting persisted and applied"}"#.utf8)
    )
    StubSupervisorURLProtocol.receivedRequestTimeoutInterval = nil
    defer {
      StubSupervisorURLProtocol.responseConfiguration = nil
      StubSupervisorURLProtocol.receivedRequestTimeoutInterval = nil
    }
    let localSupervisorClient = LocalSupervisorClient(
      urlSession: URLSession(configuration: StubSupervisorURLProtocol.urlSessionConfiguration())
    )

    _ = try await localSupervisorClient.updateMaximumMlxMemoryGigabytes(32)

    XCTAssertEqual(StubSupervisorURLProtocol.receivedRequestTimeoutInterval, 120)
  }
}

private final class StubSupervisorURLProtocol: URLProtocol, @unchecked Sendable {
  struct ResponseConfiguration {
    let statusCode: Int
    let responseBody: Data
  }

  nonisolated(unsafe) static var responseConfiguration: ResponseConfiguration?
  nonisolated(unsafe) static var receivedRequestTimeoutInterval: TimeInterval?

  override class func canInit(with request: URLRequest) -> Bool { true }

  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

  override func startLoading() {
    Self.receivedRequestTimeoutInterval = request.timeoutInterval
    guard let responseConfiguration = Self.responseConfiguration,
      let requestURL = request.url,
      let urlProtocolClient = client
    else {
      return
    }
    let response = HTTPURLResponse(
      url: requestURL,
      statusCode: responseConfiguration.statusCode,
      httpVersion: "HTTP/1.1",
      headerFields: ["Content-Type": "application/json"]
    )
    guard let response else { return }
    urlProtocolClient.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
    urlProtocolClient.urlProtocol(self, didLoad: responseConfiguration.responseBody)
    urlProtocolClient.urlProtocolDidFinishLoading(self)
  }

  override func stopLoading() {}

  static func urlSessionConfiguration() -> URLSessionConfiguration {
    let urlSessionConfiguration = URLSessionConfiguration.ephemeral
    urlSessionConfiguration.protocolClasses = [StubSupervisorURLProtocol.self]
    return urlSessionConfiguration
  }
}
