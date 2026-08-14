import Foundation
import XCTest

@testable import AstronomicalMenu

final class SupervisorClientContractTests: XCTestCase {
  func test_should_keep_stable_and_development_endpoints_and_state_separate() throws {
    let stableIdentity = applicationIdentity(channel: .stable)
    let developmentIdentity = applicationIdentity(channel: .development)
    let fictionalHomeURL = URL(fileURLWithPath: "/Users/example", isDirectory: true)

    XCTAssertEqual(try stableIdentity.endpointURL(path: "/").absoluteString, "http://127.0.0.1:6732/")
    XCTAssertEqual(
      try developmentIdentity.endpointURL(path: "/v1/status").absoluteString,
      "http://127.0.0.1:6733/v1/status")
    XCTAssertEqual(
      stableIdentity.configFileURL(homeDirectoryURL: fictionalHomeURL).path,
      "/Users/example/.astronomical/config.json")
    XCTAssertEqual(
      developmentIdentity.configFileURL(homeDirectoryURL: fictionalHomeURL).path,
      "/Users/example/.astronomical-dev/config.json")
    XCTAssertNotEqual(
      stableIdentity.daemonOwnershipURL(homeDirectoryURL: fictionalHomeURL),
      developmentIdentity.daemonOwnershipURL(homeDirectoryURL: fictionalHomeURL))
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

  func test_should_reject_status_from_the_opposite_runtime_instance() async {
    StubSupervisorURLProtocol.statusResponseConfiguration = .init(
      statusCode: 200,
      responseBody: Data(
        #"{"application":{"version":"0.2.0","build_number":100,"commit":"abcdef1","is_dirty":false,"channel":"stable","channel_display_name":"Stable","state_directory":"~/.astronomical"},"status":"ready","activity":"idle"}"#.utf8)
    )
    defer { StubSupervisorURLProtocol.statusResponseConfiguration = nil }
    let developmentClient = LocalSupervisorClient(
      applicationIdentity: applicationIdentity(channel: .development),
      urlSession: URLSession(configuration: StubSupervisorURLProtocol.urlSessionConfiguration())
    )

    do {
      _ = try await developmentClient.fetchStatus()
      XCTFail("Development must not adopt a Stable server")
    } catch {
      XCTAssertEqual(
        error.localizedDescription,
        "Expected the development server, but the stable instance answered")
    }
  }

  func test_should_reject_an_unidentified_server_for_every_runtime_instance() async {
    StubSupervisorURLProtocol.statusResponseConfiguration = .init(
      statusCode: 200,
      responseBody: Data(#"{"status":"ready","activity":"idle"}"#.utf8)
    )
    defer { StubSupervisorURLProtocol.statusResponseConfiguration = nil }

    for runtimeChannel in [ApplicationChannel.stable, ApplicationChannel.development] {
      let localSupervisorClient = LocalSupervisorClient(
        applicationIdentity: applicationIdentity(channel: runtimeChannel),
        urlSession: URLSession(configuration: StubSupervisorURLProtocol.urlSessionConfiguration())
      )

      do {
        _ = try await localSupervisorClient.fetchStatus()
        XCTFail("Every menu must reject a server that cannot prove its runtime channel")
      } catch {
        XCTAssertEqual(
          error.localizedDescription,
          "Expected the \(runtimeChannel.rawValue) server, but the unidentified instance answered")
      }
    }
  }

  func test_should_report_an_occupied_endpoint_without_trusting_its_instance_identity() async {
    StubSupervisorURLProtocol.responseConfiguration = .init(
      statusCode: 200,
      responseBody: Data(#"{"status":"ok"}"#.utf8)
    )
    defer { StubSupervisorURLProtocol.responseConfiguration = nil }
    let localSupervisorClient = LocalSupervisorClient(
      applicationIdentity: applicationIdentity(channel: .development),
      urlSession: URLSession(configuration: StubSupervisorURLProtocol.urlSessionConfiguration())
    )

    let endpointIsOccupied = await localSupervisorClient.healthIsAvailable()

    XCTAssertTrue(endpointIsOccupied)
    XCTAssertEqual(StubSupervisorURLProtocol.receivedRequestPaths.last, "/health")
  }

  func test_should_refuse_to_control_the_opposite_runtime_instance() async {
    StubSupervisorURLProtocol.statusResponseConfiguration = .init(
      statusCode: 200,
      responseBody: Data(
        #"{"application":{"version":"0.2.0","build_number":100,"commit":"abcdef1","is_dirty":false,"channel":"stable","channel_display_name":"Stable","state_directory":"~/.astronomical"},"status":"ready","activity":"idle"}"#.utf8)
    )
    StubSupervisorURLProtocol.responseConfiguration = .init(statusCode: 202, responseBody: Data())
    StubSupervisorURLProtocol.receivedRequestPaths = []
    defer {
      StubSupervisorURLProtocol.statusResponseConfiguration = nil
      StubSupervisorURLProtocol.responseConfiguration = nil
      StubSupervisorURLProtocol.receivedRequestPaths = []
    }
    let developmentClient = LocalSupervisorClient(
      applicationIdentity: applicationIdentity(channel: .development),
      urlSession: URLSession(configuration: StubSupervisorURLProtocol.urlSessionConfiguration())
    )

    do {
      try await developmentClient.requestShutdown()
      XCTFail("Development must not control a Stable server")
    } catch {
      XCTAssertEqual(
        error.localizedDescription,
        "Expected the development server, but the stable instance answered")
      XCTAssertEqual(StubSupervisorURLProtocol.receivedRequestPaths, ["/v1/status"])
    }
  }

  func test_should_refuse_to_control_same_channel_server_with_unexpected_state() async {
    StubSupervisorURLProtocol.statusResponseConfiguration = .init(
      statusCode: 200,
      responseBody: Data(
        #"{"application":{"version":"0.2.0","build_number":100,"commit":"abcdef1","is_dirty":true,"channel":"development","channel_display_name":"Development","state_directory":"custom"},"status":"ready","activity":"idle"}"#.utf8)
    )
    StubSupervisorURLProtocol.responseConfiguration = .init(statusCode: 202, responseBody: Data())
    StubSupervisorURLProtocol.receivedRequestPaths = []
    defer {
      StubSupervisorURLProtocol.statusResponseConfiguration = nil
      StubSupervisorURLProtocol.responseConfiguration = nil
      StubSupervisorURLProtocol.receivedRequestPaths = []
    }
    let developmentClient = LocalSupervisorClient(
      applicationIdentity: applicationIdentity(channel: .development),
      urlSession: URLSession(configuration: StubSupervisorURLProtocol.urlSessionConfiguration())
    )

    do {
      try await developmentClient.requestShutdown()
      XCTFail("Development must not control a custom-state server")
    } catch {
      XCTAssertEqual(
        error.localizedDescription,
        "Expected server state ~/.astronomical-dev, but the connected instance reported custom")
      XCTAssertEqual(StubSupervisorURLProtocol.receivedRequestPaths, ["/v1/status"])
    }
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

  private func applicationIdentity(channel: ApplicationChannel) -> ApplicationIdentity {
    ApplicationIdentity(
      channel: channel,
      supervisorPort: channel.defaultSupervisorPort,
      stateDirectoryName: channel.stateDirectoryName,
      version: "0.2.0",
      buildNumber: "100",
      commit: "abcdef1",
      isDirty: channel == .development
    )
  }
}

private final class StubSupervisorURLProtocol: URLProtocol, @unchecked Sendable {
  struct ResponseConfiguration {
    let statusCode: Int
    let responseBody: Data
  }

  nonisolated(unsafe) static var responseConfiguration: ResponseConfiguration?
  nonisolated(unsafe) static var statusResponseConfiguration: ResponseConfiguration?
  nonisolated(unsafe) static var receivedRequestTimeoutInterval: TimeInterval?
  nonisolated(unsafe) static var receivedRequestPaths: [String] = []

  override class func canInit(with request: URLRequest) -> Bool { true }

  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

  override func startLoading() {
    Self.receivedRequestTimeoutInterval = request.timeoutInterval
    Self.receivedRequestPaths.append(request.url?.path ?? "")
    let selectedResponseConfiguration =
      request.url?.path == "/v1/status"
      ? Self.statusResponseConfiguration ?? Self.matchingDevelopmentStatusResponse
      : Self.responseConfiguration
    guard let responseConfiguration = selectedResponseConfiguration,
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

  private static let matchingDevelopmentStatusResponse = ResponseConfiguration(
    statusCode: 200,
    responseBody: Data(
      #"{"application":{"version":"0.2.0","build_number":100,"commit":"abcdef1","is_dirty":true,"channel":"development","channel_display_name":"Development","state_directory":"~/.astronomical-dev"},"status":"ready","activity":"idle"}"#.utf8)
  )
}
