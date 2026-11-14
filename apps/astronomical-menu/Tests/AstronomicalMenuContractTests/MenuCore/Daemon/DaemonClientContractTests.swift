import Foundation
import XCTest

@testable import AstronomicalMenuCore

final class DaemonClientContractTests: XCTestCase {
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
    StubDaemonURLProtocol.responseConfiguration = .init(
      statusCode: 202,
      responseBody: Data(
        #"{"status":"reloaded","message":"Config reloaded and applied by the worker","worker_restart_completed":true,"candidate_generation":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","effective_generation":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","worker_runtime_feature_configuration":{"configuration_generation":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","persistent_prompt_cache_enabled":true,"prompt_cache_maximum_size_bytes":50000000000,"loaded_model":null}}"#.utf8)
    )
    defer { StubDaemonURLProtocol.responseConfiguration = nil }

    let localDaemonClient = LocalDaemonClient(
      urlSession: URLSession(configuration: StubDaemonURLProtocol.urlSessionConfiguration())
    )

    let configurationReloadResult = try await localDaemonClient.reloadConfiguration()

    XCTAssertEqual(configurationReloadResult.message, "Config reloaded and applied by the worker")
    XCTAssertTrue(configurationReloadResult.workerRestartCompleted)
    XCTAssertEqual(
      configurationReloadResult.workerRuntimeFeatureConfiguration?.configurationGeneration,
      configurationReloadResult.effectiveGeneration
    )
  }

  func test_should_decode_reload_generation_and_full_loaded_model_configuration() throws {
    let responseBody = Data(
      """
      {
        "message": "Config reloaded and applied by the worker",
        "worker_restart_completed": true,
        "candidate_generation": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "effective_generation": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "worker_runtime_feature_configuration": {
          "configuration_generation": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
          "persistent_prompt_cache_enabled": true,
          "prompt_cache_maximum_size_bytes": 50000000000,
          "loaded_model": {
            "kind": "autoregressive",
            "configuration": {
              "model_id": "fictional/target",
              "maximum_context_tokens": 32768,
              "maximum_output_tokens": 4096,
              "chunking": {
                "fixed_prompt_processing_chunk_size_tokens": 2048,
                "fixed_ssd_streaming_prompt_processing_chunk_size_tokens": 256,
                "full_attention_key_value_growth_tokens": 256,
                "prefill_graph_submission_layer_interval": 0,
                "experimental_ssd_paging_prefill_graph_submission_layer_interval": 1,
                "experimental_ssd_paging_generation_graph_submission_layer_interval": 0,
                "prompt_cache_block_tokens": 128,
                "prompt_cache_common_prefix_stride_blocks": 4,
                "experimental_decode_stage_attribution_enabled": false,
                "experimental_prefill_stage_attribution_enabled": false,
                "experimental_in_memory_prefix_adoption_enabled": false,
                "experimental_neural_engine_prefill_enabled": false
              }
            }
          }
        }
      }
      """.utf8)

    let reloadResult = try JSONDecoder().decode(ConfigurationReloadResult.self, from: responseBody)

    XCTAssertEqual(reloadResult.candidateGeneration, reloadResult.effectiveGeneration)
    let loadedModelConfiguration = reloadResult.workerRuntimeFeatureConfiguration?.loadedModel?
      .autoregressiveConfiguration
    XCTAssertEqual(
      loadedModelConfiguration?.maximumContextTokens, 32_768)
  }

  func test_should_decode_the_speculative_decoding_block_length_from_the_worker_policy() throws {
    let policyWithSpeculation = """
      {
        "fixed_prompt_processing_chunk_size_tokens": 2048,
        "fixed_ssd_streaming_prompt_processing_chunk_size_tokens": 256,
        "full_attention_key_value_growth_tokens": 256,
        "prefill_graph_submission_layer_interval": 0,
        "experimental_ssd_paging_prefill_graph_submission_layer_interval": 1,
        "experimental_ssd_paging_generation_graph_submission_layer_interval": 0,
        "prompt_cache_block_tokens": 128,
        "speculative_decoding_block_length": 4,
        "prompt_cache_common_prefix_stride_blocks": 4,
        "experimental_decode_stage_attribution_enabled": false,
        "experimental_prefill_stage_attribution_enabled": false,
        "experimental_in_memory_prefix_adoption_enabled": false,
        "experimental_neural_engine_prefill_enabled": false
      }
      """
    let policyWithoutSpeculation = """
      {
        "fixed_prompt_processing_chunk_size_tokens": 2048,
        "fixed_ssd_streaming_prompt_processing_chunk_size_tokens": 256,
        "full_attention_key_value_growth_tokens": 256,
        "prefill_graph_submission_layer_interval": 0,
        "experimental_ssd_paging_prefill_graph_submission_layer_interval": 1,
        "experimental_ssd_paging_generation_graph_submission_layer_interval": 0,
        "prompt_cache_block_tokens": 128,
        "prompt_cache_common_prefix_stride_blocks": 4,
        "experimental_decode_stage_attribution_enabled": false,
        "experimental_prefill_stage_attribution_enabled": false,
        "experimental_in_memory_prefix_adoption_enabled": false,
        "experimental_neural_engine_prefill_enabled": false
      }
      """

    // A speculating worker carries the key; a classic-decoding worker omits
    // it entirely because the Rust DTO skips serialization for None.
    let admittedDecoding = try JSONDecoder().decode(
      WorkerChunkingConfiguration.self, from: Data(policyWithSpeculation.utf8))
    let classicDecoding = try JSONDecoder().decode(
      WorkerChunkingConfiguration.self, from: Data(policyWithoutSpeculation.utf8))

    XCTAssertEqual(admittedDecoding.speculativeDecodingBlockLength, 4)
    XCTAssertNil(classicDecoding.speculativeDecodingBlockLength)
  }

  func test_should_reject_a_truly_unknown_worker_policy_field() throws {
    let policyWithAnUnknownField = """
      {
        "fixed_prompt_processing_chunk_size_tokens": 2048,
        "fixed_ssd_streaming_prompt_processing_chunk_size_tokens": 256,
        "full_attention_key_value_growth_tokens": 256,
        "prefill_graph_submission_layer_interval": 0,
        "experimental_ssd_paging_prefill_graph_submission_layer_interval": 1,
        "experimental_ssd_paging_generation_graph_submission_layer_interval": 0,
        "prompt_cache_block_tokens": 128,
        "prompt_cache_common_prefix_stride_blocks": 4,
        "experimental_decode_stage_attribution_enabled": false,
        "experimental_prefill_stage_attribution_enabled": false,
        "experimental_in_memory_prefix_adoption_enabled": false,
        "experimental_neural_engine_prefill_enabled": false,
        "time_travel_chunk_size_tokens": 12
      }
      """

    XCTAssertThrowsError(
      try JSONDecoder().decode(
        WorkerChunkingConfiguration.self, from: Data(policyWithAnUnknownField.utf8))
    ) { decodingError in
      // Strict decoding is the point of this decoder: an unrecognized
      // policy field must fail with the unknown-field diagnosis instead of
      // silently misreading the worker contract.
      guard case DecodingError.dataCorrupted(let context) = decodingError else {
        return XCTFail("expected the strict unknown-field rejection, got \(decodingError)")
      }
      XCTAssertTrue(
        context.debugDescription.contains("Unexpected worker runtime policy field"),
        "the rejection must diagnose the unknown field, got: \(context.debugDescription)")
    }
  }

  func test_should_reject_status_from_the_opposite_runtime_instance() async {
    StubDaemonURLProtocol.statusResponseConfiguration = .init(
      statusCode: 200,
      responseBody: Data(
        #"{"application":{"version":"0.2.0","build_number":100,"commit":"abcdef1","is_dirty":false,"channel":"stable","channel_display_name":"Stable","state_directory":"~/.astronomical"},"status":"ready","activity":"idle"}"#.utf8)
    )
    defer { StubDaemonURLProtocol.statusResponseConfiguration = nil }
    let developmentClient = LocalDaemonClient(
      applicationIdentity: applicationIdentity(channel: .development),
      urlSession: URLSession(configuration: StubDaemonURLProtocol.urlSessionConfiguration())
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

  /// The store build runs the Stable instance with its state in the app group
  /// container. Its agent must accept exactly that engine; when it expected a
  /// channel named "app-store" it refused its own engine and restarted it
  /// every few seconds (issue #2188). The label is the one the engine's
  /// instance paths produce for the same fictional container.
  func test_should_accept_the_stable_engine_in_the_group_container_for_the_store_build() async throws {
    StubDaemonURLProtocol.statusResponseConfiguration = .init(
      statusCode: 200,
      responseBody: statusBody(
        stateDirectory: "~/Library/Group Containers/TEAM123456.astronomical/Astronomical"))
    defer { StubDaemonURLProtocol.statusResponseConfiguration = nil }
    let storeClient = LocalDaemonClient(
      applicationIdentity: storeIdentity(applicationGroupIdentifier: "TEAM123456.astronomical"),
      urlSession: URLSession(configuration: StubDaemonURLProtocol.urlSessionConfiguration())
    )

    let statusDocument = try await storeClient.fetchStatus()

    XCTAssertEqual(statusDocument.status, "ready")
  }

  /// A store build without an app group keeps its state in its own
  /// container's Application Support, and expects the engine to say so.
  func test_should_accept_the_stable_engine_in_application_support_for_a_store_build_without_a_group()
    async throws
  {
    StubDaemonURLProtocol.statusResponseConfiguration = .init(
      statusCode: 200,
      responseBody: statusBody(stateDirectory: "~/Library/Application Support/Astronomical"))
    defer { StubDaemonURLProtocol.statusResponseConfiguration = nil }
    let storeClient = LocalDaemonClient(
      applicationIdentity: storeIdentity(applicationGroupIdentifier: nil),
      urlSession: URLSession(configuration: StubDaemonURLProtocol.urlSessionConfiguration())
    )

    let statusDocument = try await storeClient.fetchStatus()

    XCTAssertEqual(statusDocument.status, "ready")
  }

  /// The direct app's engine answers on the same port as the store build's.
  /// The state directory is what tells the two installs apart, so the store
  /// agent must refuse it and never control another install's engine.
  func test_should_reject_the_direct_stable_engine_for_the_store_build() async {
    StubDaemonURLProtocol.statusResponseConfiguration = .init(
      statusCode: 200, responseBody: statusBody(stateDirectory: "~/.astronomical"))
    defer { StubDaemonURLProtocol.statusResponseConfiguration = nil }
    let storeClient = LocalDaemonClient(
      applicationIdentity: storeIdentity(applicationGroupIdentifier: "TEAM123456.astronomical"),
      urlSession: URLSession(configuration: StubDaemonURLProtocol.urlSessionConfiguration())
    )

    do {
      _ = try await storeClient.fetchStatus()
      XCTFail("The store build must not adopt the direct app's engine")
    } catch let clientError as DaemonClientError {
      guard case let .wrongStateDirectory(expectedStateDirectory, connectedStateDirectory) = clientError
      else { return XCTFail("Expected a state directory refusal, got \(clientError)") }
      XCTAssertEqual(
        expectedStateDirectory, "~/Library/Group Containers/TEAM123456.astronomical/Astronomical")
      XCTAssertEqual(connectedStateDirectory, "~/.astronomical")
    } catch {
      XCTFail("Expected a daemon client error, got \(error)")
    }
  }

  func test_should_reject_an_unidentified_server_for_every_runtime_instance() async {
    StubDaemonURLProtocol.statusResponseConfiguration = .init(
      statusCode: 200,
      responseBody: Data(#"{"status":"ready","activity":"idle"}"#.utf8)
    )
    defer { StubDaemonURLProtocol.statusResponseConfiguration = nil }

    for runtimeChannel in [ApplicationChannel.stable, ApplicationChannel.development] {
      let localDaemonClient = LocalDaemonClient(
        applicationIdentity: applicationIdentity(channel: runtimeChannel),
        urlSession: URLSession(configuration: StubDaemonURLProtocol.urlSessionConfiguration())
      )

      do {
        _ = try await localDaemonClient.fetchStatus()
        XCTFail("Every menu must reject a server that cannot prove its runtime channel")
      } catch {
        XCTAssertEqual(
          error.localizedDescription,
          "Expected the \(runtimeChannel.rawValue) server, but the unidentified instance answered")
      }
    }
  }

  func test_should_report_an_occupied_endpoint_without_trusting_its_instance_identity() async {
    StubDaemonURLProtocol.responseConfiguration = .init(
      statusCode: 200,
      responseBody: Data(#"{"status":"ok"}"#.utf8)
    )
    defer { StubDaemonURLProtocol.responseConfiguration = nil }
    let localDaemonClient = LocalDaemonClient(
      applicationIdentity: applicationIdentity(channel: .development),
      urlSession: URLSession(configuration: StubDaemonURLProtocol.urlSessionConfiguration())
    )

    let endpointIsOccupied = await localDaemonClient.healthIsAvailable()

    XCTAssertTrue(endpointIsOccupied)
    XCTAssertEqual(StubDaemonURLProtocol.receivedRequestPaths.last, "/health")
  }

  func test_should_require_ready_acknowledged_effective_configuration_for_startup_health() async {
    StubDaemonURLProtocol.responseConfiguration = .init(
      statusCode: 200, responseBody: Data(#"{"status":"ok"}"#.utf8))
    defer {
      StubDaemonURLProtocol.responseConfiguration = nil
      StubDaemonURLProtocol.statusResponseConfiguration = nil
    }
    let developmentClient = LocalDaemonClient(
      applicationIdentity: applicationIdentity(channel: .development),
      urlSession: URLSession(configuration: StubDaemonURLProtocol.urlSessionConfiguration())
    )
    let applicationIdentityDocument = #""application":{"version":"0.2.0","build_number":100,"commit":"abcdef1","is_dirty":true,"channel":"development","channel_display_name":"Development","state_directory":"~/.astronomical-dev"}"#

    StubDaemonURLProtocol.statusResponseConfiguration = .init(
      statusCode: 200,
      responseBody: Data(
        "{\(applicationIdentityDocument),\"status\":\"loading\",\"activity\":\"idle\",\"worker_runtime_feature_configuration_applied\":false,\"configuration\":{\"configured_generation\":\"a\",\"resolved_generation\":\"a\",\"effective_generation\":null,\"is_effective\":false,\"restart_required\":false}}".utf8)
    )
    let loadingInstanceIsHealthy = await developmentClient.expectedInstanceIsHealthy()
    XCTAssertFalse(loadingInstanceIsHealthy)

    StubDaemonURLProtocol.statusResponseConfiguration = .init(
      statusCode: 200,
      responseBody: Data(
        "{\(applicationIdentityDocument),\"status\":\"ready\",\"activity\":\"idle\",\"worker_runtime_feature_configuration_applied\":true,\"configuration\":{\"configured_generation\":\"a\",\"resolved_generation\":\"a\",\"effective_generation\":\"a\",\"is_effective\":true,\"restart_required\":false}}".utf8)
    )
    let readyInstanceIsHealthy = await developmentClient.expectedInstanceIsHealthy()
    XCTAssertTrue(readyInstanceIsHealthy)
  }

  func test_should_refuse_to_control_the_opposite_runtime_instance() async {
    StubDaemonURLProtocol.statusResponseConfiguration = .init(
      statusCode: 200,
      responseBody: Data(
        #"{"application":{"version":"0.2.0","build_number":100,"commit":"abcdef1","is_dirty":false,"channel":"stable","channel_display_name":"Stable","state_directory":"~/.astronomical"},"status":"ready","activity":"idle"}"#.utf8)
    )
    StubDaemonURLProtocol.responseConfiguration = .init(statusCode: 202, responseBody: Data())
    StubDaemonURLProtocol.receivedRequestPaths = []
    defer {
      StubDaemonURLProtocol.statusResponseConfiguration = nil
      StubDaemonURLProtocol.responseConfiguration = nil
      StubDaemonURLProtocol.receivedRequestPaths = []
    }
    let developmentClient = LocalDaemonClient(
      applicationIdentity: applicationIdentity(channel: .development),
      urlSession: URLSession(configuration: StubDaemonURLProtocol.urlSessionConfiguration())
    )

    do {
      try await developmentClient.requestShutdown()
      XCTFail("Development must not control a Stable server")
    } catch {
      XCTAssertEqual(
        error.localizedDescription,
        "Expected the development server, but the stable instance answered")
      XCTAssertEqual(StubDaemonURLProtocol.receivedRequestPaths, ["/v1/status"])
    }
  }

  func test_should_refuse_to_control_same_channel_server_with_unexpected_state() async {
    StubDaemonURLProtocol.statusResponseConfiguration = .init(
      statusCode: 200,
      responseBody: Data(
        #"{"application":{"version":"0.2.0","build_number":100,"commit":"abcdef1","is_dirty":true,"channel":"development","channel_display_name":"Development","state_directory":"custom"},"status":"ready","activity":"idle"}"#.utf8)
    )
    StubDaemonURLProtocol.responseConfiguration = .init(statusCode: 202, responseBody: Data())
    StubDaemonURLProtocol.receivedRequestPaths = []
    defer {
      StubDaemonURLProtocol.statusResponseConfiguration = nil
      StubDaemonURLProtocol.responseConfiguration = nil
      StubDaemonURLProtocol.receivedRequestPaths = []
    }
    let developmentClient = LocalDaemonClient(
      applicationIdentity: applicationIdentity(channel: .development),
      urlSession: URLSession(configuration: StubDaemonURLProtocol.urlSessionConfiguration())
    )

    do {
      try await developmentClient.requestShutdown()
      XCTFail("Development must not control a custom-state server")
    } catch {
      XCTAssertEqual(
        error.localizedDescription,
        "Expected server state ~/.astronomical-dev, but the connected instance reported custom")
      XCTAssertEqual(StubDaemonURLProtocol.receivedRequestPaths, ["/v1/status"])
    }
  }

  func test_should_preserve_the_server_message_when_configuration_reload_is_rejected() async {
    StubDaemonURLProtocol.responseConfiguration = .init(
      statusCode: 409,
      responseBody: Data(
        #"{"status":"busy","message":"A generation is active or queued; reload aborted"}"#.utf8)
    )
    defer { StubDaemonURLProtocol.responseConfiguration = nil }

    let localDaemonClient = LocalDaemonClient(
      urlSession: URLSession(configuration: StubDaemonURLProtocol.urlSessionConfiguration())
    )

    do {
      _ = try await localDaemonClient.reloadConfiguration()
      XCTFail("A rejected reload must throw")
    } catch {
      XCTAssertEqual(error.localizedDescription, "A generation is active or queued; reload aborted")
    }
  }

  func test_should_present_invalid_configuration_feedback_when_reload_is_rejected() async {
    StubDaemonURLProtocol.responseConfiguration = .init(
      statusCode: 400,
      responseBody: Data(#"{"status":"invalid_config","message":"invalid Astronomical configuration"}"#.utf8)
    )
    defer { StubDaemonURLProtocol.responseConfiguration = nil }

    let localDaemonClient = LocalDaemonClient(
      urlSession: URLSession(configuration: StubDaemonURLProtocol.urlSessionConfiguration())
    )

    do {
      _ = try await localDaemonClient.reloadConfiguration()
      XCTFail("An invalid configuration reload must throw")
    } catch {
      XCTAssertEqual(error.localizedDescription, "invalid Astronomical configuration")
    }
  }

  func test_should_allow_a_live_memory_update_to_use_the_bounded_reclamation_timeout() async throws {
    StubDaemonURLProtocol.responseConfiguration = .init(
      statusCode: 200,
      responseBody: Data(#"{"message":"MLX memory setting persisted and applied"}"#.utf8)
    )
    StubDaemonURLProtocol.receivedRequestTimeoutInterval = nil
    defer {
      StubDaemonURLProtocol.responseConfiguration = nil
      StubDaemonURLProtocol.receivedRequestTimeoutInterval = nil
    }
    let localDaemonClient = LocalDaemonClient(
      urlSession: URLSession(configuration: StubDaemonURLProtocol.urlSessionConfiguration())
    )

    _ = try await localDaemonClient.updateMaximumMlxMemoryGigabytes(32)

    XCTAssertEqual(StubDaemonURLProtocol.receivedRequestTimeoutInterval, 120)
  }

  private func storeIdentity(applicationGroupIdentifier: String?) -> ApplicationIdentity {
    ApplicationIdentity(
      channel: .appStore,
      daemonPort: ApplicationChannel.appStore.defaultDaemonPort,
      stateDirectoryName: nil,
      version: "0.2.0",
      buildNumber: "100",
      commit: "abcdef1",
      isDirty: false,
      applicationGroupIdentifier: applicationGroupIdentifier
    )
  }

  /// A ready Stable engine's status naming `stateDirectory`.
  private func statusBody(stateDirectory: String) -> Data {
    Data(
      """
      {"application":{"version":"0.2.0","build_number":100,"commit":"abcdef1","is_dirty":false,\
      "channel":"stable","channel_display_name":"Stable","state_directory":"\(stateDirectory)"},\
      "status":"ready","activity":"idle"}
      """.utf8)
  }

  private func applicationIdentity(channel: ApplicationChannel) -> ApplicationIdentity {
    ApplicationIdentity(
      channel: channel,
      daemonPort: channel.defaultDaemonPort,
      stateDirectoryName: channel.stateDirectoryName,
      version: "0.2.0",
      buildNumber: "100",
      commit: "abcdef1",
      isDirty: channel == .development
    )
  }

}
