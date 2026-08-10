import Foundation

private let supervisorPort = 6732
private let maximumSupervisorResponseByteCount = 1_048_576
private let maximumMlxMemoryUpdateTimeoutSeconds: TimeInterval = 120

func localSupervisorEndpointURL(path: String) throws -> URL {
  guard let endpointURL = URL(string: "http://127.0.0.1:\(supervisorPort)\(path)") else {
    throw SupervisorClientError.invalidEndpoint
  }
  return endpointURL
}

protocol SupervisorClient: Sendable {
  func fetchStatus() async throws -> SupervisorStatusDocument
  func reloadConfiguration() async throws -> ConfigurationReloadResult
  func requestShutdown() async throws
  func updateMaximumMlxMemoryGigabytes(_ maximumMlxMemoryGigabytes: UInt64?) async throws -> String
  func healthIsAvailable() async -> Bool
}

extension SupervisorClient {
  func updateMaximumMlxMemoryGigabytes(_ maximumMlxMemoryGigabytes: UInt64?) async throws -> String {
    throw SupervisorClientError.serverRejected("Maximum model RAM control is unavailable")
  }
}

struct LocalSupervisorClient: SupervisorClient {
  private let urlSession: URLSession

  init(urlSession: URLSession = .shared) { self.urlSession = urlSession }

  func fetchStatus() async throws -> SupervisorStatusDocument {
    let responseBody = try await request(path: "/v1/status", method: "GET", acceptedStatusCodes: [200])
    return try JSONDecoder().decode(SupervisorStatusDocument.self, from: responseBody)
  }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    let responseBody = try await request(
      path: "/v1/config/reload", method: "POST", acceptedStatusCodes: [200, 202]
    )
    return try JSONDecoder().decode(ConfigurationReloadResult.self, from: responseBody)
  }

  func requestShutdown() async throws {
    _ = try await request(path: "/v1/control/shutdown", method: "POST", acceptedStatusCodes: [202])
  }

  func updateMaximumMlxMemoryGigabytes(_ maximumMlxMemoryGigabytes: UInt64?) async throws -> String {
    let requestBody = try JSONEncoder().encode(
      MaximumMlxMemoryRequest(maximumMlxMemoryGigabytes: maximumMlxMemoryGigabytes))
    let responseBody = try await request(
      path: "/v1/config/maximum-mlx-memory", method: "PUT", requestBody: requestBody,
      acceptedStatusCodes: [200, 202], timeoutInterval: maximumMlxMemoryUpdateTimeoutSeconds)
    return try JSONDecoder().decode(ConfigurationReloadResult.self, from: responseBody).message
  }

  func healthIsAvailable() async -> Bool {
    (try? await request(path: "/health", method: "GET", acceptedStatusCodes: [200])) != nil
  }

  private func request(
    path: String,
    method: String,
    requestBody: Data? = nil,
    acceptedStatusCodes: Set<Int>,
    timeoutInterval: TimeInterval = 2
  ) async throws -> Data {
    let endpointURL = try localSupervisorEndpointURL(path: path)
    var supervisorRequest = URLRequest(url: endpointURL)
    supervisorRequest.httpMethod = method
    supervisorRequest.httpBody = requestBody
    if requestBody != nil { supervisorRequest.setValue("application/json", forHTTPHeaderField: "Content-Type") }
    supervisorRequest.timeoutInterval = timeoutInterval
    let (responseBody, response) = try await urlSession.data(for: supervisorRequest)
    guard responseBody.count <= maximumSupervisorResponseByteCount else {
      throw SupervisorClientError.responseTooLarge
    }
    guard let httpResponse = response as? HTTPURLResponse else {
      throw SupervisorClientError.unexpectedResponse
    }
    guard acceptedStatusCodes.contains(httpResponse.statusCode) else {
      let serverMessage = try? JSONDecoder().decode(ConfigurationReloadResult.self, from: responseBody)
      throw SupervisorClientError.serverRejected(
        serverMessage?.message ?? "Server returned HTTP \(httpResponse.statusCode)"
      )
    }
    return responseBody
  }
}

struct ConfigurationReloadResult: Decodable, Equatable {
  let message: String
  let workerRestartCompleted: Bool
  let workerRuntimeFeatureConfiguration: WorkerRuntimeFeatureConfiguration?

  init(
    message: String,
    workerRestartCompleted: Bool = false,
    workerRuntimeFeatureConfiguration: WorkerRuntimeFeatureConfiguration? = nil
  ) {
    self.message = message
    self.workerRestartCompleted = workerRestartCompleted
    self.workerRuntimeFeatureConfiguration = workerRuntimeFeatureConfiguration
  }

  enum CodingKeys: String, CodingKey {
    case message
    case workerRestartCompleted = "worker_restart_completed"
    case workerRuntimeFeatureConfiguration = "worker_runtime_feature_configuration"
  }

  init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    message = try container.decode(String.self, forKey: .message)
    workerRestartCompleted = try container.decodeIfPresent(Bool.self, forKey: .workerRestartCompleted) ?? false
    workerRuntimeFeatureConfiguration = try container.decodeIfPresent(
      WorkerRuntimeFeatureConfiguration.self, forKey: .workerRuntimeFeatureConfiguration)
  }
}

/// Exact feature policy acknowledged by the worker and echoed by local control responses.
///
/// This remains Codable because status fixtures also encode SupervisorStatusDocument while
/// exercising the same localhost wire contract that the menu decodes in production.
struct WorkerRuntimeFeatureConfiguration: Codable, Equatable {
  let persistentPromptCacheEnabled: Bool
  let mtpEnabled: Bool
  let speculativePrefillEnabled: Bool

  enum CodingKeys: String, CodingKey {
    case persistentPromptCacheEnabled = "persistent_prompt_cache_enabled"
    case mtpEnabled = "mtp_enabled"
    case speculativePrefillEnabled = "speculative_prefill_enabled"
  }
}

private struct MaximumMlxMemoryRequest: Encodable {
  let maximumMlxMemoryGigabytes: UInt64?

  enum CodingKeys: String, CodingKey { case maximumMlxMemoryGigabytes = "maximum_mlx_memory_gb" }
}

enum SupervisorClientError: LocalizedError {
  case invalidEndpoint
  case responseTooLarge
  case unexpectedResponse
  case serverRejected(String)

  var errorDescription: String? {
    switch self {
    case .invalidEndpoint: "The local server address is invalid"
    case .responseTooLarge: "The server control response was too large"
    case .unexpectedResponse: "The server returned an invalid response"
    case let .serverRejected(serverMessage): serverMessage
    }
  }
}
