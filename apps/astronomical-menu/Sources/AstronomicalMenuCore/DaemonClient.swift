import Foundation

private let maximumDaemonResponseByteCount = 1_048_576
private let maximumMlxMemoryUpdateTimeoutSeconds: TimeInterval = 120

protocol DaemonClient: Sendable {
  func fetchStatus() async throws -> DaemonStatusDocument
  func reloadConfiguration() async throws -> ConfigurationReloadResult
  func requestShutdown() async throws
  func updateMaximumMlxMemoryGigabytes(_ maximumMlxMemoryGigabytes: UInt64?) async throws -> String
  func healthIsAvailable() async -> Bool
  func expectedInstanceIsHealthy() async -> Bool
  func modelsAreAvailable() async -> Bool
}

extension DaemonClient {
  func modelsAreAvailable() async -> Bool { false }
  func updateMaximumMlxMemoryGigabytes(_ maximumMlxMemoryGigabytes: UInt64?) async throws -> String {
    throw DaemonClientError.serverRejected("Maximum model RAM control is unavailable")
  }

  func expectedInstanceIsHealthy() async -> Bool {
    guard await healthIsAvailable() else { return false }
    guard let statusDocument = try? await fetchStatus() else { return false }
    return statusDocument.status == "ready"
      && statusDocument.workerRuntimeFeatureConfigurationApplied
      && statusDocument.configuration?.isEffective == true
  }
}

struct LocalDaemonClient: DaemonClient {
  private let applicationIdentity: ApplicationIdentity
  private let urlSession: URLSession

  init(
    applicationIdentity: ApplicationIdentity = .current(),
    urlSession: URLSession = .shared
  ) {
    self.applicationIdentity = applicationIdentity
    self.urlSession = urlSession
  }

  func fetchStatus() async throws -> DaemonStatusDocument {
    let responseBody = try await request(path: "/v1/status", method: "GET", acceptedStatusCodes: [200])
    return try validateExpectedInstance(responseBody: responseBody)
  }

  private func validateExpectedInstance(responseBody: Data) throws -> DaemonStatusDocument {
    let statusDocument = try JSONDecoder().decode(DaemonStatusDocument.self, from: responseBody)
    guard let connectedChannel = statusDocument.application?.channel else {
      // A menu cannot safely control a process until the process proves which
      // isolated state and endpoint contract it owns.
      throw DaemonClientError.wrongInstance(
        expected: applicationIdentity.channel.rawValue, connected: "unidentified")
    }
    guard connectedChannel == applicationIdentity.channel.rawValue else {
      throw DaemonClientError.wrongInstance(
        expected: applicationIdentity.channel.rawValue, connected: connectedChannel)
    }
    guard statusDocument.application?.stateDirectory == applicationIdentity.expectedServerStateDirectory else {
      let connectedStateDirectory: String
      if let stateDirectory = statusDocument.application?.stateDirectory {
        connectedStateDirectory = stateDirectory
      } else {
        connectedStateDirectory = "unidentified"
      }
      throw DaemonClientError.wrongStateDirectory(
        expected: applicationIdentity.expectedServerStateDirectory,
        connected: connectedStateDirectory)
    }
    return statusDocument
  }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    _ = try await fetchStatus()
    let responseBody = try await request(
      path: "/v1/config/reload", method: "POST", acceptedStatusCodes: [200, 202]
    )
    return try JSONDecoder().decode(ConfigurationReloadResult.self, from: responseBody)
  }

  func requestShutdown() async throws {
    _ = try await fetchStatus()
    _ = try await request(path: "/v1/control/shutdown", method: "POST", acceptedStatusCodes: [202])
  }

  func updateMaximumMlxMemoryGigabytes(_ maximumMlxMemoryGigabytes: UInt64?) async throws -> String {
    _ = try await fetchStatus()
    let requestBody = try JSONEncoder().encode(
      MaximumMlxMemoryRequest(maximumMlxMemoryGigabytes: maximumMlxMemoryGigabytes))
    let responseBody = try await request(
      path: "/v1/config/maximum-mlx-memory", method: "PUT", requestBody: requestBody,
      acceptedStatusCodes: [200, 202], timeoutInterval: maximumMlxMemoryUpdateTimeoutSeconds)
    return try JSONDecoder().decode(ConfigurationReloadResult.self, from: responseBody).message
  }

  func healthIsAvailable() async -> Bool {
    // Lifecycle decisions only need to know whether this channel's endpoint is
    // occupied. Every control operation separately verifies channel identity.
    (try? await request(path: "/health", method: "GET", acceptedStatusCodes: [200])) != nil
  }

  func modelsAreAvailable() async -> Bool {
    guard let responseBody = try? await request(
      path: "/v1/models", method: "GET", acceptedStatusCodes: [200]
    ), let modelsDocument = try? JSONDecoder().decode(ModelsDocument.self, from: responseBody)
    else { return false }
    return !modelsDocument.data.isEmpty
  }

  private func request(
    path: String,
    method: String,
    requestBody: Data? = nil,
    acceptedStatusCodes: Set<Int>,
    timeoutInterval: TimeInterval = 2
  ) async throws -> Data {
    let endpointURL = try applicationIdentity.endpointURL(path: path)
    var daemonRequest = URLRequest(url: endpointURL)
    daemonRequest.httpMethod = method
    daemonRequest.httpBody = requestBody
    if requestBody != nil { daemonRequest.setValue("application/json", forHTTPHeaderField: "Content-Type") }
    daemonRequest.timeoutInterval = timeoutInterval
    let (responseBody, response) = try await urlSession.data(for: daemonRequest)
    guard responseBody.count <= maximumDaemonResponseByteCount else {
      throw DaemonClientError.responseTooLarge
    }
    guard let httpResponse = response as? HTTPURLResponse else {
      throw DaemonClientError.unexpectedResponse
    }
    guard acceptedStatusCodes.contains(httpResponse.statusCode) else {
      let serverMessage = try? JSONDecoder().decode(ConfigurationReloadResult.self, from: responseBody)
      if let serverMessage {
        throw DaemonClientError.serverRejected(serverMessage.message)
      }
      throw DaemonClientError.serverRejected(
        "Server returned HTTP \(httpResponse.statusCode)"
      )
    }
    return responseBody
  }
}

private struct ModelsDocument: Decodable {
  let data: [ModelIdentity]
}

private struct ModelIdentity: Decodable {
  let id: String
}

struct ConfigurationReloadResult: Decodable, Equatable {
  let message: String
  let workerRestartCompleted: Bool
  let workerRuntimeFeatureConfiguration: WorkerRuntimeFeatureConfiguration?
  let restApiRestartRequired: Bool?
  let candidateGeneration: String?
  let effectiveGeneration: String?

  init(
    message: String,
    workerRestartCompleted: Bool = false,
    workerRuntimeFeatureConfiguration: WorkerRuntimeFeatureConfiguration? = nil,
    restApiRestartRequired: Bool? = nil,
    candidateGeneration: String? = nil,
    effectiveGeneration: String? = nil
  ) {
    self.message = message
    self.workerRestartCompleted = workerRestartCompleted
    self.workerRuntimeFeatureConfiguration = workerRuntimeFeatureConfiguration
    self.restApiRestartRequired = restApiRestartRequired
    self.candidateGeneration = candidateGeneration
    self.effectiveGeneration = effectiveGeneration
  }

  enum CodingKeys: String, CodingKey {
    case message
    case workerRestartCompleted = "worker_restart_completed"
    case workerRuntimeFeatureConfiguration = "worker_runtime_feature_configuration"
    case restApiRestartRequired = "rest_api_restart_required"
    case candidateGeneration = "candidate_generation"
    case effectiveGeneration = "effective_generation"
  }

  init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    message = try container.decode(String.self, forKey: .message)
    let decodedWorkerRestartCompleted = try container.decodeIfPresent(
      Bool.self, forKey: .workerRestartCompleted)
    if let decodedWorkerRestartCompleted {
      workerRestartCompleted = decodedWorkerRestartCompleted
    } else {
      workerRestartCompleted = false
    }
    workerRuntimeFeatureConfiguration = try container.decodeIfPresent(
      WorkerRuntimeFeatureConfiguration.self, forKey: .workerRuntimeFeatureConfiguration)
    restApiRestartRequired = try container.decodeIfPresent(Bool.self, forKey: .restApiRestartRequired)
    candidateGeneration = try container.decodeIfPresent(String.self, forKey: .candidateGeneration)
    effectiveGeneration = try container.decodeIfPresent(String.self, forKey: .effectiveGeneration)
  }
}

private struct MaximumMlxMemoryRequest: Encodable {
  let maximumMlxMemoryGigabytes: UInt64?

  enum CodingKeys: String, CodingKey { case maximumMlxMemoryGigabytes = "maximum_mlx_memory_gb" }
}

enum DaemonClientError: LocalizedError {
  case invalidEndpoint
  case responseTooLarge
  case unexpectedResponse
  case serverRejected(String)
  case wrongInstance(expected: String, connected: String)
  case wrongStateDirectory(expected: String, connected: String)

  var errorDescription: String? {
    switch self {
    case .invalidEndpoint: "The local server address is invalid"
    case .responseTooLarge: "The server control response was too large"
    case .unexpectedResponse: "The server returned an invalid response"
    case let .serverRejected(serverMessage): serverMessage
    case let .wrongInstance(expected, connected):
      "Expected the \(expected) server, but the \(connected) instance answered"
    case let .wrongStateDirectory(expected, connected):
      "Expected server state \(expected), but the connected instance reported \(connected)"
    }
  }
}
