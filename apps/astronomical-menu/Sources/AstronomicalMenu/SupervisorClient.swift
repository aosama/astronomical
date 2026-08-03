import Foundation

private let supervisorPort = 6732
private let maximumSupervisorResponseByteCount = 1_048_576
private let maximumMlxMemoryUpdateTimeoutSeconds: TimeInterval = 120

protocol SupervisorClient: Sendable {
  func fetchStatus() async throws -> SupervisorStatusDocument
  func reloadConfiguration() async throws -> String
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

  func reloadConfiguration() async throws -> String {
    let responseBody = try await request(
      path: "/v1/config/reload", method: "POST", acceptedStatusCodes: [200, 202]
    )
    return try JSONDecoder().decode(ConfigReloadResponse.self, from: responseBody).message
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
    return try JSONDecoder().decode(ConfigReloadResponse.self, from: responseBody).message
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
    guard let endpointURL = URL(string: "http://127.0.0.1:\(supervisorPort)\(path)") else {
      throw SupervisorClientError.invalidEndpoint
    }
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
      let serverMessage = try? JSONDecoder().decode(ConfigReloadResponse.self, from: responseBody)
      throw SupervisorClientError.serverRejected(
        serverMessage?.message ?? "Server returned HTTP \(httpResponse.statusCode)"
      )
    }
    return responseBody
  }
}

private struct ConfigReloadResponse: Decodable {
  let message: String
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
