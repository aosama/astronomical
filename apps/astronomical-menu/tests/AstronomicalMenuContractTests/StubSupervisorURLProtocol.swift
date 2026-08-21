// Supplies deterministic localhost responses without contacting either installed runtime channel.

import Foundation

final class StubSupervisorURLProtocol: URLProtocol, @unchecked Sendable {
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
