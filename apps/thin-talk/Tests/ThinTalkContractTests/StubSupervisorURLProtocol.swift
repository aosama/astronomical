// Supplies deterministic localhost responses without contacting either supervisor
// channel. Tests set the response bodies before building a client and clear them
// afterwards so each test's stub is isolated.

import Foundation

final class StubSupervisorURLProtocol: URLProtocol, @unchecked Sendable {
  struct ResponseConfiguration {
    let statusCode: Int
    let responseBody: Data
  }

  nonisolated(unsafe) static var statusResponse: ResponseConfiguration?
  nonisolated(unsafe) static var modelsResponse: ResponseConfiguration?
  nonisolated(unsafe) static var chatResponse: ResponseConfiguration?
  nonisolated(unsafe) static var chatHoldOpen = false
  nonisolated(unsafe) static var receivedRequestPaths: [String] = []
  nonisolated(unsafe) static var receivedRequestMethods: [String] = []
  /// Outbound JSON bodies, so a test can observe what actually left the process
  /// instead of trusting that a field took the whole path to the wire.
  nonisolated(unsafe) static var receivedRequestBodies: [Data] = []

  override class func canInit(with request: URLRequest) -> Bool { true }
  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

  override func startLoading() {
    let path = request.url?.path ?? ""
    Self.receivedRequestPaths.append(path)
    Self.receivedRequestMethods.append(request.httpMethod ?? "GET")
    Self.receivedRequestBodies.append(Self.wireBody(of: request))

    guard let requestURL = request.url, let urlProtocolClient = client else {
      return
    }

    // A stalled generation has a response that never ends and never delivers a
    // token; the chat surface must detect it through its own client-side watchdog.
    if path == "/v1/chat/completions" && Self.chatHoldOpen {
      let heldResponse = HTTPURLResponse(
        url: requestURL, statusCode: 200, httpVersion: "HTTP/1.1",
        headerFields: ["Content-Type": "text/event-stream"])
      if let heldResponse {
        urlProtocolClient.urlProtocol(self, didReceive: heldResponse, cacheStoragePolicy: .notAllowed)
      }
      return
    }

    let selectedResponse: ResponseConfiguration?
    switch path {
    case "/v1/status":
      selectedResponse = Self.statusResponse
        ?? Self.matchingStableStatusResponse
    case "/v1/models":
      selectedResponse = Self.modelsResponse
    case "/v1/chat/completions":
      selectedResponse = Self.chatResponse
    default:
      selectedResponse = nil
    }

    guard let selectedResponse else { return }
    let response = HTTPURLResponse(
      url: requestURL,
      statusCode: selectedResponse.statusCode,
      httpVersion: "HTTP/1.1",
      headerFields: ["Content-Type": "text/event-stream"]
    )
    guard let response else { return }
    urlProtocolClient.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
    urlProtocolClient.urlProtocol(self, didLoad: selectedResponse.responseBody)
    urlProtocolClient.urlProtocolDidFinishLoading(self)
  }

  override func stopLoading() {}

  static func urlSessionConfiguration() -> URLSessionConfiguration {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [StubSupervisorURLProtocol.self]
    return configuration
  }

  /// Reads the outbound JSON from the request.
  ///
  /// A data task reaches a URL protocol with its body attached as a stream rather
  /// than as `httpBody`, so reading only `httpBody` would leave every captured
  /// request looking empty and let a broken wire contract pass unnoticed.
  private static func wireBody(of request: URLRequest) -> Data {
    if let body = request.httpBody, !body.isEmpty {
      return body
    }
    guard let stream = request.httpBodyStream else { return Data() }
    stream.open()
    defer { stream.close() }
    var body = Data()
    var buffer = [UInt8](repeating: 0, count: 4096)
    while stream.hasBytesAvailable {
      let readCount = stream.read(&buffer, maxLength: buffer.count)
      if readCount <= 0 { break }
      body.append(buffer, count: readCount)
    }
    return body
  }

  static let matchingStableStatusResponse: ResponseConfiguration = .init(
    statusCode: 200,
    responseBody: Data(
      #"{"application":{"version":"0.1.0","build_number":1,"commit":"test","is_dirty":false,"channel":"stable","state_directory":"~/.astronomical"},"status":"ready","activity":"idle"}"#.utf8)
  )

  /// A stub chat completion with a seeded SSE body.
  static func stubbedChat(sseBody: Data, statusCode: Int = 200) -> ResponseConfiguration {
    ResponseConfiguration(statusCode: statusCode, responseBody: sseBody)
  }

  static func stubbedChatHoldOpen() {
    chatHoldOpen = true
  }
}
