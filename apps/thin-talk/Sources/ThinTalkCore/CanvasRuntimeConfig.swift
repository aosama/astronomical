import Foundation

/// The runtime configuration the Swift host injects into the page before the
/// bundle runs: where the daemon listens and which channel this window
/// belongs to. The page cannot discover these itself (the daemon port is
/// channel-specific), so the host is the single source of truth.
public struct CanvasRuntimeConfig: Equatable, Sendable, Encodable {
  private enum CodingKeys: String, CodingKey {
    case daemonBaseURL
    case expectedChannel
    case expectedStateDirectory
  }

  public let daemonBaseURL: String
  public let expectedChannel: String
  public let expectedStateDirectory: String

  public init(daemonBaseURL: String, expectedChannel: String, expectedStateDirectory: String) {
    self.daemonBaseURL = daemonBaseURL
    self.expectedChannel = expectedChannel
    self.expectedStateDirectory = expectedStateDirectory
  }

  /// Builds the configuration for one application identity.
  public init(applicationIdentity: ThinTalkApplicationIdentity) {
    self.init(
      daemonBaseURL: "http://127.0.0.1:\(applicationIdentity.daemonPort)",
      expectedChannel: applicationIdentity.channel.rawValue,
      expectedStateDirectory: applicationIdentity.expectedServerStateDirectory
    )
  }

  /// The document-start user script that installs the configuration. The page's
  /// AppConfig reads exactly these three keys and refuses to boot without them.
  public func userScriptSource() throws -> String {
    let encoded = try JSONEncoder().encode(self)
    guard let json = String(data: encoded, encoding: .utf8) else {
      throw CanvasRuntimeConfigError.unencodableConfiguration
    }
    return "window.__thintalkConfig = \(json);"
  }
}

public enum CanvasRuntimeConfigError: Error {
  case unencodableConfiguration
}
