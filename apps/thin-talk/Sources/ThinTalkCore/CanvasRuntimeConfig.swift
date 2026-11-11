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
    case titlebarInset
    case canRestartEngine
    case printedPaper
  }

  /// A sheet of paper in points, as macOS measures paper.
  public struct PrintedPaper: Equatable, Sendable, Encodable {
    public let widthPoints: Double
    public let heightPoints: Double

    public init(widthPoints: Double, heightPoints: Double) {
      self.widthPoints = widthPoints
      self.heightPoints = heightPoints
    }
  }

  public let daemonBaseURL: String
  public let expectedChannel: String
  public let expectedStateDirectory: String
  /// Points of native title bar and toolbar drawn over the top of the page, or
  /// nil for a host with an ordinary title bar. The page starts its content
  /// below this strip because the strip takes the clicks there (it drags the
  /// window). Encoded only when present, so the key stays optional for the page.
  public let titlebarInset: Int?
  /// Whether the host can restart the local engine when the page asks
  /// (the Astronomical main app can; the standalone development app cannot), so
  /// the page offers a restart only where pressing it does something.
  public let canRestartEngine: Bool
  /// The paper this Mac prints on (A4 or Letter by region), which the page
  /// lays an export out for so its preview breaks pages where the PDF will.
  /// Nil for a host that does not say; the page then assumes Letter.
  public let printedPaper: PrintedPaper?

  public init(
    daemonBaseURL: String,
    expectedChannel: String,
    expectedStateDirectory: String,
    titlebarInset: Int? = nil,
    canRestartEngine: Bool = false,
    printedPaper: PrintedPaper? = nil
  ) {
    self.daemonBaseURL = daemonBaseURL
    self.expectedChannel = expectedChannel
    self.expectedStateDirectory = expectedStateDirectory
    self.titlebarInset = titlebarInset
    self.canRestartEngine = canRestartEngine
    self.printedPaper = printedPaper
  }

  /// Builds the configuration for one application identity.
  public init(
    applicationIdentity: ThinTalkApplicationIdentity, titlebarInset: Int? = nil, canRestartEngine: Bool = false,
    printedPaper: PrintedPaper? = nil
  ) {
    self.init(
      daemonBaseURL: "http://127.0.0.1:\(applicationIdentity.daemonPort)",
      expectedChannel: applicationIdentity.channel.rawValue,
      expectedStateDirectory: applicationIdentity.expectedServerStateDirectory,
      titlebarInset: titlebarInset,
      canRestartEngine: canRestartEngine,
      printedPaper: printedPaper
    )
  }

  /// The document-start user script that installs the configuration. The page's
  /// AppConfig requires the first three keys and refuses to boot without them;
  /// `titlebarInset`, `canRestartEngine` and `printedPaper` are optional.
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
