import Foundation

/// The URL decisions the canvas bridge makes, kept free of WebKit so they can be
/// tested directly.
///
/// The canvas loads only from a private scheme and can never navigate. Links the
/// reader clicks are handed to the system browser instead, and only for schemes a
/// browser can actually open.
public enum CanvasSecurityPolicy {
  /// Private scheme for every asset the canvas loads.
  public static let shellScheme = "thintalk-asset"
  /// Host serving the canvas shell: its own HTML, CSS, JavaScript, and fonts.
  public static let shellHost = "shell"
  /// Host serving files the application registered, addressed by opaque token.
  public static let attachmentHost = "attachment"

  /// Schemes that may leave the application and open in the system browser.
  public static let externallyOpenableSchemes: Set<String> = ["http", "https", "mailto"]

  /// The document the canvas is allowed to display.
  public static let shellDocumentPath = "/index.html"

  public static func shellDocumentURL() -> URL? {
    URL(string: "\(shellScheme)://\(shellHost)\(shellDocumentPath)")
  }

  /// Whether a loaded document is the canvas shell itself.
  public static func isShellDocument(_ url: URL) -> Bool {
    url.scheme?.lowercased() == shellScheme
      && url.host?.lowercased() == shellHost
      && url.path == shellDocumentPath
  }

  /// Whether a request stays inside the private scheme. Anything else is either a
  /// remote fetch or a local file read, and the canvas is allowed neither.
  public static func isInternalAsset(_ url: URL) -> Bool {
    guard url.scheme?.lowercased() == shellScheme else { return false }
    guard let host = url.host?.lowercased() else { return false }
    return host == shellHost || host == attachmentHost
  }

  /// The browser-openable URL behind a clicked link, or nil when the link must be
  /// ignored. Rejecting by allowlist means `javascript:`, `data:`, `file:`, and
  /// application-internal schemes cannot be launched from rendered content.
  public static func externalURL(fromRawValue rawValue: String) -> URL? {
    let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty, let components = URLComponents(string: trimmed) else { return nil }
    guard let scheme = components.scheme?.lowercased() else { return nil }
    guard externallyOpenableSchemes.contains(scheme) else { return nil }
    if scheme != "mailto" {
      guard let host = components.host, !host.isEmpty else { return nil }
    }
    return components.url
  }

  /// The attachment token addressed by an internal URL, or nil when the URL is
  /// not an attachment request.
  public static func attachmentToken(from url: URL) -> String? {
    guard url.scheme?.lowercased() == shellScheme else { return nil }
    guard url.host?.lowercased() == attachmentHost else { return nil }
    let token = url.path.hasPrefix("/") ? String(url.path.dropFirst()) : url.path
    return token.isEmpty ? nil : token
  }
}