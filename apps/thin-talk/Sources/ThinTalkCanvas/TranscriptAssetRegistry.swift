import Foundation

/// Maps opaque tokens to files the application chose to expose to the canvas.
///
/// The canvas renders model output, so it is never given a file path. Swift
/// registers a file it already controls and sends the page a token; the scheme
/// handler refuses every token that was not registered, which keeps a fabricated
/// image URL from reading anything on disk.
public final class TranscriptAssetRegistry: @unchecked Sendable {
  private let lock = NSLock()
  private var fileURLsByToken: [String: URL] = [:]

  public init() {}

  /// Registers a readable file and returns the token that addresses it.
  public func register(fileURL: URL) -> String {
    let token = UUID().uuidString.lowercased()
    lock.lock()
    fileURLsByToken[token] = fileURL
    lock.unlock()
    return token
  }

  /// The asset URL the canvas should use for a registered token.
  public func assetURL(forToken token: String) -> String {
    "\(CanvasSecurityPolicy.shellScheme)://\(CanvasSecurityPolicy.attachmentHost)/\(token)"
  }

  /// Registers a file and returns the URL the canvas can reference.
  public func registerAssetURL(fileURL: URL) -> String {
    assetURL(forToken: register(fileURL: fileURL))
  }

  public func fileURL(forToken token: String) -> URL? {
    lock.lock()
    defer { lock.unlock() }
    return fileURLsByToken[token]
  }
}