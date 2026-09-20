import Foundation
import UniformTypeIdentifiers
import WebKit

/// Serves the canvas shell and registered attachments over the private scheme.
///
/// Serving through a scheme handler rather than `file://` gives the whole page one
/// origin, keeps the vendored scripts and stylesheets resolvable with relative
/// paths, and leaves no file system path anywhere in the rendered document.
public final class CanvasAssetSchemeHandler: NSObject, WKURLSchemeHandler {
  private let webDirectory: URL
  private let assetRegistry: TranscriptAssetRegistry
  private let fileManager = FileManager()

  /// Directories the canvas may read from, keyed by the host that addresses them.
  public init(webDirectory: URL, assetRegistry: TranscriptAssetRegistry) {
    self.webDirectory = webDirectory.standardizedFileURL
    self.assetRegistry = assetRegistry
    super.init()
  }

  public func webView(_ webView: WKWebView, start urlSchemeTask: WKURLSchemeTask) {
    guard let url = urlSchemeTask.request.url,
      let fileURL = resolveFileURL(for: url)
    else {
      urlSchemeTask.didFailWithError(URLError(.unsupportedURL))
      return
    }
    serveFile(at: fileURL, urlSchemeTask: urlSchemeTask)
  }

  public func webView(_ webView: WKWebView, stop urlSchemeTask: WKURLSchemeTask) {
    // Nothing is buffered per task, so there is no in-flight work to cancel.
  }

  private func resolveFileURL(for url: URL) -> URL? {
    switch url.host?.lowercased() {
    case CanvasSecurityPolicy.shellHost:
      return shellFileURL(forPath: url.path)
    case CanvasSecurityPolicy.attachmentHost:
      guard let token = CanvasSecurityPolicy.attachmentToken(from: url) else { return nil }
      return assetRegistry.fileURL(forToken: token)
    default:
      return nil
    }
  }

  /// Resolves a shell path inside the web directory and refuses anything that
  /// escapes it, including encoded `..` traversal.
  private func shellFileURL(forPath path: String) -> URL? {
    let relativePath = path.hasPrefix("/") ? String(path.dropFirst()) : path
    guard !relativePath.isEmpty else { return nil }
    let candidate = webDirectory.appendingPathComponent(relativePath).standardizedFileURL
    let directoryPath = webDirectory.path.hasSuffix("/") ? webDirectory.path : webDirectory.path + "/"
    guard candidate.path.hasPrefix(directoryPath) else { return nil }
    var isDirectory: ObjCBool = false
    guard fileManager.fileExists(atPath: candidate.path, isDirectory: &isDirectory),
      !isDirectory.boolValue
    else {
      return nil
    }
    return candidate
  }

  private func serveFile(at fileURL: URL, urlSchemeTask: WKURLSchemeTask) {
    guard let data = try? Data(contentsOf: fileURL) else {
      urlSchemeTask.didFailWithError(URLError(.fileDoesNotExist))
      return
    }
    let response = URLResponse(
      url: urlSchemeTask.request.url ?? fileURL,
      mimeType: mimeType(for: fileURL),
      expectedContentLength: data.count,
      textEncodingName: "utf-8"
    )
    urlSchemeTask.didReceive(response)
    urlSchemeTask.didReceive(data)
    urlSchemeTask.didFinish()
  }

  private func mimeType(for fileURL: URL) -> String {
    let pathExtension = fileURL.pathExtension
    guard !pathExtension.isEmpty, let type = UTType(filenameExtension: pathExtension) else {
      return "application/octet-stream"
    }
    return type.preferredMIMEType ?? "application/octet-stream"
  }
}