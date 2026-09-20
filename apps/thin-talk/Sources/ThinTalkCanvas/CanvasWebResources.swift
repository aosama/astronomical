import Foundation

/// Locates the bundled canvas shell.
///
/// The shell ships inside this target's resource bundle, so the canvas has no
/// network dependency and no requirement that a particular file exist on disk.
public enum CanvasWebResources {

  public static let directoryName = "web"
  /// The buddy bundle SwiftPM emits for this target's copied resources.
  public static let resourceBundleName = "ThinTalk_ThinTalkCanvas"

  /// Marker used only so `Bundle(for:)` can resolve the bundle that owns the
  /// object code of this module at runtime.
  private final class BundleToken {}

  /// The directory holding the shell document and the vendored renderer assets.
  ///
  /// This deliberately does NOT call `Bundle.module`: that SwiftPM-generated accessor
  /// ends in `fatalError` when the bundle is not where it expects, which killed the
  /// host process the moment ThinTalkCanvas was linked into the menu app. Instead it
  /// replicates that accessor's search, in the same order —
  ///   1. the `PACKAGE_RESOURCE_BUNDLE_PATH`/`_URL` environment override that `swift
  ///      test` injects in debug builds (the path the contracts are exercised on),
  ///   2. `Bundle.main.resourceURL`  (package linked into an app),
  ///   3. `Bundle(for:).resourceURL` (package linked into a framework),
  ///   4. `Bundle.main.bundleURL`    (command-line tool),
  /// — but degrades to `nil` rather than trapping, so a missing bundle renders a
  /// visible canvas-unavailable state instead of crashing the process hosting chat.
  public static func bundledDirectory() -> URL? {
    let environment = ProcessInfo.processInfo.environment
    var candidateDirectories: [URL] = []
    if let overridePath = environment["PACKAGE_RESOURCE_BUNDLE_PATH"] ?? environment["PACKAGE_RESOURCE_BUNDLE_URL"] {
      candidateDirectories.append(URL(fileURLWithPath: overridePath))
    }
    if let mainResourcesURL = Bundle.main.resourceURL {
      candidateDirectories.append(mainResourcesURL)
    }
    if let forResourceURL = Bundle(for: BundleToken.self).resourceURL {
      candidateDirectories.append(forResourceURL)
    }
    candidateDirectories.append(Bundle.main.bundleURL)

    let bundleFileName = "\(resourceBundleName).bundle"
    for directory in candidateDirectories {
      let bundleURL = directory.appendingPathComponent(bundleFileName)
      guard let resourceBundle = Bundle(url: bundleURL),
            let webDirectory = resourceBundle.url(forResource: directoryName, withExtension: nil)
      else {
        continue
      }
      return webDirectory
    }
    return nil
  }
}