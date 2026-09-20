import Foundation

/// Locates the bundled canvas shell.
///
/// The shell ships inside this target's resource bundle, so the canvas has no
/// network dependency and no requirement that a particular file exist on disk.
public enum CanvasWebResources {

  public static let directoryName = "web"
  /// The buddy bundle SwiftPM emits for this target's copied resources.
  public static let resourceBundleName = "ThinTalk_ThinTalkCanvas"

  /// Marker used only so `Bundle(for:)` can resolve the bundle that owns this
  /// target's object code at runtime.
  private final class BundleToken {}

  /// The directory holding the shell document and the vendored renderer assets.
  ///
  /// This deliberately does NOT call `Bundle.module`. That generated accessor ends in
  /// `fatalError` when it cannot place the bundle, which killed the host process the
  /// moment ThinTalkCanvas was linked into the menu app; here a missing bundle
  /// degrades to a visible canvas-unavailable state instead.
  ///
  /// The replacement cannot simply mirror `Bundle.module`, because SwiftPM generates
  /// two different accessors. The Swift Build (xcodebuild-routed) one searches the
  /// app, framework, and tool locations. The classic one searches only
  /// `Bundle.main.bundleURL` and then a *baked absolute build path*, so it only works
  /// on the machine that performed the build. Neither may be hardcoded here.
  ///
  /// What both layouts have in common at runtime is where the bundle sits relative to
  /// the code that needs it: inside the host's resources, next to the host bundle, or
  /// as a sibling of the bundle that owns this target's object code (the classic test
  /// and tool placement). These candidates cover those relationships in every host
  /// this package ships into: the app, the test runner, and a command-line tool.
  public static func bundledDirectory() -> URL? {
    let bundleFileName = "\(resourceBundleName).bundle"
    let mainBundle = Bundle.main
    let moduleBundle = Bundle(for: BundleToken.self)

    var candidateDirectories: [URL] = []
    let environment = ProcessInfo.processInfo.environment
    if let overridePath = environment["PACKAGE_RESOURCE_BUNDLE_PATH"] ?? environment["PACKAGE_RESOURCE_BUNDLE_URL"] {
      candidateDirectories.append(URL(fileURLWithPath: overridePath))
    }
    let relationships: [URL?] = [
      // Resources of the host bundle (the packaged app, and the test bundle when the
      // build copies resources into it).
      mainBundle.resourceURL,
      moduleBundle.resourceURL,
      // The host bundle itself, and the bundle owning this target's object code,
      // which is where a command-line tool and a framework-hosted build place it.
      mainBundle.bundleURL,
      moduleBundle.bundleURL,
      // Siblings of those bundles: the classic SwiftPM placement for tests and tools.
      moduleBundle.bundleURL.deletingLastPathComponent(),
      mainBundle.bundleURL.deletingLastPathComponent(),
      // The executables' own directories, for a tool whose bundle URL is not a bundle.
      mainBundle.executableURL?.deletingLastPathComponent(),
      moduleBundle.executableURL?.deletingLastPathComponent(),
    ]
    candidateDirectories.append(contentsOf: relationships.compactMap { $0 })

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