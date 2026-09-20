import Foundation

/// Locates the bundled canvas shell.
///
/// The shell ships inside this target's resource bundle, so the canvas has no
/// network dependency and no requirement that a particular file exist on disk.
/// The generated `Bundle.module` accessor is deliberately NOT used: it traps
/// (EXC_BREAKPOINT) whenever the resource bundle is not immediately findable,
/// which killed the host app when this target is linked into astronomical-menu.
/// A manual buddy-bundle lookup returns nil instead, and the presenter shows a
/// visible canvas-unavailable state.
public enum CanvasWebResources {
  public static let directoryName = "web"
  /// The buddy bundle SwiftPM generates for this target's copied resources.
  public static let resourceBundleName = "ThinTalk_ThinTalkCanvas"

  /// The directory holding the shell document and the vendored renderer assets.
  public static func bundledDirectory() -> URL? {
    for bundle in Bundle.allBundles + [Bundle.main] {
      guard let bundleURL = bundle.url(forResource: resourceBundleName, withExtension: "bundle"),
        let resourceBundle = Bundle(url: bundleURL),
        let webDirectory = resourceBundle.url(forResource: directoryName, withExtension: nil)
      else { continue }
      return webDirectory
    }
    return nil
  }
}