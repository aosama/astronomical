import Foundation

/// Locates the bundled canvas shell.
///
/// The shell ships inside this target's resource bundle, so the canvas has no
/// network dependency and no requirement that a particular file exist on disk.
public enum CanvasWebResources {
  public static let directoryName = "web"

  /// The directory holding the shell document and the vendored renderer assets.
  public static func bundledDirectory() -> URL? {
    Bundle.module.url(forResource: directoryName, withExtension: nil)
  }
}