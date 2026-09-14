import Foundation

/// Decides and records whether the menu application shows its first-run welcome
/// window. One versioned marker file in the instance state directory is the
/// whole state machine: the window must appear at most once per major moment,
/// and never again after the user has seen it once — replaying it after an
/// update reads as a bug (issue #610).
///
/// Existing installs are migrated silently: a state directory that already
/// carries a configuration file proves prior use, so those users are marked
/// acknowledged on first evaluation instead of being interrupted by a welcome
/// they do not need.
struct FirstRunWelcomeAcknowledgmentStore {
  /// Marker format version; bump to re-show the welcome after a future
  /// product moment that deserves one.
  static let markerFileName = "first-run-welcome-v1.json"

  enum Decision: Equatable {
    case showWelcome
    case alreadyAcknowledged
  }

  let stateDirectoryURL: URL
  var fileManager: FileManager = .default

  func welcomeDecision() -> Decision {
    if markerFileExists() {
      return .alreadyAcknowledged
    }
    if fileManager.fileExists(atPath: stateDirectoryURL.appendingPathComponent("config.json").path) {
      acknowledgeWelcome()
      return .alreadyAcknowledged
    }
    return .showWelcome
  }

  func acknowledgeWelcome() {
    try? fileManager.createDirectory(at: stateDirectoryURL, withIntermediateDirectories: true)
    // A failed marker write must not crash the menu application; the worst case
    // is a repeated welcome on the next launch, which the reopenable menu item
    // also covers.
    try? Data("{}".utf8).write(to: markerFileURL(), options: .atomic)
  }

  private func markerFileURL() -> URL {
    stateDirectoryURL.appendingPathComponent(Self.markerFileName)
  }

  private func markerFileExists() -> Bool {
    fileManager.fileExists(atPath: markerFileURL().path)
  }
}
