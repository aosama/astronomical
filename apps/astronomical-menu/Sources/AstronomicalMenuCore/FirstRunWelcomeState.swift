import Foundation

/// Decides and records whether the menu application opens the chat window by
/// itself on first launch, where Astro's first flight is the welcome. One
/// versioned marker file in the instance state directory is the whole state
/// machine: the window opens by itself at most once, and never again after
/// that — reopening it after an update reads as a bug (issue #610). The marker
/// name is kept from the retired native welcome window, so a Mac that saw
/// that window is not welcomed twice.
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
    // is the chat window opening by itself once more on the next launch.
    try? Data("{}".utf8).write(to: markerFileURL(), options: .atomic)
  }

  private func markerFileURL() -> URL {
    stateDirectoryURL.appendingPathComponent(Self.markerFileName)
  }

  private func markerFileExists() -> Bool {
    fileManager.fileExists(atPath: markerFileURL().path)
  }
}
