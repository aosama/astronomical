import Foundation
import os

/// Moves an App Store install's state out of the main app's own container
/// into the app group container, once.
///
/// Before the engine moved to the menu-bar agent, a store build was a single
/// sandboxed application and kept everything (configuration, downloaded
/// models, caches) in its container's Application Support. The agent is a
/// separate sandboxed application and cannot open that container, so the
/// engine it runs would start empty and the models already downloaded would
/// be stranded. Only the main app can still read the old location, which is
/// why the move happens here, before the main app starts the agent.
///
/// Both locations are on the user's volume, so the move is a rename: it takes
/// no time and no extra disk space however many gigabytes of models it covers.
struct ContainerStateMigration {
  enum Outcome: Equatable {
    /// Nothing was ever stored in the main app's container.
    case nothingToMove
    /// The group container already holds state. The two are never merged:
    /// the group container is what the engine is using.
    case alreadyInGroupContainer
    case moved
    case failed(String)
  }

  let legacyStateDirectoryURL: URL
  let groupStateDirectoryURL: URL
  var fileManager: FileManager = .default

  func moveLegacyStateIntoGroupContainer() -> Outcome {
    guard fileManager.fileExists(atPath: legacyStateDirectoryURL.path) else { return .nothingToMove }
    guard !fileManager.fileExists(atPath: groupStateDirectoryURL.path) else {
      return .alreadyInGroupContainer
    }
    do {
      try fileManager.createDirectory(
        at: groupStateDirectoryURL.deletingLastPathComponent(), withIntermediateDirectories: true)
      try fileManager.moveItem(at: legacyStateDirectoryURL, to: groupStateDirectoryURL)
      return .moved
    } catch {
      return .failed(error.localizedDescription)
    }
  }
}

/// Runs the move for a build that has an app group, and does nothing for
/// every other build. A failure is logged and launch carries on: the engine
/// then starts with an empty group container, and the old state stays where
/// it was for the next launch to try again.
@MainActor
func moveContainerStateIntoGroupContainerIfNeeded(
  applicationIdentity: ApplicationIdentity,
  groupContainerURL: ApplicationIdentity.GroupContainerResolver = ApplicationIdentity.systemGroupContainerURL,
  homeDirectoryURL: URL = FileManager.default.homeDirectoryForCurrentUser
) -> ContainerStateMigration.Outcome {
  guard applicationIdentity.applicationGroupContainerURL(groupContainerURL: groupContainerURL) != nil else {
    return .nothingToMove
  }
  let migrationOutcome = ContainerStateMigration(
    legacyStateDirectoryURL: applicationIdentity.legacyContainerStateDirectoryURL(
      homeDirectoryURL: homeDirectoryURL),
    groupStateDirectoryURL: applicationIdentity.stateDirectoryURL(
      homeDirectoryURL: homeDirectoryURL, groupContainerURL: groupContainerURL)
  ).moveLegacyStateIntoGroupContainer()
  if case .failed(let failureDescription) = migrationOutcome {
    Logger(subsystem: "dev.astronomical.app", category: "container-state-migration")
      .error("Existing state could not move to the group container: \(failureDescription, privacy: .public)")
  }
  return migrationOutcome
}
