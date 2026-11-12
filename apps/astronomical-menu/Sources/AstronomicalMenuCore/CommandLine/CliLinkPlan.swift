import Foundation

/// The observed state of the terminal-command link destination.
struct CliLinkStatus: Equatable, Sendable {
  let exists: Bool
  let isSymlink: Bool
  // Resolved (dereferenced) destination of an existing symlink, when readable.
  let symlinkDestinationPath: String?
  // Whether the resolved destination exists on disk (false for a dangling link).
  let symlinkDestinationExists: Bool
}

/// What should happen to the terminal-command link this launch.
enum CliLinkAction: Equatable {
  // The destination already resolves to this bundle's CLI; nothing to do.
  case alreadyCurrent
  // No destination exists; create the link (requires admin on this volume).
  case installMissing
  // A link this app owns dangles because the bundle moved or was replaced;
  // recreate it against the current bundle path.
  case repairOwnedDanglingLink
  // The destination exists but was not created by this app (foreign symlink
  // or a regular file); it must never be overwritten.
  case refuseForeignDestination
  // This channel does not own the public terminal command.
  case notSupportedForChannel
}

/// Decides what the installer should do from an observed status. Pure so the
/// ownership contract (issue #825 items 3-5) is testable without touching
/// /usr/local/bin.
enum CliLinkPlan {
  /// Stable owns the public name `astronomical`. Development builds may not
  /// install it: two apps competing for one public command would make the
  /// resolved daemon depend on install order.
  static func linkName(for channel: ApplicationChannel) -> String? {
    switch channel {
    case .stable, .appStore: "astronomical"
    case .development: nil
    }
  }

  static func decide(
    status: CliLinkStatus,
    channel: ApplicationChannel,
    bundleCliPath: String
  ) -> CliLinkAction {
    guard let linkName = linkName(for: channel) else {
      return .notSupportedForChannel
    }
    guard status.exists else {
      return .installMissing
    }
    guard status.isSymlink else {
      // A regular file or directory at the destination was put there by
      // something else; writing through it would destroy foreign state.
      return .refuseForeignDestination
    }
    guard let destinationPath = status.symlinkDestinationPath else {
      return .refuseForeignDestination
    }
    if status.symlinkDestinationExists {
      // Owned links resolve into an Astronomical app bundle; anything else
      // pointing at an unrelated live binary is not ours to replace.
      let alreadyOurs = destinationPath == bundleCliPath
      if alreadyOurs {
        return .alreadyCurrent
      }
      return resolvesIntoOwnedBundle(destinationPath, linkName: linkName)
        ? .repairOwnedDanglingLink : .refuseForeignDestination
    }
    // A dangling link is only ours to repair if it dangles into an owned
    // bundle; a dangling foreign link is evidence of something we cannot
    // reason about.
    return resolvesIntoOwnedBundle(destinationPath, linkName: linkName)
      ? .repairOwnedDanglingLink : .refuseForeignDestination
  }

  // Ownership heuristic for a link destination: it must resolve into the
  // MacOS directory of an app bundle named Astronomical (a trailing variant
  // suffix such as "Astronomical 2.app" appears during updates and renamed
  // installs). Only an Astronomical install puts a CLI named `astronomical`
  // there, so matching the bundle family — not one exact path — is what
  // keeps repair working after the bundle moves or gets renamed.
  static func resolvesIntoOwnedBundle(_ destinationPath: String, linkName: String) -> Bool {
    var pathComponents = destinationPath.split(separator: "/").map(String.init)
    guard let binaryName = pathComponents.popLast(), binaryName == linkName else {
      return false
    }
    guard let macOsDirectoryName = pathComponents.popLast(),
      macOsDirectoryName == "MacOS"
    else {
      return false
    }
    guard let contentsDirectoryName = pathComponents.popLast(),
      contentsDirectoryName == "Contents"
    else {
      return false
    }
    guard let bundleName = pathComponents.popLast(),
      bundleName.hasPrefix("Astronomical"), bundleName.hasSuffix(".app")
    else {
      return false
    }
    return !pathComponents.isEmpty
  }
}
