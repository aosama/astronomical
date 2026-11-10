// Decides whether the feed holds an update worth telling someone about.
// Kept apart from the network and from the settings store so every rule here
// runs in a test.

import Foundation

/// A newer version the agent may mention.
struct AvailableUpdate: Equatable, Sendable {
  let displayVersion: String
  let buildNumber: Int
}

/// The update choices a person made in the main app, as the agent needs them.
struct UpdateNoticePreferences: Equatable {
  /// "Check for Updates Automatically". Off means the agent neither fetches
  /// the feed nor shows a notice.
  let checksAutomatically: Bool
  /// The feed channels the selected update channel admits besides Stable.
  let allowedFeedChannels: Set<String>
  /// The build a person chose to skip in the updater's dialog, if any.
  let skippedBuildNumber: Int?
}

enum AvailableUpdatePolicy {
  /// The newest release the main app's updater would offer, or nil when
  /// there is none. Mirrors the updater's own filters, so the agent never
  /// announces a version that "Update…" then fails to offer:
  /// - only builds newer than the installed one;
  /// - only the selected channel's releases;
  /// - only releases this macOS can run;
  /// - nothing at or below a skipped build, which is how the updater reads
  ///   "Skip This Version".
  static func newestApplicableUpdate(
    feedItems: [UpdateFeedItem],
    installedBuildNumber: Int,
    preferences: UpdateNoticePreferences,
    operatingSystemVersion: OperatingSystemVersion
  ) -> AvailableUpdate? {
    guard preferences.checksAutomatically else { return nil }
    var newestApplicableItem: UpdateFeedItem?
    for feedItem in feedItems {
      guard feedItem.buildNumber > installedBuildNumber,
        admitsChannel(of: feedItem, allowedFeedChannels: preferences.allowedFeedChannels),
        runsOn(operatingSystemVersion, minimumSystemVersion: feedItem.minimumSystemVersion)
      else { continue }
      if let skippedBuildNumber = preferences.skippedBuildNumber,
        feedItem.buildNumber <= skippedBuildNumber
      {
        continue
      }
      if let currentNewestItem = newestApplicableItem,
        currentNewestItem.buildNumber >= feedItem.buildNumber
      {
        continue
      }
      newestApplicableItem = feedItem
    }
    guard let newestApplicableItem else { return nil }
    return AvailableUpdate(
      displayVersion: newestApplicableItem.displayVersion,
      buildNumber: newestApplicableItem.buildNumber)
  }

  private static func admitsChannel(
    of feedItem: UpdateFeedItem, allowedFeedChannels: Set<String>
  ) -> Bool {
    guard let channelName = feedItem.channelName else { return true }
    return allowedFeedChannels.contains(channelName)
  }

  /// A release with no stated minimum, or one whose minimum cannot be read,
  /// is taken to run here: the updater makes the binding decision, and a
  /// notice that leads to "you're up to date" costs less than a missed one.
  private static func runsOn(
    _ operatingSystemVersion: OperatingSystemVersion, minimumSystemVersion: String?
  ) -> Bool {
    guard let minimumSystemVersion else { return true }
    var minimumVersionParts: [Int] = []
    for versionPartText in minimumSystemVersion.split(separator: ".") {
      guard let versionPart = Int(versionPartText) else { return true }
      minimumVersionParts.append(versionPart)
    }
    guard !minimumVersionParts.isEmpty else { return true }
    let runningVersionParts = [
      operatingSystemVersion.majorVersion, operatingSystemVersion.minorVersion,
      operatingSystemVersion.patchVersion,
    ]
    for (partIndex, runningVersionPart) in runningVersionParts.enumerated() {
      let minimumVersionPart = partIndex < minimumVersionParts.count ? minimumVersionParts[partIndex] : 0
      if runningVersionPart != minimumVersionPart { return runningVersionPart > minimumVersionPart }
    }
    return true
  }
}
