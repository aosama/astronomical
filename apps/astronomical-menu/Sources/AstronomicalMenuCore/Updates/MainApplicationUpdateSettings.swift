// What the agent needs to know about updates, read from the main app.
//
// The main app owns the updater and its settings. The agent keeps no update
// settings of its own: it reads the main app's bundle for the feed address
// and the main app's defaults for the choices made in its menu, so the two
// can never disagree and there is nothing new to configure.

import Foundation

/// The update feed and choices in force right now.
struct MainApplicationUpdateSettings: Equatable {
  let feedURL: URL
  let preferences: UpdateNoticePreferences

  /// The updater's own keys, in the main app's bundle and defaults.
  private static let feedURLKey = "SUFeedURL"
  private static let automaticChecksKey = "SUEnableAutomaticChecks"
  private static let skippedBuildKey = "SUSkippedVersion"

  /// Reads the settings, or nil when the main app has no update feed: a
  /// store build, or a bare executable run with no bundle around it.
  ///
  /// A choice the person never made falls back to what the bundle ships
  /// with, which is how the updater itself resolves the same key.
  init?(
    mainApplicationInfo: [String: Any],
    mainApplicationDefaults: UserDefaults,
    applicationChannel: ApplicationChannel
  ) {
    guard let feedAddress = mainApplicationInfo[Self.feedURLKey] as? String,
      let feedURL = URL(string: feedAddress)
    else { return nil }
    self.feedURL = feedURL
    let bundledAutomaticChecks = mainApplicationInfo[Self.automaticChecksKey] as? Bool ?? false
    let chosenAutomaticChecks = mainApplicationDefaults.object(forKey: Self.automaticChecksKey) as? Bool
    let selectedChannel = UpdateChannelSelection.storedChannel(
      in: mainApplicationDefaults, applicationChannel: applicationChannel)
    var skippedBuildNumber: Int?
    if let skippedBuildText = mainApplicationDefaults.string(forKey: Self.skippedBuildKey) {
      skippedBuildNumber = Int(skippedBuildText)
    }
    preferences = UpdateNoticePreferences(
      checksAutomatically: chosenAutomaticChecks ?? bundledAutomaticChecks,
      allowedFeedChannels: sparkleUpdateChannels(for: selectedChannel),
      skippedBuildNumber: skippedBuildNumber)
  }
}

/// Supplies the settings each time they are needed, so a choice changed in
/// the main app's menu takes effect at the agent's next look.
@MainActor
protocol UpdateSettingsReading {
  func currentUpdateSettings() -> MainApplicationUpdateSettings?
}

/// Reads the settings of the main app this agent is nested in.
@MainActor
struct BundledMainApplicationUpdateSettingsReader: UpdateSettingsReading {
  let applicationChannel: ApplicationChannel
  let agentBundleURL: URL

  func currentUpdateSettings() -> MainApplicationUpdateSettings? {
    // A store build's two applications are sandboxed apart, and it takes its
    // updates from the store, so it has neither a feed nor readable defaults.
    guard applicationChannel != .appStore,
      let mainApplicationBundleURL = ApplicationBundleLayout.mainApplicationBundleURL(
        forAgentBundleAt: agentBundleURL),
      let mainApplicationInfo = Bundle(url: mainApplicationBundleURL)?.infoDictionary,
      let mainApplicationDefaults = UserDefaults(suiteName: applicationChannel.applicationBundleIdentifier)
    else { return nil }
    return MainApplicationUpdateSettings(
      mainApplicationInfo: mainApplicationInfo,
      mainApplicationDefaults: mainApplicationDefaults,
      applicationChannel: applicationChannel)
  }
}
