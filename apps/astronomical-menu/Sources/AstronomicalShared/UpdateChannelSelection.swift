import Foundation

/// Which releases an install follows.
///
/// Shared by both processes: the main app's updater installs from the chosen
/// channel, and the menu-bar agent, which only notices that an update exists,
/// has to judge the same feed by the same choice or it would announce
/// versions the updater then refuses to offer.
public enum ApplicationUpdateChannel: String, CaseIterable, Hashable, Identifiable, Sendable {
  case stable
  case releaseCandidate = "release-candidate"
  case development

  public var id: String { rawValue }

  public var displayName: String {
    switch self {
    case .stable: "Stable"
    case .releaseCandidate: "Release Candidate"
    case .development: "Development"
    }
  }
}

/// The feed channels a selection admits besides the default one. A feed item
/// that names no channel is a Stable release and is offered to everyone.
public func sparkleUpdateChannels(for updateChannel: ApplicationUpdateChannel) -> Set<String> {
  switch updateChannel {
  case .stable: []
  case .releaseCandidate: ["release-candidate"]
  case .development: ["release-candidate", "development"]
  }
}

/// Where the chosen update channel is kept, and what it is before anyone has
/// chosen. The main app writes the key in its own defaults; the agent reads
/// the same key from the main app's defaults.
public enum UpdateChannelSelection {
  public static let userDefaultsKey = "AstronomicalSelectedUpdateChannel"

  public static func defaultChannel(for applicationChannel: ApplicationChannel) -> ApplicationUpdateChannel {
    applicationChannel == .stable ? .stable : .development
  }

  public static func storedChannel(
    in userDefaults: UserDefaults, applicationChannel: ApplicationChannel
  ) -> ApplicationUpdateChannel {
    guard let storedChannelName = userDefaults.string(forKey: userDefaultsKey),
      let storedChannel = ApplicationUpdateChannel(rawValue: storedChannelName)
    else { return defaultChannel(for: applicationChannel) }
    return storedChannel
  }
}
