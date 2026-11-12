import Foundation

/// Remembers which update was already announced, so one version produces one
/// notification however many daily checks see it and however often the agent
/// restarts in between.
struct UpdateAnnouncementRecord {
  static let announcedBuildKey = "AstronomicalAnnouncedUpdateBuildNumber"

  let userDefaults: UserDefaults

  /// True for a build newer than every build announced so far. A missing
  /// record reads as zero, which every real build number exceeds.
  func needsAnnouncement(of availableUpdate: AvailableUpdate) -> Bool {
    availableUpdate.buildNumber > userDefaults.integer(forKey: Self.announcedBuildKey)
  }

  func recordAnnouncement(of availableUpdate: AvailableUpdate) {
    userDefaults.set(availableUpdate.buildNumber, forKey: Self.announcedBuildKey)
  }
}
