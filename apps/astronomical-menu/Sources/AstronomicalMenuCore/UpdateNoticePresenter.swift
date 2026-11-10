// Tells someone, once, that a new version exists, through a macOS notification.

import Foundation
import UserNotifications

/// Posts the "update available" notification. A protocol so the rule for
/// when to announce runs in a test without a permission prompt.
@MainActor
protocol UpdateNoticePresenting {
  /// Returns whether the notification was handed to macOS.
  func present(_ availableUpdate: AvailableUpdate) async -> Bool
}

@MainActor
struct UpdateNoticePresenter: UpdateNoticePresenting {
  /// The router recognises a click on this notification by this identifier.
  /// One identifier also means a newer version's notice replaces an older
  /// one still sitting in Notification Center.
  nonisolated static let notificationIdentifier = "update-available"

  func present(_ availableUpdate: AvailableUpdate) async -> Bool {
    // The center requires an app bundle and stops the process without one,
    // which is what a test or a bare executable runs as.
    guard Bundle.main.bundleIdentifier != nil else { return false }
    let notificationCenter = UNUserNotificationCenter.current()
    do {
      guard try await notificationCenter.requestAuthorization(options: [.alert]) else { return false }
      let content = UNMutableNotificationContent()
      content.title = "Astronomical \(availableUpdate.displayVersion) is available"
      content.body = "Click to see what’s new and update."
      try await notificationCenter.add(
        UNNotificationRequest(
          identifier: Self.notificationIdentifier, content: content, trigger: nil))
      return true
    } catch {
      // The popover row says the same thing, so a missing notification
      // loses nothing else.
      NSLog("Could not post the update notice: \(error.localizedDescription)")
      return false
    }
  }
}
