// The agent's one notification delegate.
//
// macOS allows a process a single delegate for its notifications, and the
// agent posts two kinds (Neural Engine status, and an available update), so
// the delegate role lives here rather than in either presenter.

import Foundation
import UserNotifications

@MainActor
final class AgentNotificationRouter: NSObject, UNUserNotificationCenterDelegate {
  /// Runs when someone clicks the "update available" notification.
  var onUpdateNoticeOpened: (@MainActor () -> Void)?

  override init() {
    super.init()
    // The center requires an app bundle and stops the process without one,
    // which is what a test or a bare executable runs as.
    guard Bundle.main.bundleIdentifier != nil else { return }
    UNUserNotificationCenter.current().delegate = self
  }

  // macOS hides a banner from the frontmost app by default, and the agent is
  // frontmost whenever its popover is open, which is also when its news is
  // most likely to arrive.
  nonisolated func userNotificationCenter(
    _ center: UNUserNotificationCenter, willPresent notification: UNNotification
  ) async -> UNNotificationPresentationOptions {
    [.banner]
  }

  nonisolated func userNotificationCenter(
    _ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse
  ) async {
    let openedNotificationIdentifier = response.notification.request.identifier
    guard openedNotificationIdentifier == UpdateNoticePresenter.notificationIdentifier else { return }
    await openUpdateNotice()
  }

  private func openUpdateNotice() {
    onUpdateNoticeOpened?()
  }
}
