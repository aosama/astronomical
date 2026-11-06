// Raises Neural Engine banners through macOS notifications. Permission is asked for only when the
// first banner is due, so someone who never enables the Neural Engine setting never sees a prompt.

import Foundation
import UserNotifications

@MainActor
final class NeuralEngineBannerPresenter: NSObject, UNUserNotificationCenterDelegate {
  // One identifier, so the "ready" banner replaces the "prepping" one in Notification Center
  // instead of leaving both behind.
  private static let notificationIdentifier = "neural-engine-status"
  private var pendingBanner: NeuralEngineBanner?
  private var postingTask: Task<Void, Never>?

  override init() {
    super.init()
    // The center requires an app bundle and stops the process without one, which is what a test
    // or a bare executable runs as.
    guard Bundle.main.bundleIdentifier != nil else { return }
    UNUserNotificationCenter.current().delegate = self
  }

  // One task posts at a time, newest banner next. Answering the permission prompt can take longer
  // than the preparation, and "prepping" landing after "ready" would replace it for good.
  func present(_ banner: NeuralEngineBanner) {
    guard Bundle.main.bundleIdentifier != nil else { return }
    pendingBanner = banner
    guard postingTask == nil else { return }
    postingTask = Task { await postPendingBanners() }
  }

  private func postPendingBanners() async {
    while let banner = pendingBanner {
      pendingBanner = nil
      await post(banner)
    }
    postingTask = nil
  }

  private func post(_ banner: NeuralEngineBanner) async {
    let notificationCenter = UNUserNotificationCenter.current()
    do {
      guard try await notificationCenter.requestAuthorization(options: [.alert]) else { return }
      let content = UNMutableNotificationContent()
      content.title = banner.title
      content.body = banner.body
      try await notificationCenter.add(
        UNNotificationRequest(
          identifier: Self.notificationIdentifier, content: content, trigger: nil))
    } catch {
      // The menu and popover already show this state, so a missing banner loses nothing else.
      NSLog("Could not post the Neural Engine banner: \(error.localizedDescription)")
    }
  }

  // macOS hides a banner from the frontmost app by default, and this app's own chat window can be
  // frontmost during the very wait the banner is about.
  nonisolated func userNotificationCenter(
    _ center: UNUserNotificationCenter, willPresent notification: UNNotification
  ) async -> UNNotificationPresentationOptions {
    [.banner]
  }
}
