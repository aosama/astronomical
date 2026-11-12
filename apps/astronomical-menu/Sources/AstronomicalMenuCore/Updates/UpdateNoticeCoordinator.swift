// Runs the agent's daily look at the update feed and announces what it finds.
//
// The main app owns the updater, and its scheduled checks run only while it
// is open. The agent is open from login, so without this someone who uses
// Astronomical only from the menu bar, Observatory or the command line would
// never hear about a new version. The agent only notices and tells: both the
// popover row and the notification open the main app's updater, which is the
// one thing that downloads and installs.

import Foundation

@MainActor
final class UpdateNoticeCoordinator {
  /// How long after launch the first look happens. The agent starts at
  /// login, when the network is often not up yet and the engine is starting;
  /// the feed can wait for both.
  static let firstCheckDelay: Duration = .seconds(30)
  /// One look a day, the same rhythm as the updater's own scheduled check.
  static let checkIntervalSeconds: TimeInterval = 86_400
  /// macOS may move the daily look by this much to batch it with other work.
  static let checkToleranceSeconds: TimeInterval = 3_600

  let store: UpdateNoticeStore
  private let presenter: any UpdateNoticePresenting
  private let announcementRecord: UpdateAnnouncementRecord
  private let mainApplicationPresence: any MainApplicationPresenceChecking
  private let activityIdentifier: String
  private var firstCheckTask: Task<Void, Never>?
  private var isLooking = false
  private var dailyCheckScheduler: NSBackgroundActivityScheduler?

  init(
    store: UpdateNoticeStore,
    presenter: any UpdateNoticePresenting,
    announcementRecord: UpdateAnnouncementRecord,
    mainApplicationPresence: any MainApplicationPresenceChecking,
    activityIdentifier: String
  ) {
    self.store = store
    self.presenter = presenter
    self.announcementRecord = announcementRecord
    self.mainApplicationPresence = mainApplicationPresence
    self.activityIdentifier = activityIdentifier
  }

  /// Looks once shortly after launch, then once a day for as long as the
  /// agent runs. The system scheduler is used rather than a sleeping task
  /// because it keeps counting across sleep and picks a moment when the Mac
  /// is not busy or on low power.
  func start() {
    guard firstCheckTask == nil else { return }
    firstCheckTask = Task { [weak self] in
      try? await Task.sleep(for: Self.firstCheckDelay)
      guard !Task.isCancelled else { return }
      await self?.checkAndAnnounce()
    }
    let scheduler = NSBackgroundActivityScheduler(identifier: activityIdentifier)
    scheduler.repeats = true
    scheduler.interval = Self.checkIntervalSeconds
    scheduler.tolerance = Self.checkToleranceSeconds
    scheduler.qualityOfService = .utility
    scheduler.schedule { [weak self] finishActivity in
      Task { @MainActor in
        await self?.checkAndAnnounce()
        finishActivity(.finished)
      }
    }
    dailyCheckScheduler = scheduler
  }

  func stop() {
    firstCheckTask?.cancel()
    firstCheckTask = nil
    dailyCheckScheduler?.invalidate()
    dailyCheckScheduler = nil
  }

  /// One look at the feed. The popover row follows the store by itself; the
  /// notification is posted here, once per version, and only while the main
  /// app is closed. A version is recorded as announced only after macOS took
  /// the notification, so one that could not be shown today (the main app
  /// was open, or permission was not given yet) is tried again tomorrow.
  ///
  /// The first look and a scheduled one can fall together; the second is
  /// dropped, or both would reach the notification before either recorded it.
  func checkAndAnnounce() async {
    guard !isLooking else { return }
    isLooking = true
    defer { isLooking = false }
    await store.checkFeed()
    guard let availableUpdate = store.availableUpdate,
      announcementRecord.needsAnnouncement(of: availableUpdate),
      !mainApplicationPresence.isMainApplicationRunning
    else { return }
    if await presenter.present(availableUpdate) {
      announcementRecord.recordAnnouncement(of: availableUpdate)
    }
  }
}
