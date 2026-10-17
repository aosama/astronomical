// Sparkle-backed update controller for the direct-distribution executable.
//
// Lives in its own target so the App Store executable can link the menu core
// without Sparkle: the store binary never links this framework, which is the
// structural form of the App Review rule that store builds must not contain
// self-updating code (guideline 2.4.5(vii)).
//
// Development launches run this controller as a bare SwiftPM executable with
// no app bundle, and Sparkle cannot start outside a packaged .app: it needs a
// bundle identifier, an SUFeedURL, and its XPC update helpers. Starting it on
// a bare executable only surfaces Sparkle's "Unable to Check For Updates"
// failure alert, so the controller stays dormant unless the host is packaged.

import AstronomicalMenuCore
import Foundation
import Sparkle

@MainActor
public final class SparkleApplicationUpdateController: NSObject, ApplicationUpdateChecking, SPUUpdaterDelegate {
  private static let selectedChannelUserDefaultsKey = "AstronomicalSelectedUpdateChannel"

  private let userDefaults: UserDefaults
  private let isHostPackagedApplication: Bool
  private var allowedChannels: Set<String>
  // Creation defers the updater start (startingUpdater: false); start() is the
  // only path that may call startUpdater(), and only for a packaged host.
  private lazy var sparkleUpdaterController = SPUStandardUpdaterController(
    startingUpdater: false,
    updaterDelegate: self,
    userDriverDelegate: nil
  )

  public private(set) var selectedChannel: ApplicationUpdateChannel

  public init(
    applicationChannel: ApplicationChannel,
    userDefaults: UserDefaults = .standard,
    isHostPackagedApplication: Bool = Bundle.main.bundleIdentifier != nil
  ) {
    self.userDefaults = userDefaults
    self.isHostPackagedApplication = isHostPackagedApplication
    let defaultUpdateChannel: ApplicationUpdateChannel =
      applicationChannel == .stable ? .stable : .development
    selectedChannel = defaultUpdateChannel
    let storedChannelValue = userDefaults.string(forKey: Self.selectedChannelUserDefaultsKey)
    if let storedChannelValue, let storedChannel = ApplicationUpdateChannel(rawValue: storedChannelValue) {
      selectedChannel = storedChannel
    }
    allowedChannels = sparkleUpdateChannels(for: selectedChannel)
  }

  public let supportsUserUpdateControls = true

  public var canCheckForUpdates: Bool {
    guard isHostPackagedApplication else { return false }
    return sparkleUpdaterController.updater.canCheckForUpdates
  }

  public var automaticallyChecksForUpdates: Bool {
    get {
      guard isHostPackagedApplication else { return false }
      return sparkleUpdaterController.updater.automaticallyChecksForUpdates
    }
    set {
      // Ignored while dormant, mirroring the store controller: a stray
      // preference write cannot revive an updater that never started.
      guard isHostPackagedApplication else { return }
      sparkleUpdaterController.updater.automaticallyChecksForUpdates = newValue
    }
  }

  public func checkForUpdates() {
    guard isHostPackagedApplication else { return }
    sparkleUpdaterController.checkForUpdates(nil)
  }

  public func start() {
    // Scheduled checks require eager startup; this lazy owner would otherwise
    // remain dormant until the user first opened the popover.
    guard isHostPackagedApplication else { return }
    sparkleUpdaterController.startUpdater()
  }

  public func allowedChannels(for updater: SPUUpdater) -> Set<String> {
    allowedChannels
  }

  public func selectUpdateChannel(_ selectedChannel: ApplicationUpdateChannel) {
    self.selectedChannel = selectedChannel
    allowedChannels = sparkleUpdateChannels(for: selectedChannel)
    userDefaults.set(selectedChannel.rawValue, forKey: Self.selectedChannelUserDefaultsKey)
    // The channel choice stays persisted while dormant so a later packaged
    // launch picks it up; only a live updater needs its cycle reset.
    guard isHostPackagedApplication else { return }
    sparkleUpdaterController.updater.resetUpdateCycleAfterShortDelay()
  }
}

public func sparkleUpdateChannels(for updateChannel: ApplicationUpdateChannel) -> Set<String> {
  switch updateChannel {
  case .stable: []
  case .releaseCandidate: ["release-candidate"]
  case .development: ["release-candidate", "development"]
  }
}