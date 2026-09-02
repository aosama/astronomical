// Sparkle-backed update controller for the direct-distribution executable.
//
// Lives in its own target so the App Store executable can link the menu core
// without Sparkle: the store binary never links this framework, which is the
// structural form of the App Review rule that store builds must not contain
// self-updating code (guideline 2.4.5(vii)).

import AstronomicalMenuCore
import Foundation
import Sparkle

@MainActor
public final class SparkleApplicationUpdateController: NSObject, ApplicationUpdateChecking, SPUUpdaterDelegate {
  private static let selectedChannelUserDefaultsKey = "AstronomicalSelectedUpdateChannel"

  private let userDefaults: UserDefaults
  private var allowedChannels: Set<String>
  private lazy var sparkleUpdaterController = SPUStandardUpdaterController(
    updaterDelegate: self,
    userDriverDelegate: nil
  )

  public private(set) var selectedChannel: ApplicationUpdateChannel

  public init(applicationChannel: ApplicationChannel, userDefaults: UserDefaults = .standard) {
    self.userDefaults = userDefaults
    let defaultUpdateChannel: ApplicationUpdateChannel =
      applicationChannel == .stable ? .stable : .development
    selectedChannel = userDefaults.string(forKey: Self.selectedChannelUserDefaultsKey)
      .flatMap(ApplicationUpdateChannel.init(rawValue:)) ?? defaultUpdateChannel
    allowedChannels = sparkleUpdateChannels(for: selectedChannel)
  }

  public let supportsUserUpdateControls = true

  public var canCheckForUpdates: Bool {
    sparkleUpdaterController.updater.canCheckForUpdates
  }

  public var automaticallyChecksForUpdates: Bool {
    get { sparkleUpdaterController.updater.automaticallyChecksForUpdates }
    set { sparkleUpdaterController.updater.automaticallyChecksForUpdates = newValue }
  }

  public func checkForUpdates() {
    sparkleUpdaterController.checkForUpdates(nil)
  }

  public func start() {
    // Scheduled checks require eager startup; this lazy owner would otherwise
    // remain dormant until the user first opened the popover.
    _ = sparkleUpdaterController
  }

  public func allowedChannels(for updater: SPUUpdater) -> Set<String> {
    allowedChannels
  }

  public func selectUpdateChannel(_ selectedChannel: ApplicationUpdateChannel) {
    self.selectedChannel = selectedChannel
    allowedChannels = sparkleUpdateChannels(for: selectedChannel)
    userDefaults.set(selectedChannel.rawValue, forKey: Self.selectedChannelUserDefaultsKey)
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