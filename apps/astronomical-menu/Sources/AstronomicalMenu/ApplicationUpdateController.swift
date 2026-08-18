// Owns Sparkle's update lifecycle while exposing a small seam for menu actions and channel policy tests.

import Foundation
import Sparkle

enum ApplicationUpdateChannel: String, CaseIterable, Hashable, Identifiable {
  case stable
  case releaseCandidate = "release-candidate"
  case development

  var id: String { rawValue }

  var displayName: String {
    switch self {
    case .stable: "Stable"
    case .releaseCandidate: "Release Candidate"
    case .development: "Development"
    }
  }
}

@MainActor
protocol ApplicationUpdateChecking: AnyObject {
  var canCheckForUpdates: Bool { get }
  var automaticallyChecksForUpdates: Bool { get set }

  func checkForUpdates()
}

@MainActor
final class ApplicationUpdateController: NSObject, ApplicationUpdateChecking, SPUUpdaterDelegate {
  private static let selectedChannelUserDefaultsKey = "AstronomicalSelectedUpdateChannel"

  private let userDefaults: UserDefaults
  private var allowedChannels: Set<String>
  private lazy var sparkleUpdaterController = SPUStandardUpdaterController(
    updaterDelegate: self,
    userDriverDelegate: nil
  )

  private(set) var selectedChannel: ApplicationUpdateChannel

  init(applicationChannel: ApplicationChannel, userDefaults: UserDefaults = .standard) {
    self.userDefaults = userDefaults
    let defaultUpdateChannel: ApplicationUpdateChannel =
      applicationChannel == .stable ? .stable : .development
    selectedChannel = userDefaults.string(forKey: Self.selectedChannelUserDefaultsKey)
      .flatMap(ApplicationUpdateChannel.init(rawValue:)) ?? defaultUpdateChannel
    allowedChannels = sparkleUpdateChannels(for: selectedChannel)
  }

  var canCheckForUpdates: Bool {
    sparkleUpdaterController.updater.canCheckForUpdates
  }

  var automaticallyChecksForUpdates: Bool {
    get { sparkleUpdaterController.updater.automaticallyChecksForUpdates }
    set { sparkleUpdaterController.updater.automaticallyChecksForUpdates = newValue }
  }

  func checkForUpdates() {
    sparkleUpdaterController.checkForUpdates(nil)
  }

  func start() {
    // Scheduled checks require eager startup; this lazy owner would otherwise
    // remain dormant until the user first opened the popover.
    _ = sparkleUpdaterController
  }

  func allowedChannels(for updater: SPUUpdater) -> Set<String> {
    allowedChannels
  }

  func selectUpdateChannel(_ selectedChannel: ApplicationUpdateChannel) {
    self.selectedChannel = selectedChannel
    allowedChannels = sparkleUpdateChannels(for: selectedChannel)
    userDefaults.set(selectedChannel.rawValue, forKey: Self.selectedChannelUserDefaultsKey)
    sparkleUpdaterController.updater.resetUpdateCycleAfterShortDelay()
  }
}

@MainActor
func requestManualApplicationUpdateCheck(using updateChecker: any ApplicationUpdateChecking) {
  guard updateChecker.canCheckForUpdates else { return }
  updateChecker.checkForUpdates()
}

@MainActor
func setAutomaticApplicationUpdateChecks(
  _ automaticallyChecksForUpdates: Bool,
  using updateChecker: any ApplicationUpdateChecking
) {
  updateChecker.automaticallyChecksForUpdates = automaticallyChecksForUpdates
}

func sparkleUpdateChannels(for updateChannel: ApplicationUpdateChannel) -> Set<String> {
  switch updateChannel {
  case .stable: []
  case .releaseCandidate: ["release-candidate"]
  case .development: ["release-candidate", "development"]
  }
}
