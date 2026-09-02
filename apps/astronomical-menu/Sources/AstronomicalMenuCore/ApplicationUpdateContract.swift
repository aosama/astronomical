// One shared seam for application updates, with App Store semantics as the default.
//
// The direct channel executable installs a Sparkle-backed controller through the
// installer below. Every other build — the App Store executable in particular —
// relies on the default controller, which reports that update controls are not
// user-facing because a store build receives its updates from the store itself.
// This default-first design makes a store binary structurally incapable of
// initiating an update check even if a future executable forgets to configure
// anything, which is the property App Review requires (guideline 2.4.5(vii)).

import Foundation

public enum ApplicationUpdateChannel: String, CaseIterable, Hashable, Identifiable {
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

@MainActor
public protocol ApplicationUpdateChecking: AnyObject {
  /// Whether this build surfaces update controls in the user interface.
  /// Store builds return false so the popover never presents dead controls.
  var supportsUserUpdateControls: Bool { get }
  var canCheckForUpdates: Bool { get }
  var automaticallyChecksForUpdates: Bool { get set }
  var selectedChannel: ApplicationUpdateChannel { get }

  func checkForUpdates()
  func start()
  func selectUpdateChannel(_ selectedChannel: ApplicationUpdateChannel)
}

@MainActor
public enum ApplicationUpdateControllerInstaller {
  private static var factory: ((ApplicationChannel) -> any ApplicationUpdateChecking)?

  /// Installs the executable's update controller factory. The direct channel
  /// calls this once from its App entry point before launch completes; the
  /// App Store executable never calls it and therefore always receives the
  /// store-semantics default.
  public static func install(
    _ controllerFactory: @escaping (ApplicationChannel) -> any ApplicationUpdateChecking
  ) {
    factory = controllerFactory
  }

  public static func makeController(
    applicationChannel: ApplicationChannel
  ) -> any ApplicationUpdateChecking {
    factory?(applicationChannel) ?? AppStoreUpdateController()
  }
}

@MainActor
final class AppStoreUpdateController: ApplicationUpdateChecking {
  let supportsUserUpdateControls = false
  let canCheckForUpdates = false
  // Store builds never check for updates automatically; writes are ignored so
  // a stray preference write cannot re-enable in-process update checks.
  var automaticallyChecksForUpdates: Bool {
    get { false }
    set {}
  }
  private(set) var selectedChannel = ApplicationUpdateChannel.stable

  func checkForUpdates() {
    // Store builds never check for updates in-process; the App Store owns the
    // update lifecycle.
  }

  func start() {}

  func selectUpdateChannel(_: ApplicationUpdateChannel) {
    // Inert by design; the channel picker is hidden in store builds.
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