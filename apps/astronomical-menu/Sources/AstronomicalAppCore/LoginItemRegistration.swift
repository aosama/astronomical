// One shared seam for starting Astronomical at login (issue #1341).
//
// The system's Login Items registration is the single source of truth: the
// menu item reads the live registration status instead of a cached
// preference, so a change made in System Settings under General → Login Items
// & Extensions is reflected the next time the menu opens, and the check mark
// can never disagree with what launchd will actually do.
//
// What starts at login is the menu-bar agent, not the main app: the agent owns
// the engine, so the engine is up after login without a chat window opening.

import Foundation
import ServiceManagement

/// The system's view of whether this application starts at login, mirroring
/// `SMAppService.Status` without exposing the framework type through the seam.
public enum LoginItemRegistrationStatus: Equatable, Sendable {
  case notRegistered
  case enabled
  case requiresApproval
  case notFound
}

/// The verified result of a user's switch flip, so the presenting layer can
/// react without re-reading system state.
public enum LoginItemRegistrationOutcome: Equatable {
  case enabled
  case disabled
  case needsSystemApproval
  case failed
}

@MainActor
public protocol LoginItemRegistering: AnyObject {
  var launchAtLoginStatus: LoginItemRegistrationStatus { get }
  func registerLaunchAtLogin() throws
  func unregisterLaunchAtLogin() throws
  func openLoginItemsSettings()
}

/// Releases before the agent existed registered the whole application as the
/// login item, which would now open a chat window at every login.
@MainActor
public protocol WholeApplicationLoginItemRetiring: AnyObject {
  var wholeApplicationStartsAtLogin: Bool { get }
  func unregisterWholeApplicationAtLogin() throws
}

@MainActor
public final class SystemLoginItemRegistrar: LoginItemRegistering, WholeApplicationLoginItemRetiring {
  // Registers the agent bundled in Contents/Library/LoginItems, the only
  // place macOS looks for a helper named by identifier.
  private let agentLoginItem: SMAppService

  public init(agentBundleIdentifier: String) {
    agentLoginItem = SMAppService.loginItem(identifier: agentBundleIdentifier)
  }

  public var wholeApplicationStartsAtLogin: Bool { SMAppService.mainApp.status == .enabled }

  public func unregisterWholeApplicationAtLogin() throws {
    try SMAppService.mainApp.unregister()
  }

  public var launchAtLoginStatus: LoginItemRegistrationStatus {
    switch agentLoginItem.status {
    case .notRegistered: .notRegistered
    case .enabled: .enabled
    case .requiresApproval: .requiresApproval
    case .notFound: .notFound
    @unknown default: .notRegistered
    }
  }

  public func registerLaunchAtLogin() throws {
    try agentLoginItem.register()
  }

  public func unregisterLaunchAtLogin() throws {
    try agentLoginItem.unregister()
  }

  public func openLoginItemsSettings() {
    SMAppService.openSystemSettingsLoginItems()
  }
}

/// The switch is on only when the system will actually launch the app;
/// every other status — including a registration awaiting the user's approval
/// in System Settings — renders as off rather than promising a launch.
func launchAtLoginToggleIsOn(for registrationStatus: LoginItemRegistrationStatus) -> Bool {
  registrationStatus == .enabled
}

/// The agent supervises the bundled server as a child process, so starting
/// the agent at login is sufficient: the server follows it. The
/// control therefore belongs to the product installations a user keeps.
/// Development launches are contributor workspaces that rebuild and move, and
/// a bare executable run has no bundle for launchd to register, so neither
/// may present a switch the system cannot honor.
func launchAtLoginControlsSupported(
  applicationChannel: ApplicationChannel,
  hostIsPackagedApplication: Bool
) -> Bool {
  hostIsPackagedApplication && applicationChannel != .development
}

/// Applies a switch flip and verifies the system state afterwards: launchd,
/// not the absence of a thrown error, decides whether the switch may show the
/// requested state.
@MainActor
func applyLaunchAtLoginChoice(
  _ isEnabledByUser: Bool,
  using registrar: any LoginItemRegistering
) -> LoginItemRegistrationOutcome {
  do {
    if isEnabledByUser {
      try registrar.registerLaunchAtLogin()
    } else {
      try registrar.unregisterLaunchAtLogin()
    }
  } catch {
    return .failed
  }
  if isEnabledByUser {
    switch registrar.launchAtLoginStatus {
    case .enabled:
      return .enabled
    case .requiresApproval:
      return .needsSystemApproval
    case .notRegistered, .notFound:
      return .failed
    }
  }
  return registrar.launchAtLoginStatus == .notRegistered ? .disabled : .failed
}

/// Moves a whole-application registration from an earlier release onto the
/// agent, once. The old registration is removed only after the agent's is in
/// place, so a failure leaves the person with what they had rather than with
/// nothing starting at login.
@MainActor
func moveWholeApplicationLoginItemToAgent(
  using registrar: any LoginItemRegistering & WholeApplicationLoginItemRetiring
) {
  guard registrar.wholeApplicationStartsAtLogin else { return }
  guard applyLaunchAtLoginChoice(true, using: registrar) != .failed else { return }
  try? registrar.unregisterWholeApplicationAtLogin()
}
