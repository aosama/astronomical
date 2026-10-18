// One shared seam for starting Astronomical at login (issue #1341).
//
// The system's Login Items registration is the single source of truth: the
// popover switch reads the live registration status instead of a cached
// preference, so a change made in System Settings under General → Login Items
// & Extensions is reflected the next time the popover renders, and the switch
// can never disagree with what launchd will actually do.

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

@MainActor
public final class SystemLoginItemRegistrar: LoginItemRegistering {
  // Registers the application itself, not a bundled helper: the menu app
  // supervises the bundled server as a child process, so the application
  // starting at login brings the whole stack up through the normal launch
  // path, and quitting keeps today's meaning of stopping the server.
  public var launchAtLoginStatus: LoginItemRegistrationStatus {
    switch SMAppService.mainApp.status {
    case .notRegistered: .notRegistered
    case .enabled: .enabled
    case .requiresApproval: .requiresApproval
    case .notFound: .notFound
    @unknown default: .notRegistered
    }
  }

  public func registerLaunchAtLogin() throws {
    try SMAppService.mainApp.register()
  }

  public func unregisterLaunchAtLogin() throws {
    try SMAppService.mainApp.unregister()
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

/// The menu app supervises the bundled server as a child process, so starting
/// the application at login is sufficient: the server follows the app. The
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
