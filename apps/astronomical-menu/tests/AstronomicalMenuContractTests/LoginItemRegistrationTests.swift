import Foundation
import XCTest

@testable import AstronomicalMenuCore

/// Scripts the registration service so switch decisions are verified against
/// every system state the real registrar can report, without touching the
/// machine's actual Login Items registration.
@MainActor
private final class ScriptedLoginItemRegistrar: LoginItemRegistering {
  var launchAtLoginStatus: LoginItemRegistrationStatus
  var registrationError: Error?
  var unregistrationError: Error?
  /// The status the system settles on after a successful register call;
  /// normally enabled, but the system can also park a registration for
  /// approval or leave it stale, and the switch decision must survive both.
  var statusAfterSuccessfulRegister: LoginItemRegistrationStatus = .enabled
  /// Normally deregistered; a throw-free call can still leave the app enabled,
  /// which the switch decision must catch instead of claiming success.
  var statusAfterSuccessfulUnregister: LoginItemRegistrationStatus = .notRegistered

  init(reportedStatus: LoginItemRegistrationStatus = .notRegistered) {
    self.launchAtLoginStatus = reportedStatus
  }

  func registerLaunchAtLogin() throws {
    if let registrationError { throw registrationError }
    launchAtLoginStatus = statusAfterSuccessfulRegister
  }

  func unregisterLaunchAtLogin() throws {
    if let unregistrationError { throw unregistrationError }
    launchAtLoginStatus = statusAfterSuccessfulUnregister
  }

  func openLoginItemsSettings() {}
}

private struct StubbedRegistrationError: Error {}

@MainActor
final class LoginItemRegistrationTests: XCTestCase {
  func test_should_show_the_switch_on_only_when_the_system_enables_the_app() {
    XCTAssertTrue(launchAtLoginToggleIsOn(for: .enabled))
    XCTAssertFalse(launchAtLoginToggleIsOn(for: .notRegistered))
    XCTAssertFalse(launchAtLoginToggleIsOn(for: .requiresApproval))
    XCTAssertFalse(launchAtLoginToggleIsOn(for: .notFound))
  }

  func test_should_offer_launch_at_login_only_on_packaged_product_builds() {
    XCTAssertTrue(
      launchAtLoginControlsSupported(
        applicationChannel: .stable, hostIsPackagedApplication: true))
    XCTAssertTrue(
      launchAtLoginControlsSupported(
        applicationChannel: .appStore, hostIsPackagedApplication: true))
    XCTAssertFalse(
      launchAtLoginControlsSupported(
        applicationChannel: .development, hostIsPackagedApplication: true))
    XCTAssertFalse(
      launchAtLoginControlsSupported(
        applicationChannel: .stable, hostIsPackagedApplication: false))
  }

  func test_should_report_success_when_enabling_confirms_an_enabled_registration() {
    let registrar = ScriptedLoginItemRegistrar(reportedStatus: .notRegistered)

    let outcome = applyLaunchAtLoginChoice(true, using: registrar)

    XCTAssertEqual(outcome, .enabled)
  }

  func test_should_report_success_when_re_enabling_an_already_enabled_registration() {
    let registrar = ScriptedLoginItemRegistrar(reportedStatus: .enabled)

    let outcome = applyLaunchAtLoginChoice(true, using: registrar)

    XCTAssertEqual(outcome, .enabled)
  }

  func test_should_report_pending_approval_when_the_system_holds_the_registration() {
    let registrar = ScriptedLoginItemRegistrar(reportedStatus: .notRegistered)
    registrar.statusAfterSuccessfulRegister = .requiresApproval

    let outcome = applyLaunchAtLoginChoice(true, using: registrar)

    XCTAssertEqual(outcome, .needsSystemApproval)
  }

  func test_should_report_failure_when_registration_throws() {
    let registrar = ScriptedLoginItemRegistrar(reportedStatus: .notRegistered)
    registrar.registrationError = StubbedRegistrationError()

    let outcome = applyLaunchAtLoginChoice(true, using: registrar)

    XCTAssertEqual(outcome, .failed)
  }

  func test_should_report_failure_when_a_throw_free_register_never_becomes_enabled() {
    let registrar = ScriptedLoginItemRegistrar(reportedStatus: .notRegistered)
    registrar.statusAfterSuccessfulRegister = .notRegistered

    let outcome = applyLaunchAtLoginChoice(true, using: registrar)

    XCTAssertEqual(outcome, .failed)
  }

  func test_should_report_success_when_disabling_confirms_deregistration() {
    let registrar = ScriptedLoginItemRegistrar(reportedStatus: .enabled)

    let outcome = applyLaunchAtLoginChoice(false, using: registrar)

    XCTAssertEqual(outcome, .disabled)
  }

  func test_should_report_success_when_disabling_an_already_unregistered_app() {
    // The user can remove Astronomical from Login Items in System Settings
    // while the popover is open; a later switch-off must settle, not error.
    let registrar = ScriptedLoginItemRegistrar(reportedStatus: .notRegistered)

    let outcome = applyLaunchAtLoginChoice(false, using: registrar)

    XCTAssertEqual(outcome, .disabled)
  }

  func test_should_report_failure_when_unregistration_throws() {
    let registrar = ScriptedLoginItemRegistrar(reportedStatus: .enabled)
    registrar.unregistrationError = StubbedRegistrationError()

    let outcome = applyLaunchAtLoginChoice(false, using: registrar)

    XCTAssertEqual(outcome, .failed)
  }

  func test_should_report_failure_when_a_throw_free_unregister_stays_enabled() {
    let registrar = ScriptedLoginItemRegistrar(reportedStatus: .enabled)
    // The system keeps the app enabled even though unregister did not throw:
    // the switch must not claim success while launchd still starts the app.
    registrar.statusAfterSuccessfulUnregister = .enabled

    let outcome = applyLaunchAtLoginChoice(false, using: registrar)

    XCTAssertEqual(outcome, .failed)
  }
}
