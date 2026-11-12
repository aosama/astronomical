import Foundation
import XCTest

@testable import AstronomicalAppCore

/// Releases before the agent existed registered the whole app as the login
/// item. These tests pin the one-time move onto the agent, including that a
/// failed move leaves the person's existing registration alone.
@MainActor
private final class ScriptedMigratingRegistrar: LoginItemRegistering, WholeApplicationLoginItemRetiring {
  var launchAtLoginStatus = LoginItemRegistrationStatus.notRegistered
  var wholeApplicationStartsAtLogin: Bool
  var statusAfterRegister = LoginItemRegistrationStatus.enabled
  var registrationError: Error?
  private(set) var agentRegistrationCount = 0
  private(set) var wholeApplicationUnregistrationCount = 0

  init(wholeApplicationStartsAtLogin: Bool) {
    self.wholeApplicationStartsAtLogin = wholeApplicationStartsAtLogin
  }

  func registerLaunchAtLogin() throws {
    if let registrationError { throw registrationError }
    agentRegistrationCount += 1
    launchAtLoginStatus = statusAfterRegister
  }

  func unregisterLaunchAtLogin() throws { launchAtLoginStatus = .notRegistered }
  func openLoginItemsSettings() {}

  func unregisterWholeApplicationAtLogin() throws {
    wholeApplicationUnregistrationCount += 1
    wholeApplicationStartsAtLogin = false
  }
}

private struct StubbedMigrationError: Error {}

@MainActor
final class LoginItemMigrationTests: XCTestCase {
  func test_should_move_a_whole_application_registration_onto_the_agent() {
    let registrar = ScriptedMigratingRegistrar(wholeApplicationStartsAtLogin: true)

    moveWholeApplicationLoginItemToAgent(using: registrar)

    XCTAssertEqual(registrar.launchAtLoginStatus, .enabled)
    XCTAssertFalse(registrar.wholeApplicationStartsAtLogin)
    XCTAssertEqual(registrar.wholeApplicationUnregistrationCount, 1)
  }

  func test_should_do_nothing_when_the_whole_application_was_never_registered() {
    let registrar = ScriptedMigratingRegistrar(wholeApplicationStartsAtLogin: false)

    moveWholeApplicationLoginItemToAgent(using: registrar)

    XCTAssertEqual(registrar.agentRegistrationCount, 0)
    XCTAssertEqual(registrar.wholeApplicationUnregistrationCount, 0)
  }

  func test_should_keep_the_old_registration_when_the_agent_cannot_be_registered() {
    let registrar = ScriptedMigratingRegistrar(wholeApplicationStartsAtLogin: true)
    registrar.registrationError = StubbedMigrationError()

    moveWholeApplicationLoginItemToAgent(using: registrar)

    XCTAssertTrue(registrar.wholeApplicationStartsAtLogin, "nothing starting at login is worse than the old item")
    XCTAssertEqual(registrar.wholeApplicationUnregistrationCount, 0)
  }

  /// A registration the system parks for approval still belongs to the agent,
  /// so the old whole-application item is retired.
  func test_should_retire_the_old_registration_when_the_agent_awaits_approval() {
    let registrar = ScriptedMigratingRegistrar(wholeApplicationStartsAtLogin: true)
    registrar.statusAfterRegister = .requiresApproval

    moveWholeApplicationLoginItemToAgent(using: registrar)

    XCTAssertEqual(registrar.wholeApplicationUnregistrationCount, 1)
  }
}
