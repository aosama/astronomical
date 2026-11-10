import AppKit
import XCTest

@testable import AstronomicalAppCore

@MainActor
private final class ScriptedUpdateController: ApplicationUpdateChecking {
  let supportsUserUpdateControls: Bool
  let canCheckForUpdates = true
  var automaticallyChecksForUpdates = true
  private(set) var selectedChannel = ApplicationUpdateChannel.stable
  var prepareForRelaunch: (@MainActor () -> Void)?
  let installsUpdateOnQuit = false

  init(supportsUserUpdateControls: Bool) {
    self.supportsUserUpdateControls = supportsUserUpdateControls
  }

  func checkForUpdates() {}
  func start() {}
  func selectUpdateChannel(_ selectedChannel: ApplicationUpdateChannel) { self.selectedChannel = selectedChannel }
}

@MainActor
private final class ScriptedLoginRegistrar: LoginItemRegistering {
  var launchAtLoginStatus = LoginItemRegistrationStatus.notRegistered
  func registerLaunchAtLogin() throws { launchAtLoginStatus = .enabled }
  func unregisterLaunchAtLogin() throws { launchAtLoginStatus = .notRegistered }
  func openLoginItemsSettings() {}
}

/// The app menu's settings read their check marks from the live source each
/// time the menu opens, and offer only what this build can actually do.
@MainActor
final class ApplicationSettingsMenuTests: XCTestCase {
  private func makeSettingsMenu(
    supportsUserUpdateControls: Bool = true, launchAtLoginSupported: Bool = true
  ) -> (ApplicationSettingsMenu, ScriptedUpdateController, ScriptedLoginRegistrar) {
    let updateController = ScriptedUpdateController(supportsUserUpdateControls: supportsUserUpdateControls)
    let loginRegistrar = ScriptedLoginRegistrar()
    let settingsMenu = ApplicationSettingsMenu(
      applicationUpdateController: updateController, loginItemRegistrar: loginRegistrar,
      launchAtLoginSupported: launchAtLoginSupported)
    return (settingsMenu, updateController, loginRegistrar)
  }

  func test_should_offer_update_and_login_items_where_both_are_supported() {
    let (settingsMenu, _, _) = makeSettingsMenu()

    XCTAssertEqual(
      settingsMenu.makeMenuItems().map(\.title),
      ["Check for Updates Automatically", "Update Channel", "Launch at Login"])
  }

  func test_should_offer_no_update_items_in_a_store_build_and_no_login_item_where_unsupported() {
    let (storeSettingsMenu, _, _) = makeSettingsMenu(supportsUserUpdateControls: false)
    let (developmentSettingsMenu, _, _) = makeSettingsMenu(launchAtLoginSupported: false)
    let (bareSettingsMenu, _, _) = makeSettingsMenu(supportsUserUpdateControls: false, launchAtLoginSupported: false)

    XCTAssertEqual(storeSettingsMenu.makeMenuItems().map(\.title), ["Launch at Login"])
    XCTAssertEqual(
      developmentSettingsMenu.makeMenuItems().map(\.title), ["Check for Updates Automatically", "Update Channel"])
    XCTAssertTrue(bareSettingsMenu.makeMenuItems().isEmpty)
  }

  func test_should_check_exactly_the_selected_update_channel() throws {
    let (settingsMenu, updateController, _) = makeSettingsMenu()
    let channelItems = try XCTUnwrap(settingsMenu.makeMenuItems()[1].submenu).items
    updateController.selectUpdateChannel(.releaseCandidate)

    for channelItem in channelItems { _ = settingsMenu.validateMenuItem(channelItem) }

    XCTAssertEqual(channelItems.map(\.title), ["Stable", "Release Candidate", "Development"])
    XCTAssertEqual(channelItems.map(\.state), [.off, .on, .off])
  }

  func test_should_select_the_channel_of_the_chosen_item() throws {
    let (settingsMenu, updateController, _) = makeSettingsMenu()
    let developmentChannelItem = try XCTUnwrap(settingsMenu.makeMenuItems()[1].submenu).items[2]

    settingsMenu.selectUpdateChannel(developmentChannelItem)

    XCTAssertEqual(updateController.selectedChannel, .development)
  }

  func test_should_flip_automatic_checks_and_show_the_live_state() {
    let (settingsMenu, updateController, _) = makeSettingsMenu()
    let automaticChecksItem = settingsMenu.makeMenuItems()[0]
    _ = settingsMenu.validateMenuItem(automaticChecksItem)
    XCTAssertEqual(automaticChecksItem.state, .on)

    settingsMenu.toggleAutomaticUpdateChecks(nil)
    _ = settingsMenu.validateMenuItem(automaticChecksItem)

    XCTAssertFalse(updateController.automaticallyChecksForUpdates)
    XCTAssertEqual(automaticChecksItem.state, .off)
  }

  func test_should_register_and_unregister_the_agent_from_the_login_item() {
    let (settingsMenu, _, loginRegistrar) = makeSettingsMenu()
    let launchAtLoginItem = settingsMenu.makeMenuItems()[2]

    settingsMenu.toggleLaunchAtLogin(nil)
    _ = settingsMenu.validateMenuItem(launchAtLoginItem)
    XCTAssertEqual(loginRegistrar.launchAtLoginStatus, .enabled)
    XCTAssertEqual(launchAtLoginItem.state, .on)

    settingsMenu.toggleLaunchAtLogin(nil)
    _ = settingsMenu.validateMenuItem(launchAtLoginItem)
    XCTAssertEqual(loginRegistrar.launchAtLoginStatus, .notRegistered)
    XCTAssertEqual(launchAtLoginItem.state, .off)
  }

  /// A check mark shows only when the system will launch the agent; a
  /// registration parked for approval is not a promise.
  func test_should_show_launch_at_login_off_while_the_system_awaits_approval() {
    let (settingsMenu, _, loginRegistrar) = makeSettingsMenu()
    let launchAtLoginItem = settingsMenu.makeMenuItems()[2]
    loginRegistrar.launchAtLoginStatus = .requiresApproval

    _ = settingsMenu.validateMenuItem(launchAtLoginItem)

    XCTAssertEqual(launchAtLoginItem.state, .off)
  }
}
