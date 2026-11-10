import AppKit

/// The app menu's settings: how Astronomical updates and whether it starts at
/// login. They sit in the main app's menu because the main app owns both the
/// updater and the login-item registration of its agent.
///
/// Each item's check mark is read from the live source when the menu opens
/// (`validateMenuItem`), never cached, so the menu cannot disagree with the
/// updater or with System Settings.
@MainActor
final class ApplicationSettingsMenu: NSObject, NSMenuItemValidation {
  private let applicationUpdateController: any ApplicationUpdateChecking
  private let loginItemRegistrar: any LoginItemRegistering
  private let launchAtLoginSupported: Bool

  init(
    applicationUpdateController: any ApplicationUpdateChecking,
    loginItemRegistrar: any LoginItemRegistering,
    launchAtLoginSupported: Bool
  ) {
    self.applicationUpdateController = applicationUpdateController
    self.loginItemRegistrar = loginItemRegistrar
    self.launchAtLoginSupported = launchAtLoginSupported
    super.init()
  }

  /// Update items appear only where this build updates itself (never the App
  /// Store build); Launch at Login only where the system can honour it.
  func makeMenuItems() -> [NSMenuItem] {
    var settingsMenuItems: [NSMenuItem] = []
    if applicationUpdateController.supportsUserUpdateControls {
      settingsMenuItems.append(
        makeItem("Check for Updates Automatically", #selector(toggleAutomaticUpdateChecks(_:))))
      let updateChannelItem = NSMenuItem(title: "Update Channel", action: nil, keyEquivalent: "")
      let updateChannelMenu = NSMenu(title: "Update Channel")
      for updateChannel in ApplicationUpdateChannel.allCases {
        let channelItem = makeItem(updateChannel.displayName, #selector(selectUpdateChannel(_:)))
        channelItem.representedObject = updateChannel.rawValue
        updateChannelMenu.addItem(channelItem)
      }
      updateChannelItem.submenu = updateChannelMenu
      settingsMenuItems.append(updateChannelItem)
    }
    if launchAtLoginSupported {
      settingsMenuItems.append(makeItem("Launch at Login", #selector(toggleLaunchAtLogin(_:))))
    }
    return settingsMenuItems
  }

  private func makeItem(_ title: String, _ action: Selector) -> NSMenuItem {
    let settingsMenuItem = NSMenuItem(title: title, action: action, keyEquivalent: "")
    settingsMenuItem.target = self
    return settingsMenuItem
  }

  func validateMenuItem(_ menuItem: NSMenuItem) -> Bool {
    switch menuItem.action {
    case #selector(toggleAutomaticUpdateChecks(_:)):
      menuItem.state = applicationUpdateController.automaticallyChecksForUpdates ? .on : .off
    case #selector(selectUpdateChannel(_:)):
      let isSelectedChannel =
        menuItem.representedObject as? String == applicationUpdateController.selectedChannel.rawValue
      menuItem.state = isSelectedChannel ? .on : .off
    case #selector(toggleLaunchAtLogin(_:)):
      menuItem.state = launchAtLoginToggleIsOn(for: loginItemRegistrar.launchAtLoginStatus) ? .on : .off
    default:
      break
    }
    return true
  }

  @objc func toggleAutomaticUpdateChecks(_ sender: Any?) {
    setAutomaticApplicationUpdateChecks(
      !applicationUpdateController.automaticallyChecksForUpdates, using: applicationUpdateController)
  }

  @objc func selectUpdateChannel(_ sender: NSMenuItem) {
    guard let channelName = sender.representedObject as? String,
      let updateChannel = ApplicationUpdateChannel(rawValue: channelName)
    else { return }
    applicationUpdateController.selectUpdateChannel(updateChannel)
  }

  @objc func toggleLaunchAtLogin(_ sender: Any?) {
    let isEnabledByUser = !launchAtLoginToggleIsOn(for: loginItemRegistrar.launchAtLoginStatus)
    switch applyLaunchAtLoginChoice(isEnabledByUser, using: loginItemRegistrar) {
    case .enabled, .disabled: break
    case .needsSystemApproval:
      presentLoginItemsPrompt(
        messageText: "Astronomical is waiting for approval to start at login",
        informativeText: "Approve Astronomical under System Settings → General → Login Items & Extensions.")
    case .failed:
      presentLoginItemsPrompt(
        messageText: "Astronomical could not change starting at login",
        informativeText:
          "macOS did not accept the change. You can still manage Astronomical under System Settings → General → Login Items & Extensions."
      )
    }
  }

  private func presentLoginItemsPrompt(messageText: String, informativeText: String) {
    let loginItemsPrompt = NSAlert()
    loginItemsPrompt.messageText = messageText
    loginItemsPrompt.informativeText = informativeText
    loginItemsPrompt.addButton(withTitle: "Open Login Items Settings")
    loginItemsPrompt.addButton(withTitle: "Not Now")
    if loginItemsPrompt.runModal() == .alertFirstButtonReturn {
      loginItemRegistrar.openLoginItemsSettings()
    }
  }
}
