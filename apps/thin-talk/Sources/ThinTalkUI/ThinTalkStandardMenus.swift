import AppKit

/// The menus every Mac app carries, which hold nothing specific to a
/// conversation: the application menu, Edit, Window and Help.
@MainActor
enum ThinTalkStandardMenus {
  static func applicationMenu(
    hostActions: ThinTalkHostMenuActions, conversationActions: ThinTalkConversationActions
  ) -> NSMenu {
    let applicationName = hostActions.applicationName
    let applicationMenu = NSMenu(title: applicationName)
    applicationMenu.addItem(
      ThinTalkMainMenu.menuItem(
        "About \(applicationName)", #selector(NSApplication.orderFrontStandardAboutPanel(_:)), "", nil))
    if let checkForUpdates = hostActions.checkForUpdates {
      applicationMenu.addItem(closureItem("Check for Updates…", "", checkForUpdates))
    }
    for hostSettingsMenuItem in hostActions.hostSettingsMenuItems {
      applicationMenu.addItem(hostSettingsMenuItem)
    }
    applicationMenu.addItem(.separator())
    // Settings opens Your Astro in the page, which holds every choice the
    // window has: the welcome's answers and whether Astro is shown at all.
    applicationMenu.addItem(
      ThinTalkMainMenu.menuItem(
        "Settings…", #selector(ThinTalkConversationActions.openYourAstro(_:)), ",", conversationActions))
    applicationMenu.addItem(.separator())
    if let openObservatory = hostActions.openObservatory {
      applicationMenu.addItem(closureItem("Open Observatory", "", openObservatory))
      applicationMenu.addItem(.separator())
    }
    // AppKit fills this submenu once `ThinTalkMainMenu.install` hands it over.
    let servicesItem = NSMenuItem(title: "Services", action: nil, keyEquivalent: "")
    servicesItem.submenu = NSMenu(title: "Services")
    applicationMenu.addItem(servicesItem)
    applicationMenu.addItem(.separator())
    applicationMenu.addItem(
      ThinTalkMainMenu.menuItem("Hide \(applicationName)", #selector(NSApplication.hide(_:)), "h", nil))
    let hideOthersItem = ThinTalkMainMenu.menuItem(
      "Hide Others", #selector(NSApplication.hideOtherApplications(_:)), "h", nil)
    hideOthersItem.keyEquivalentModifierMask = [.command, .option]
    applicationMenu.addItem(hideOthersItem)
    applicationMenu.addItem(
      ThinTalkMainMenu.menuItem("Show All", #selector(NSApplication.unhideAllApplications(_:)), "", nil))
    applicationMenu.addItem(.separator())
    applicationMenu.addItem(
      ThinTalkMainMenu.menuItem("Quit \(applicationName)", #selector(NSApplication.terminate(_:)), "q", nil))
    // ⌘Q closes the chat app and leaves the engine serving other clients;
    // this is the way to stop both from here. It has no shortcut: ⌥⌘Q is
    // already AppKit's "Quit and Keep Windows", which it adds beside Quit.
    if let quitEverything = hostActions.quitEverything {
      applicationMenu.addItem(closureItem("Quit and Stop Engine", "", quitEverything))
    }
    return applicationMenu
  }

  static func editMenu() -> NSMenu {
    let editMenu = NSMenu(title: "Edit")
    editMenu.addItem(ThinTalkMainMenu.menuItem("Undo", Selector(("undo:")), "z", nil))
    let redoItem = ThinTalkMainMenu.menuItem("Redo", Selector(("redo:")), "z", nil)
    redoItem.keyEquivalentModifierMask = [.command, .shift]
    editMenu.addItem(redoItem)
    editMenu.addItem(.separator())
    editMenu.addItem(ThinTalkMainMenu.menuItem("Cut", #selector(NSText.cut(_:)), "x", nil))
    editMenu.addItem(ThinTalkMainMenu.menuItem("Copy", #selector(NSText.copy(_:)), "c", nil))
    editMenu.addItem(ThinTalkMainMenu.menuItem("Paste", #selector(NSText.paste(_:)), "v", nil))
    let plainPasteItem = ThinTalkMainMenu.menuItem(
      "Paste and Match Style", #selector(NSTextView.pasteAsPlainText(_:)), "v", nil)
    plainPasteItem.keyEquivalentModifierMask = [.command, .option, .shift]
    editMenu.addItem(plainPasteItem)
    editMenu.addItem(ThinTalkMainMenu.menuItem("Select All", #selector(NSText.selectAll(_:)), "a", nil))
    // No Delete item: the web view does not answer `delete:`, so AppKit would
    // show it permanently disabled. AppKit appends AutoFill, Start Dictation
    // and Emoji & Symbols to the menu titled Edit by itself; adding them here
    // would show them twice.
    return editMenu
  }

  static func windowMenu() -> NSMenu {
    let windowMenu = NSMenu(title: "Window")
    windowMenu.addItem(
      ThinTalkMainMenu.menuItem("Minimize", #selector(NSWindow.performMiniaturize(_:)), "m", nil))
    windowMenu.addItem(ThinTalkMainMenu.menuItem("Zoom", #selector(NSWindow.performZoom(_:)), "", nil))
    windowMenu.addItem(.separator())
    windowMenu.addItem(
      ThinTalkMainMenu.menuItem("Bring All to Front", #selector(NSApplication.arrangeInFront(_:)), "", nil))
    return windowMenu
  }

  static func helpMenu(hostActions: ThinTalkHostMenuActions) -> NSMenu {
    let helpMenu = NSMenu(title: "Help")
    if let helpURL = hostActions.helpURL {
      let openHelp: @MainActor () -> Void = { NSWorkspace.shared.open(helpURL) }
      helpMenu.addItem(closureItem("\(hostActions.applicationName) Help", "?", openHelp))
    }
    return helpMenu
  }

  /// A menu item that runs a host-supplied action. NSMenuItem holds its target
  /// weakly, so the item also keeps the handler as its represented object; the
  /// handler then lives exactly as long as the item.
  static func closureItem(_ title: String, _ keyEquivalent: String, _ perform: @escaping @MainActor () -> Void)
    -> NSMenuItem
  {
    let hostActionHandler = MenuActionHandler(perform: perform)
    let hostActionMenuItem = NSMenuItem(
      title: title, action: #selector(MenuActionHandler.performAction(_:)), keyEquivalent: keyEquivalent)
    hostActionMenuItem.target = hostActionHandler
    hostActionMenuItem.representedObject = hostActionHandler
    return hostActionMenuItem
  }
}

/// Adapts a host closure to AppKit's target-action menu items.
@MainActor
final class MenuActionHandler: NSObject {
  private let perform: @MainActor () -> Void

  init(perform: @escaping @MainActor () -> Void) {
    self.perform = perform
  }

  @objc func performAction(_ sender: Any?) {
    perform()
  }
}
