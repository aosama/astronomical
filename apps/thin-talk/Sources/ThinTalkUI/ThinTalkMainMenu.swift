import AppKit

/// What the hosting application adds to the menu bar beyond the conversation:
/// its name, and the app-level commands only that host can perform.
public struct ThinTalkHostMenuActions {
  public let applicationName: String
  /// Shown only when the host can update itself (not the App Store build).
  public let checkForUpdates: (@MainActor () -> Void)?
  public let openObservatory: (@MainActor () -> Void)?
  public let helpURL: URL?

  public init(
    applicationName: String,
    checkForUpdates: (@MainActor () -> Void)? = nil,
    openObservatory: (@MainActor () -> Void)? = nil,
    helpURL: URL? = nil
  ) {
    self.applicationName = applicationName
    self.checkForUpdates = checkForUpdates
    self.openObservatory = openObservatory
    self.helpURL = helpURL
  }
}

/// Builds the menu bar the conversation window installs: the standard macOS
/// menus plus a Conversation menu, in the order Apple's chat apps use.
///
/// The Edit menu is not decoration. AppKit delivers ⌘C, ⌘V, ⌘X, ⌘A and ⌘Z to
/// the focused view through the Edit menu's key equivalents, so without it
/// those keys do nothing in the message field. Its items therefore target the
/// responder chain (nil target), where the web view answers them.
@MainActor
public enum ThinTalkMainMenu {
  public static func make(
    hostActions: ThinTalkHostMenuActions,
    conversationActions: ThinTalkConversationActions
  ) -> NSMenu {
    let mainMenu = NSMenu(title: "Main Menu")
    let windowMenu = ThinTalkStandardMenus.windowMenu()
    let helpMenu = ThinTalkStandardMenus.helpMenu(hostActions: hostActions)
    for submenu in [
      ThinTalkStandardMenus.applicationMenu(hostActions: hostActions),
      fileMenu(conversationActions: conversationActions),
      ThinTalkStandardMenus.editMenu(),
      viewMenu(conversationActions: conversationActions),
      conversationMenu(conversationActions: conversationActions),
      windowMenu,
      helpMenu,
    ] {
      let menuBarItem = NSMenuItem(title: submenu.title, action: nil, keyEquivalent: "")
      menuBarItem.submenu = submenu
      mainMenu.addItem(menuBarItem)
    }
    return mainMenu
  }

  /// Makes `mainMenu` the application's menu bar and hands AppKit the Window,
  /// Help and Services menus, so it lists open windows, adds its help search
  /// field and fills in the system services.
  public static func install(_ mainMenu: NSMenu, in application: NSApplication) {
    application.mainMenu = mainMenu
    application.windowsMenu = mainMenu.item(withTitle: "Window")?.submenu
    application.helpMenu = mainMenu.item(withTitle: "Help")?.submenu
    application.servicesMenu = mainMenu.items.first?.submenu?.item(withTitle: "Services")?.submenu
  }

  static func fileMenu(conversationActions: ThinTalkConversationActions) -> NSMenu {
    let fileMenu = NSMenu(title: "File")
    fileMenu.addItem(
      menuItem("New Chat", #selector(ThinTalkConversationActions.newChat(_:)), "n", conversationActions))
    fileMenu.addItem(.separator())
    fileMenu.addItem(menuItem("Close Window", #selector(NSWindow.performClose(_:)), "w", nil))
    return fileMenu
  }

  static func viewMenu(conversationActions: ThinTalkConversationActions) -> NSMenu {
    let viewMenu = NSMenu(title: "View")
    let sidebarItem = menuItem(
      "Toggle Sidebar", #selector(ThinTalkConversationActions.toggleConversationSidebar(_:)), "s",
      conversationActions)
    sidebarItem.keyEquivalentModifierMask = [.command, .control]
    viewMenu.addItem(sidebarItem)
    viewMenu.addItem(
      menuItem("Show or Hide Astro", #selector(ThinTalkConversationActions.toggleAstro(_:)), "", conversationActions))
    viewMenu.addItem(
      menuItem("Meet Astro Again", #selector(ThinTalkConversationActions.meetAstro(_:)), "", conversationActions))
    viewMenu.addItem(.separator())
    // "=" rather than "+": AppKit matches key equivalents against the key's
    // unshifted character, which is "=" for ⌘+ as well (the canvas also answers
    // ⌘+ itself, see CanvasZoomController.isZoomInKeyAlias).
    viewMenu.addItem(
      menuItem("Zoom In", #selector(ThinTalkConversationActions.zoomIn(_:)), "=", conversationActions))
    viewMenu.addItem(
      menuItem("Zoom Out", #selector(ThinTalkConversationActions.zoomOut(_:)), "-", conversationActions))
    viewMenu.addItem(
      menuItem("Actual Size", #selector(ThinTalkConversationActions.actualSize(_:)), "0", conversationActions))
    viewMenu.addItem(.separator())
    let fullScreenItem = menuItem("Enter Full Screen", #selector(NSWindow.toggleFullScreen(_:)), "f", nil)
    fullScreenItem.keyEquivalentModifierMask = [.command, .control]
    viewMenu.addItem(fullScreenItem)
    return viewMenu
  }

  static func conversationMenu(conversationActions: ThinTalkConversationActions) -> NSMenu {
    let conversationMenu = NSMenu(title: "Conversation")
    conversationMenu.addItem(
      menuItem(
        "Stop Generating", #selector(ThinTalkConversationActions.stopGenerating(_:)), ".",
        conversationActions))
    let copyItem = menuItem(
      "Copy Last Answer", #selector(ThinTalkConversationActions.copyLastAnswer(_:)), "c", conversationActions)
    copyItem.keyEquivalentModifierMask = [.command, .shift]
    conversationMenu.addItem(copyItem)
    conversationMenu.addItem(
      menuItem(
        "Focus Message Field", #selector(ThinTalkConversationActions.focusMessageField(_:)), "l",
        conversationActions))
    conversationMenu.addItem(.separator())
    // No shortcut: deleting cannot be undone, and ⌘⌫ already deletes to the
    // start of the line while typing.
    conversationMenu.addItem(
      menuItem(
        "Delete Conversation…", #selector(ThinTalkConversationActions.deleteConversation(_:)), "",
        conversationActions))
    return conversationMenu
  }

  /// One menu item. A nil target sends the action up the responder chain.
  /// Selector names differ from the page's command names on purpose where a
  /// standard AppKit selector already exists: `toggleSidebar:` belongs to
  /// NSSplitViewController and would be answered by the wrong object.
  static func menuItem(_ title: String, _ action: Selector, _ keyEquivalent: String, _ target: AnyObject?)
    -> NSMenuItem
  {
    let targetedMenuItem = NSMenuItem(title: title, action: action, keyEquivalent: keyEquivalent)
    targetedMenuItem.target = target
    return targetedMenuItem
  }
}
