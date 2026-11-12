import AppKit
import ThinTalkCanvas
import XCTest

@testable import ThinTalkUI

/// The conversation window's menu bar. The Edit menu is load-bearing: AppKit
/// routes ⌘C/⌘V/⌘X/⌘A/⌘Z to the message field through it, so these contracts
/// pin its actions and keys, not just the menu titles.
@MainActor
final class ThinTalkMainMenuTests: XCTestCase {
  private var menuBarBeforeTest: NSMenu?

  // The async overrides are the forms XCTest runs on the main actor in every
  // toolchain this repository builds with; the synchronous ones inherit
  // XCTestCase's nonisolated declaration under older Xcode releases and
  // cannot touch NSApp there.
  @MainActor
  override func setUp() async throws {
    try await super.setUp()
    _ = NSApplication.shared
    menuBarBeforeTest = NSApp.mainMenu
  }

  // Installing a menu bar changes process-wide state; later tests must not
  // inherit it.
  @MainActor
  override func tearDown() async throws {
    NSApp.mainMenu = menuBarBeforeTest
    try await super.tearDown()
  }

  private func conversationActions() -> ThinTalkConversationActions {
    ThinTalkConversationActions(commandController: CanvasCommandController(), zoomController: CanvasZoomController())
  }

  private func menuBar(_ hostActions: ThinTalkHostMenuActions) -> (NSMenu, ThinTalkConversationActions) {
    _ = NSApplication.shared
    let actions = conversationActions()
    return (ThinTalkMainMenu.make(hostActions: hostActions, conversationActions: actions), actions)
  }

  private func submenu(_ menuBar: NSMenu, _ title: String) throws -> NSMenu {
    try XCTUnwrap(menuBar.item(withTitle: title)?.submenu, "the menu bar must have a \(title) menu")
  }

  func test_should_lay_out_the_menu_bar_in_the_order_mac_chat_apps_use() {
    let (mainMenu, _) = menuBar(ThinTalkHostMenuActions(applicationName: "Astronomical"))
    XCTAssertEqual(
      mainMenu.items.map(\.title),
      ["Astronomical", "File", "Edit", "View", "Conversation", "Window", "Help"])
  }

  func test_should_route_the_edit_keys_up_the_responder_chain_to_the_message_field() throws {
    let (mainMenu, _) = menuBar(ThinTalkHostMenuActions(applicationName: "Astronomical"))
    let editMenu = try submenu(mainMenu, "Edit")
    let expectedEditItems: [(String, String, String, NSEvent.ModifierFlags)] = [
      ("Undo", "undo:", "z", [.command]), ("Redo", "redo:", "z", [.command, .shift]),
      ("Cut", "cut:", "x", [.command]), ("Copy", "copy:", "c", [.command]),
      ("Paste", "paste:", "v", [.command]),
      ("Paste and Match Style", "pasteAsPlainText:", "v", [.command, .option, .shift]),
      ("Select All", "selectAll:", "a", [.command]),
    ]
    for (title, actionName, keyEquivalent, modifierMask) in expectedEditItems {
      let editItem = try XCTUnwrap(editMenu.item(withTitle: title), title)
      XCTAssertEqual(editItem.action, Selector((actionName)), title)
      XCTAssertEqual(editItem.keyEquivalent, keyEquivalent, title)
      XCTAssertEqual(editItem.keyEquivalentModifierMask, modifierMask, title)
      XCTAssertNil(editItem.target, "\(title) must reach the focused view, not a fixed target")
    }
    XCTAssertNil(editMenu.item(withTitle: "Delete"), "the web view never enables Delete")
  }

  func test_should_bind_the_view_menu_to_the_sidebar_zoom_and_full_screen_keys() throws {
    let (mainMenu, actions) = menuBar(ThinTalkHostMenuActions(applicationName: "Astronomical"))
    let viewMenu = try submenu(mainMenu, "View")
    let expectedViewItems: [(String, String, NSEvent.ModifierFlags, Bool)] = [
      ("Toggle Sidebar", "s", [.command, .control], true),
      ("Show or Hide Astro", "", [.command], true),
      ("Zoom In", "=", [.command], true), ("Zoom Out", "-", [.command], true),
      ("Actual Size", "0", [.command], true),
      ("Enter Full Screen", "f", [.command, .control], false),
    ]
    for (title, keyEquivalent, modifierMask, targetsConversation) in expectedViewItems {
      let viewItem = try XCTUnwrap(viewMenu.item(withTitle: title), title)
      XCTAssertEqual(viewItem.keyEquivalent, keyEquivalent, title)
      XCTAssertEqual(viewItem.keyEquivalentModifierMask, modifierMask, title)
      XCTAssertEqual(viewItem.target === actions, targetsConversation, title)
    }
    XCTAssertNil(viewMenu.item(withTitle: "Enter Full Screen")?.target, "full screen goes to the key window")
  }

  func test_should_open_the_settings_page_with_command_comma() throws {
    let (mainMenu, actions) = menuBar(ThinTalkHostMenuActions(applicationName: "Astronomical"))
    let applicationMenu = try submenu(mainMenu, "Astronomical")
    let settingsItem = try XCTUnwrap(applicationMenu.item(withTitle: "Settings…"))
    XCTAssertEqual(settingsItem.action, #selector(ThinTalkConversationActions.openSettings(_:)))
    XCTAssertEqual(settingsItem.keyEquivalent, ",")
    XCTAssertEqual(settingsItem.keyEquivalentModifierMask, [.command])
    XCTAssertTrue(settingsItem.target === actions, "Settings opens Settings in the conversation page")
    XCTAssertNil(
      try submenu(mainMenu, "View").item(withTitle: "Meet Astro Again"),
      "the replay lives on Settings, not in the View menu")
  }

  func test_should_quit_through_the_normal_terminate_path() throws {
    let (mainMenu, _) = menuBar(ThinTalkHostMenuActions(applicationName: "Astronomical"))
    let quitItem = try XCTUnwrap(try submenu(mainMenu, "Astronomical").item(withTitle: "Quit Astronomical"))
    XCTAssertEqual(quitItem.action, #selector(NSApplication.terminate(_:)))
    XCTAssertEqual(quitItem.keyEquivalent, "q")
    XCTAssertNil(quitItem.target, "quit must reach the application delegate's terminate handling")
  }

  func test_should_disable_conversation_items_while_their_window_is_closed() throws {
    let (mainMenu, actions) = menuBar(ThinTalkHostMenuActions(applicationName: "Astronomical"))
    let newChatItem = try XCTUnwrap(try submenu(mainMenu, "File").item(withTitle: "New Chat"))
    XCTAssertFalse(actions.validateMenuItem(newChatItem), "no window at all")
    let hostWindow = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 200, height: 100), styleMask: [.titled],
      backing: .buffered, defer: false)
    hostWindow.isReleasedWhenClosed = false
    actions.hostWindow = hostWindow
    XCTAssertFalse(actions.validateMenuItem(newChatItem), "window not on screen")
    hostWindow.orderFront(nil)
    XCTAssertTrue(actions.validateMenuItem(newChatItem), "window on screen")
    hostWindow.close()
    XCTAssertFalse(actions.validateMenuItem(newChatItem), "window closed again")
  }

  func test_should_send_conversation_commands_to_the_conversation_actions() throws {
    let (mainMenu, actions) = menuBar(ThinTalkHostMenuActions(applicationName: "Astronomical"))
    let newChatItem = try XCTUnwrap(try submenu(mainMenu, "File").item(withTitle: "New Chat"))
    XCTAssertTrue(newChatItem.target === actions)
    XCTAssertEqual(newChatItem.keyEquivalent, "n")
    let conversationMenu = try submenu(mainMenu, "Conversation")
    XCTAssertEqual(
      conversationMenu.items.filter { !$0.isSeparatorItem }.map(\.title),
      ["Stop Generating", "Copy Last Answer", "Focus Message Field", "Delete Conversation…"])
    for conversationItem in conversationMenu.items where !conversationItem.isSeparatorItem {
      XCTAssertTrue(conversationItem.target === actions, conversationItem.title)
    }
  }

  func test_should_export_from_the_file_menu_with_shift_command_e() throws {
    let (mainMenu, actions) = menuBar(ThinTalkHostMenuActions(applicationName: "Astronomical"))
    let fileMenu = try submenu(mainMenu, "File")
    let exportItem = try XCTUnwrap(fileMenu.item(withTitle: "Export…"))
    XCTAssertTrue(exportItem.target === actions)
    XCTAssertEqual(exportItem.action, #selector(ThinTalkConversationActions.exportConversation(_:)))
    XCTAssertEqual(exportItem.keyEquivalent, "e")
    XCTAssertEqual(exportItem.keyEquivalentModifierMask, [.command, .shift], "⌘E alone is Use Selection for Find")
    XCTAssertEqual(
      fileMenu.items.filter { !$0.isSeparatorItem }.map(\.title), ["New Chat", "Export…", "Close Window"])
  }

  func test_should_leave_the_irreversible_delete_without_a_keyboard_shortcut() throws {
    let (mainMenu, _) = menuBar(ThinTalkHostMenuActions(applicationName: "Astronomical"))
    let deleteItem = try XCTUnwrap(
      try submenu(mainMenu, "Conversation").item(withTitle: "Delete Conversation…"))
    XCTAssertEqual(deleteItem.keyEquivalent, "")
  }

  func test_should_offer_updates_observatory_and_help_only_when_the_host_supplies_them() throws {
    let (plainMenu, _) = menuBar(ThinTalkHostMenuActions(applicationName: "Thin Talk"))
    let plainApplicationMenu = try submenu(plainMenu, "Thin Talk")
    XCTAssertNil(plainApplicationMenu.item(withTitle: "Check for Updates…"))
    XCTAssertNil(plainApplicationMenu.item(withTitle: "Open Observatory"))
    XCTAssertEqual(try submenu(plainMenu, "Help").items.count, 0)
    XCTAssertNotNil(plainApplicationMenu.item(withTitle: "Quit Thin Talk"))

    var performedHostActions: [String] = []
    let (hostMenu, _) = menuBar(
      ThinTalkHostMenuActions(
        applicationName: "Astronomical",
        checkForUpdates: { performedHostActions.append("updates") },
        openObservatory: { performedHostActions.append("observatory") },
        helpURL: URL(string: "https://example.invalid/help")))
    let hostApplicationMenu = try submenu(hostMenu, "Astronomical")
    for (title, expectedAction) in [("Check for Updates…", "updates"), ("Open Observatory", "observatory")] {
      let hostItem = try XCTUnwrap(hostApplicationMenu.item(withTitle: title), title)
      let hostActionHandler = try XCTUnwrap(hostItem.target as? MenuActionHandler, title)
      hostActionHandler.performAction(hostItem)
      XCTAssertEqual(performedHostActions.last, expectedAction)
    }
    XCTAssertNotNil(try submenu(hostMenu, "Help").item(withTitle: "Astronomical Help"))
  }

  func test_should_hand_appkit_the_window_and_help_menus_when_installed() throws {
    let (mainMenu, _) = menuBar(ThinTalkHostMenuActions(applicationName: "Astronomical"))
    ThinTalkMainMenu.install(mainMenu, in: NSApp)
    XCTAssertTrue(NSApp.mainMenu === mainMenu)
    XCTAssertTrue(NSApp.windowsMenu === mainMenu.item(withTitle: "Window")?.submenu)
    XCTAssertTrue(NSApp.helpMenu === mainMenu.item(withTitle: "Help")?.submenu)
    XCTAssertTrue(
      NSApp.servicesMenu === mainMenu.item(withTitle: "Astronomical")?.submenu?.item(withTitle: "Services")?.submenu)
  }

  /// A host's own settings sit directly under Check for Updates, in the order
  /// the host gave them, so the app menu reads updates first, then settings.
  func test_should_place_the_hosts_settings_items_under_check_for_updates() throws {
    let firstSettingsItem = NSMenuItem(title: "Check for Updates Automatically", action: nil, keyEquivalent: "")
    let secondSettingsItem = NSMenuItem(title: "Launch at Login", action: nil, keyEquivalent: "")
    let (mainMenu, _) = menuBar(
      ThinTalkHostMenuActions(
        applicationName: "Host", checkForUpdates: {},
        hostSettingsMenuItems: [firstSettingsItem, secondSettingsItem]))
    let applicationMenuTitles = try XCTUnwrap(mainMenu.items.first?.submenu).items.map(\.title)
    let updatesIndex = try XCTUnwrap(applicationMenuTitles.firstIndex(of: "Check for Updates…"))

    XCTAssertEqual(
      Array(applicationMenuTitles[(updatesIndex + 1)...(updatesIndex + 2)]),
      ["Check for Updates Automatically", "Launch at Login"])
  }

  func test_should_add_no_settings_items_for_a_host_that_supplies_none() throws {
    let (mainMenu, _) = menuBar(ThinTalkHostMenuActions(applicationName: "Host"))
    let applicationMenuTitles = try XCTUnwrap(mainMenu.items.first?.submenu).items.map(\.title)

    XCTAssertEqual(Array(applicationMenuTitles.prefix(2)), ["About Host", ""], "About, then the separator")
  }

  /// A host whose Quit leaves an engine serving offers the full stop beside
  /// it. The item has no shortcut: ⌥⌘Q is AppKit's own Quit and Keep Windows.
  func test_should_offer_quit_and_stop_engine_last_without_a_shortcut_when_the_host_can() throws {
    var stopRequestCount = 0
    let (mainMenu, _) = menuBar(
      ThinTalkHostMenuActions(applicationName: "Host", quitEverything: { stopRequestCount += 1 }))
    let applicationMenu = try XCTUnwrap(mainMenu.items.first?.submenu)
    let quitEverythingItem = try XCTUnwrap(applicationMenu.items.last)

    XCTAssertEqual(quitEverythingItem.title, "Quit and Stop Engine")
    XCTAssertEqual(quitEverythingItem.keyEquivalent, "")
    XCTAssertEqual(applicationMenu.items[applicationMenu.items.count - 2].title, "Quit Host")
    _ = quitEverythingItem.target?.perform(quitEverythingItem.action, with: nil)
    XCTAssertEqual(stopRequestCount, 1)
  }

  func test_should_end_the_app_menu_at_quit_for_a_host_with_nothing_else_to_stop() throws {
    let (mainMenu, _) = menuBar(ThinTalkHostMenuActions(applicationName: "Host"))
    let applicationMenu = try XCTUnwrap(mainMenu.items.first?.submenu)

    XCTAssertEqual(applicationMenu.items.last?.title, "Quit Host")
    XCTAssertNil(applicationMenu.item(withTitle: "Quit and Stop Engine"))
  }
}
