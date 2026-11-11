import AppKit

/// The right-click menu for one conversation in the sidebar.
///
/// The page owns the sidebar but asks for a native menu so it looks and
/// behaves like every other macOS context menu. The menu only reports the
/// choice; renaming happens in the page, and deleting is confirmed here first
/// because a delete cannot be undone.
@MainActor
final class ConversationContextMenu: NSObject {
  private var pendingChoice: CheckedContinuation<ConversationMenuChoice, Never>?

  /// Shows the menu at the pointer over `view` and waits for the user. A view
  /// that is not in a visible window shows nothing.
  func choose(in view: NSView) async -> ConversationMenuChoice {
    guard let window = view.window, window.isVisible, pendingChoice == nil else { return .dismissed }
    let pointerInView = view.convert(window.convertPoint(fromScreen: NSEvent.mouseLocation), from: nil)
    let conversationMenu = makeMenu()
    let menuChoice = await awaitChoice(trackingMenu: { conversationMenu.popUp(positioning: nil, at: pointerInView, in: view) })
    guard menuChoice == .delete else { return menuChoice }
    return await DeleteConversationConfirmation.confirm(on: window) ? .delete : .dismissed
  }

  /// Waits for the user while `trackingMenu` runs the menu; it returns
  /// whether an item was chosen, as `NSMenu.popUp` does. Separate from
  /// `choose` so the settling rules can be tested without a real menu.
  func awaitChoice(trackingMenu: () -> Bool) async -> ConversationMenuChoice {
    await withCheckedContinuation { continuation in
      pendingChoice = continuation
      settleAfterTracking(didChooseItem: trackingMenu())
    }
  }

  private func makeMenu() -> NSMenu {
    let conversationMenu = NSMenu(title: "Conversation")
    conversationMenu.autoenablesItems = false
    conversationMenu.addItem(menuItem(title: "Rename…", action: #selector(chooseRename(_:))))
    conversationMenu.addItem(menuItem(title: "Export…", action: #selector(chooseExport(_:))))
    conversationMenu.addItem(.separator())
    conversationMenu.addItem(menuItem(title: "Delete…", action: #selector(chooseDelete(_:))))
    return conversationMenu
  }

  private func menuItem(title: String, action: Selector) -> NSMenuItem {
    let conversationMenuItem = NSMenuItem(title: title, action: action, keyEquivalent: "")
    conversationMenuItem.target = self
    return conversationMenuItem
  }

  @objc func chooseRename(_ sender: Any?) {
    finish(with: .rename)
  }

  @objc func chooseExport(_ sender: Any?) {
    finish(with: .export)
  }

  @objc func chooseDelete(_ sender: Any?) {
    finish(with: .delete)
  }

  /// Called once menu tracking ends. `popUp` reports whether an item was
  /// chosen, so a closed menu is settled as dismissed only when nothing was;
  /// a chosen item settles the wait through its own action, whichever order
  /// AppKit sends that action and returns from `popUp`. Settling on a guess
  /// (a later main-queue turn) could let "dismissed" win over the user's pick.
  func settleAfterTracking(didChooseItem: Bool) {
    if !didChooseItem {
      finish(with: .dismissed)
    }
  }

  private func finish(with menuChoice: ConversationMenuChoice) {
    pendingChoice?.resume(returning: menuChoice)
    pendingChoice = nil
  }
}
