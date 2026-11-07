import AppKit

extension NSToolbarItem.Identifier {
  static let thinTalkToggleSidebar = NSToolbarItem.Identifier("ThinTalkToggleSidebar")
  static let thinTalkNewChat = NSToolbarItem.Identifier("ThinTalkNewChat")
}

/// The conversation window's toolbar: a sidebar toggle and a New Chat button
/// beside the window controls, the layout Apple's own chat apps use.
///
/// The toolbar is also what gives the window its frame: a unified toolbar turns
/// the title bar into one strip with the content and gives the window the larger
/// corner radius macOS reserves for toolbar windows.
@MainActor
public final class ThinTalkToolbarController: NSObject, NSToolbarDelegate {
  public static let toolbarIdentifier = NSToolbar.Identifier("ThinTalkConversationToolbar")
  private static let itemIdentifiers: [NSToolbarItem.Identifier] = [
    .thinTalkToggleSidebar, .thinTalkNewChat,
  ]

  private let conversationActions: ThinTalkConversationActions

  public init(conversationActions: ThinTalkConversationActions) {
    self.conversationActions = conversationActions
    super.init()
  }

  public func makeToolbar() -> NSToolbar {
    let toolbar = NSToolbar(identifier: Self.toolbarIdentifier)
    toolbar.delegate = self
    toolbar.displayMode = .iconOnly
    toolbar.allowsUserCustomization = false
    return toolbar
  }

  public func toolbarDefaultItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] {
    Self.itemIdentifiers
  }

  public func toolbarAllowedItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] {
    Self.itemIdentifiers
  }

  public func toolbar(
    _ toolbar: NSToolbar,
    itemForItemIdentifier itemIdentifier: NSToolbarItem.Identifier,
    willBeInsertedIntoToolbar flag: Bool
  ) -> NSToolbarItem? {
    switch itemIdentifier {
    case .thinTalkToggleSidebar:
      return toolbarButton(
        itemIdentifier, label: "Sidebar", symbolName: "sidebar.left",
        action: #selector(ThinTalkConversationActions.toggleConversationSidebar(_:)))
    case .thinTalkNewChat:
      return toolbarButton(
        itemIdentifier, label: "New Chat", symbolName: "square.and.pencil",
        action: #selector(ThinTalkConversationActions.newChat(_:)))
    default:
      return nil
    }
  }

  private func toolbarButton(
    _ itemIdentifier: NSToolbarItem.Identifier, label: String, symbolName: String, action: Selector
  ) -> NSToolbarItem {
    let toolbarItem = NSToolbarItem(itemIdentifier: itemIdentifier)
    toolbarItem.label = label
    toolbarItem.toolTip = label
    toolbarItem.image = NSImage(systemSymbolName: symbolName, accessibilityDescription: label)
    toolbarItem.isBordered = true
    toolbarItem.target = conversationActions
    toolbarItem.action = action
    return toolbarItem
  }
}
