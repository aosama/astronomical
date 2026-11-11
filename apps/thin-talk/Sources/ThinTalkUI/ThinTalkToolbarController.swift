import AppKit

extension NSToolbarItem.Identifier {
  static let thinTalkToggleSidebar = NSToolbarItem.Identifier("ThinTalkToggleSidebar")
  static let thinTalkNewChat = NSToolbarItem.Identifier("ThinTalkNewChat")
  static let thinTalkExport = NSToolbarItem.Identifier("ThinTalkExport")
}

/// The conversation window's toolbar: a sidebar toggle and a New Chat button
/// beside the window controls, the layout Apple's own chat apps use, and an
/// Export button at the far end, where Mac apps keep Share.
///
/// The toolbar is also what gives the window its frame: a unified toolbar turns
/// the title bar into one strip with the content and gives the window the larger
/// corner radius macOS reserves for toolbar windows.
@MainActor
public final class ThinTalkToolbarController: NSObject, NSToolbarDelegate {
  public static let toolbarIdentifier = NSToolbar.Identifier("ThinTalkConversationToolbar")
  private static let itemIdentifiers: [NSToolbarItem.Identifier] = [
    .thinTalkToggleSidebar, .thinTalkNewChat, .flexibleSpace, .thinTalkExport,
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
    case .thinTalkExport:
      return toolbarButton(
        itemIdentifier, label: "Export", symbolName: "square.and.arrow.up",
        action: #selector(ThinTalkConversationActions.exportConversation(_:)))
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
