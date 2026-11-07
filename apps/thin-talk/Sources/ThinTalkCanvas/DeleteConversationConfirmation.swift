import AppKit

/// The one question asked before a conversation is deleted, shared by the menu
/// bar command and the sidebar's right-click menu so both say the same thing.
/// Deleting cannot be undone, so it always asks, as a sheet on the
/// conversation window rather than a free-floating alert.
@MainActor
public enum DeleteConversationConfirmation {
  public static func makeAlert() -> NSAlert {
    let confirmation = NSAlert()
    confirmation.messageText = "Delete this conversation?"
    confirmation.informativeText = "It is removed from this Mac and cannot be restored."
    confirmation.alertStyle = .warning
    confirmation.addButton(withTitle: "Delete").hasDestructiveAction = true
    confirmation.addButton(withTitle: "Cancel")
    return confirmation
  }

  /// True only when the user pressed Delete on the sheet.
  public static func confirm(on window: NSWindow) async -> Bool {
    let response = await makeAlert().beginSheetModal(for: window)
    return response == .alertFirstButtonReturn
  }
}
