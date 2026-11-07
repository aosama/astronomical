import AppKit
import ThinTalkCanvas

/// The conversation commands the menu bar and the toolbar share.
///
/// Menu items and toolbar buttons are target-action, so one object answers both
/// and the two can never drift apart. Each action forwards to the canvas; the
/// page decides whether there is anything to act on (no answer to copy, no
/// conversation to delete), so an action on an empty conversation does nothing.
/// The menu items are disabled while the window is closed, because the menu bar
/// can outlive it.
@MainActor
public final class ThinTalkConversationActions: NSObject, NSMenuItemValidation {
  private let commandController: CanvasCommandController
  private let zoomController: CanvasZoomController
  /// The window these actions serve: confirmation sheets attach to it, and the
  /// menu items are enabled only while it is on screen. Weak because
  /// `ThinTalkConversationWindow` owns both the window and these actions.
  weak var hostWindow: NSWindow?

  public init(commandController: CanvasCommandController, zoomController: CanvasZoomController) {
    self.commandController = commandController
    self.zoomController = zoomController
    super.init()
  }

  public func validateMenuItem(_ menuItem: NSMenuItem) -> Bool {
    hostWindow?.isVisible == true
  }

  @objc public func newChat(_ sender: Any?) {
    commandController.run(.newChat)
  }

  @objc public func toggleConversationSidebar(_ sender: Any?) {
    commandController.run(.toggleSidebar)
  }

  @objc public func stopGenerating(_ sender: Any?) {
    commandController.run(.stopGenerating)
  }

  @objc public func copyLastAnswer(_ sender: Any?) {
    commandController.run(.copyLastAnswer)
  }

  @objc public func focusMessageField(_ sender: Any?) {
    commandController.run(.focusComposer)
  }

  @objc public func zoomIn(_ sender: Any?) {
    zoomController.zoomIn()
  }

  @objc public func zoomOut(_ sender: Any?) {
    zoomController.zoomOut()
  }

  @objc public func actualSize(_ sender: Any?) {
    zoomController.resetZoom()
  }

  /// Deleting is the one command that cannot be undone, so it asks first; the
  /// question is shared with the sidebar's right-click menu.
  @objc public func deleteConversation(_ sender: Any?) {
    let confirmation = DeleteConversationConfirmation.makeAlert()
    let deleteAfterConfirmation: (NSApplication.ModalResponse) -> Void = { [weak self] response in
      guard response == .alertFirstButtonReturn else { return }
      self?.commandController.run(.deleteConversation)
    }
    if let hostWindow {
      confirmation.beginSheetModal(for: hostWindow, completionHandler: deleteAfterConfirmation)
    } else {
      deleteAfterConfirmation(confirmation.runModal())
    }
  }
}
