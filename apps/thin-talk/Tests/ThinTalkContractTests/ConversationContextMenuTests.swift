import AppKit
import XCTest

@testable import ThinTalkCanvas

/// The right-click menu's paths that must end without showing anything, and
/// the delete question it shares with the menu bar. Picking from the menu and
/// answering the sheet need a person, so they are exercised live.
@MainActor
final class ConversationContextMenuTests: XCTestCase {
  func test_should_show_nothing_for_a_view_outside_a_visible_window() async {
    let detachedView = NSView(frame: NSRect(x: 0, y: 0, width: 100, height: 100))
    let menuChoice = await ConversationContextMenu().choose(in: detachedView)
    XCTAssertEqual(menuChoice, .dismissed)
  }

  func test_should_report_an_item_chosen_during_tracking() async {
    let contextMenu = ConversationContextMenu()
    let menuChoice = await contextMenu.awaitChoice(trackingMenu: {
      contextMenu.chooseRename(nil)
      return true
    })
    XCTAssertEqual(menuChoice, .rename)
  }

  func test_should_report_an_item_whose_action_arrives_after_tracking_ends() async {
    let contextMenu = ConversationContextMenu()
    let menuChoice = await contextMenu.awaitChoice(trackingMenu: {
      // The action is delivered after popUp has already returned "chosen";
      // the wait must hold for it rather than settle as dismissed.
      DispatchQueue.main.async { contextMenu.chooseDelete(nil) }
      return true
    })
    XCTAssertEqual(menuChoice, .delete)
  }

  func test_should_report_dismissed_when_the_menu_closes_without_a_choice() async {
    let contextMenu = ConversationContextMenu()
    let menuChoice = await contextMenu.awaitChoice(trackingMenu: { false })
    XCTAssertEqual(menuChoice, .dismissed)
  }

  func test_should_send_each_choice_to_the_page_by_the_name_it_expects() {
    XCTAssertEqual(ConversationMenuChoice.rename.rawValue, "rename")
    XCTAssertEqual(ConversationMenuChoice.export.rawValue, "export")
    XCTAssertEqual(ConversationMenuChoice.delete.rawValue, "delete")
    XCTAssertEqual(ConversationMenuChoice.dismissed.rawValue, "dismissed")
  }

  func test_should_report_export_when_export_is_chosen() async {
    let contextMenu = ConversationContextMenu()
    let menuChoice = await contextMenu.awaitChoice(trackingMenu: {
      contextMenu.chooseExport(nil)
      return true
    })
    XCTAssertEqual(menuChoice, .export)
  }

  func test_should_ask_before_deleting_with_delete_as_the_destructive_first_button() {
    let confirmation = DeleteConversationConfirmation.makeAlert()
    XCTAssertEqual(confirmation.buttons.map(\.title), ["Delete", "Cancel"])
    XCTAssertTrue(confirmation.buttons[0].hasDestructiveAction)
    XCTAssertEqual(confirmation.alertStyle, .warning)
  }
}
