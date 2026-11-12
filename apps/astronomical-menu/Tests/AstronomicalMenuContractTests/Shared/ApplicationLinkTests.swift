import Foundation
import XCTest

@testable import AstronomicalShared

/// The menu-bar agent asks the main app for a window by opening one of these
/// links, so the two processes must agree on every destination and a link
/// must round-trip through its own channel's scheme and no other.
final class ApplicationLinkTests: XCTestCase {
  func test_should_round_trip_every_link_through_its_own_channel() throws {
    for applicationChannel in [ApplicationChannel.stable, .development, .appStore] {
      for applicationLink in ApplicationLink.allCases {
        let linkURL = try XCTUnwrap(applicationLink.url(channel: applicationChannel))
        XCTAssertEqual(ApplicationLink.parse(linkURL, channel: applicationChannel), applicationLink)
      }
    }
  }

  func test_should_open_settings_from_its_own_link_and_not_as_chat() throws {
    let settingsURL = try XCTUnwrap(URL(string: "astronomical-dev://settings"))

    XCTAssertEqual(ApplicationLink.parse(settingsURL, channel: .development), .settings)
    XCTAssertFalse(isThinTalkChatOpenRequest(settingsURL, channel: .development))
  }

  func test_should_ask_for_an_update_check_from_its_own_link_and_not_as_a_window() throws {
    let updateCheckURL = try XCTUnwrap(URL(string: "astronomical://check-for-updates"))

    XCTAssertEqual(ApplicationLink.parse(updateCheckURL, channel: .stable), .checkForUpdates)
    XCTAssertFalse(isThinTalkChatOpenRequest(updateCheckURL, channel: .stable))
  }

  func test_should_refuse_an_update_check_link_of_another_channel() throws {
    let developmentUpdateCheckURL = try XCTUnwrap(ApplicationLink.checkForUpdates.url(channel: .development))

    XCTAssertNil(ApplicationLink.parse(developmentUpdateCheckURL, channel: .stable))
  }

  func test_should_refuse_a_link_of_another_channel() throws {
    let stableSettingsURL = try XCTUnwrap(ApplicationLink.settings.url(channel: .stable))

    XCTAssertNil(ApplicationLink.parse(stableSettingsURL, channel: .development))
  }

  func test_should_refuse_unknown_destinations_and_extra_path_components() throws {
    for refusedLink in ["astronomical://preferences", "astronomical://chat/extra", "astronomical://", "astronomical:///a/b"] {
      let refusedURL = try XCTUnwrap(URL(string: refusedLink))
      XCTAssertNil(ApplicationLink.parse(refusedURL, channel: .stable), refusedLink)
    }
  }

  func test_should_accept_the_path_form_and_ignore_letter_case() throws {
    XCTAssertEqual(ApplicationLink.parse(try XCTUnwrap(URL(string: "astronomical:///chat")), channel: .stable), .chat)
    XCTAssertEqual(ApplicationLink.parse(try XCTUnwrap(URL(string: "ASTRONOMICAL://CHAT")), channel: .stable), .chat)
  }

  func test_should_carry_a_draft_through_a_link_and_back_whatever_it_contains() throws {
    for draftText in ["What is 15% of 80?", "a&b=c #tag ?x", "Explique la photosynthèse 🌱", "line one\nline two"] {
      let draftURL = try XCTUnwrap(DraftLink.url(draftText: draftText, channel: .development))
      XCTAssertEqual(draftURL.scheme, "astronomical-dev")
      XCTAssertEqual(DraftLink.draftText(from: draftURL, channel: .development), draftText)
    }
  }

  /// A web page can open a link. Only the Development app may answer this
  /// one, so the app people use cannot have text put in its message field.
  func test_should_answer_a_draft_link_only_in_the_development_channel() throws {
    let stableURL = try XCTUnwrap(DraftLink.url(draftText: "hello", channel: .stable))
    let developmentURL = try XCTUnwrap(DraftLink.url(draftText: "hello", channel: .development))

    XCTAssertNil(DraftLink.draftText(from: stableURL, channel: .stable))
    XCTAssertNil(DraftLink.draftText(from: stableURL, channel: .appStore))
    XCTAssertNil(DraftLink.draftText(from: developmentURL, channel: .stable), "another channel's link")
    XCTAssertNil(DraftLink.draftText(from: stableURL, channel: .development), "another channel's scheme")
  }

  func test_should_refuse_a_draft_link_with_no_text_or_too_much() throws {
    let longestDraft = String(repeating: "a", count: DraftLink.maximumDraftCharacters)
    XCTAssertEqual(
      DraftLink.draftText(from: try XCTUnwrap(DraftLink.url(draftText: longestDraft, channel: .development)), channel: .development),
      longestDraft)
    let refusedLinks = [
      DraftLink.url(draftText: longestDraft + "a", channel: .development),
      DraftLink.url(draftText: "   ", channel: .development),
      URL(string: "astronomical-dev://draft"),
      URL(string: "astronomical-dev://draft?other=hello"),
      URL(string: "astronomical-dev://chat?text=hello"),
    ]
    for refusedLink in refusedLinks {
      XCTAssertNil(DraftLink.draftText(from: try XCTUnwrap(refusedLink), channel: .development), "\(String(describing: refusedLink))")
    }
  }

  func test_should_not_mistake_a_draft_link_for_a_window_link() throws {
    let draftURL = try XCTUnwrap(DraftLink.url(draftText: "hello", channel: .development))
    XCTAssertNil(ApplicationLink.parse(draftURL, channel: .development))
  }

  func test_should_carry_an_export_folder_through_a_link_and_back() throws {
    for folderPath in ["/work/exports", "/work/my exports & drafts", "/travail/exposés"] {
      let folderLink = try XCTUnwrap(ExportFolderLink.url(folderPath: folderPath, channel: .development))
      XCTAssertEqual(ExportFolderLink.folderURL(from: folderLink, channel: .development)?.path, folderPath)
    }
  }

  /// A web page can open a link. Only the Development app may answer this
  /// one, so in the app people use a file never leaves without a save panel.
  func test_should_answer_an_export_folder_link_only_in_the_development_channel() throws {
    let stableLink = try XCTUnwrap(ExportFolderLink.url(folderPath: "/work/exports", channel: .stable))
    let developmentLink = try XCTUnwrap(ExportFolderLink.url(folderPath: "/work/exports", channel: .development))

    XCTAssertNil(ExportFolderLink.folderURL(from: stableLink, channel: .stable))
    XCTAssertNil(ExportFolderLink.folderURL(from: stableLink, channel: .appStore))
    XCTAssertNil(ExportFolderLink.folderURL(from: developmentLink, channel: .stable), "another channel's link")
    XCTAssertNil(ExportFolderLink.folderURL(from: stableLink, channel: .development), "another channel's scheme")
  }

  func test_should_refuse_an_export_folder_link_without_an_absolute_path() throws {
    let refusedLinks = [
      ExportFolderLink.url(folderPath: "exports", channel: .development),
      ExportFolderLink.url(folderPath: "/", channel: .development),
      ExportFolderLink.url(folderPath: "", channel: .development),
      URL(string: "astronomical-dev://export-folder"),
      URL(string: "astronomical-dev://export-folder?other=/work"),
      URL(string: "astronomical-dev://draft?path=/work"),
    ]
    for refusedLink in refusedLinks {
      XCTAssertNil(ExportFolderLink.folderURL(from: try XCTUnwrap(refusedLink), channel: .development), "\(String(describing: refusedLink))")
    }
  }

  func test_should_not_mistake_an_export_folder_link_for_a_window_or_draft_link() throws {
    let folderLink = try XCTUnwrap(ExportFolderLink.url(folderPath: "/work/exports", channel: .development))
    XCTAssertNil(ApplicationLink.parse(folderLink, channel: .development))
    XCTAssertNil(DraftLink.draftText(from: folderLink, channel: .development))
  }

  // MARK: - The chat link the Observatory console opens
  //
  // The web console links chat through its channel's scheme; these pin that
  // link so the console and the packaged app cannot drift apart, and so no
  // channel can ever open its chat window from another channel's link.

  func test_should_give_development_a_scheme_no_other_channel_uses() {
    XCTAssertEqual(ApplicationChannel.development.chatURLScheme, "astronomical-dev")
    XCTAssertEqual(ApplicationChannel.stable.chatURLScheme, "astronomical")
    XCTAssertEqual(ApplicationChannel.appStore.chatURLScheme, "astronomical")
  }

  func test_should_accept_each_channels_own_chat_link() {
    XCTAssertTrue(
      isThinTalkChatOpenRequest(URL(string: "astronomical-dev://chat")!, channel: .development))
    XCTAssertTrue(isThinTalkChatOpenRequest(URL(string: "astronomical://chat")!, channel: .stable))
    XCTAssertTrue(isThinTalkChatOpenRequest(URL(string: "astronomical://chat")!, channel: .appStore))
  }

  func test_should_refuse_the_development_chat_link_in_stable_and_the_app_store() {
    XCTAssertFalse(isThinTalkChatOpenRequest(URL(string: "astronomical-dev://chat")!, channel: .stable))
    XCTAssertFalse(
      isThinTalkChatOpenRequest(URL(string: "astronomical-dev://chat")!, channel: .appStore))
  }

  func test_should_not_treat_another_destination_as_a_chat_request() {
    for otherLink in ["astronomical-dev://settings", "astronomical-dev://chat/1234", "astronomical-dev://"] {
      XCTAssertFalse(isThinTalkChatOpenRequest(URL(string: otherLink)!, channel: .development), otherLink)
    }
  }

  func test_should_refuse_chat_links_from_other_schemes() {
    XCTAssertFalse(isThinTalkChatOpenRequest(URL(string: "https://chat")!, channel: .development))
    XCTAssertFalse(
      isThinTalkChatOpenRequest(URL(string: "astronomical-copy://chat")!, channel: .development))
  }
}
