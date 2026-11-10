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

  func test_should_open_your_astro_from_its_own_link_and_not_as_chat() throws {
    let yourAstroURL = try XCTUnwrap(URL(string: "astronomical-dev://your-astro"))

    XCTAssertEqual(ApplicationLink.parse(yourAstroURL, channel: .development), .yourAstro)
    XCTAssertFalse(isThinTalkChatOpenRequest(yourAstroURL, channel: .development))
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
    let stableYourAstroURL = try XCTUnwrap(ApplicationLink.yourAstro.url(channel: .stable))

    XCTAssertNil(ApplicationLink.parse(stableYourAstroURL, channel: .development))
  }

  func test_should_refuse_unknown_destinations_and_extra_path_components() throws {
    for refusedLink in ["astronomical://settings", "astronomical://chat/extra", "astronomical://", "astronomical:///a/b"] {
      let refusedURL = try XCTUnwrap(URL(string: refusedLink))
      XCTAssertNil(ApplicationLink.parse(refusedURL, channel: .stable), refusedLink)
    }
  }

  func test_should_accept_the_path_form_and_ignore_letter_case() throws {
    XCTAssertEqual(ApplicationLink.parse(try XCTUnwrap(URL(string: "astronomical:///chat")), channel: .stable), .chat)
    XCTAssertEqual(ApplicationLink.parse(try XCTUnwrap(URL(string: "ASTRONOMICAL://CHAT")), channel: .stable), .chat)
  }
}
