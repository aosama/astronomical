import Foundation
import XCTest

@testable import AstronomicalMenuCore

/// The Observatory console links chat through astronomical://chat; these tests
/// pin the link contract so the web console and the packaged app cannot drift
/// apart on what opens the conversation window.
final class ChatOpenRequestTests: XCTestCase {
  func test_accepts_the_chat_link_from_the_observatory_console() {
    XCTAssertTrue(isThinTalkChatOpenRequest(URL(string: "astronomical://chat")!))
  }

  func test_accepts_the_chat_link_regardless_of_scheme_case() {
    XCTAssertTrue(isThinTalkChatOpenRequest(URL(string: "ASTRONOMICAL://chat")!))
  }

  func test_rejects_links_asking_for_something_else() {
    XCTAssertFalse(isThinTalkChatOpenRequest(URL(string: "astronomical://settings")!))
    XCTAssertFalse(isThinTalkChatOpenRequest(URL(string: "astronomical://chat/1234")!))
    XCTAssertFalse(isThinTalkChatOpenRequest(URL(string: "astronomical://")!))
  }

  func test_rejects_links_from_other_schemes() {
    XCTAssertFalse(isThinTalkChatOpenRequest(URL(string: "https://chat")!))
    XCTAssertFalse(isThinTalkChatOpenRequest(URL(string: "astronomical-copy://chat")!))
  }

  /// The matcher is only half the contract: if the application delegate stops
  /// registering the GURL Apple-event handler at launch, the Observatory link
  /// silently dead-ends and no behavioral test can catch it without a real
  /// event loop. This structural test pins the wiring the way the menu's other
  /// source-scanning contracts do.
  func test_the_application_registers_the_link_handler_at_launch() throws {
    let packageDirectoryURL = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
    let applicationSource = try String(
      contentsOf: packageDirectoryURL
        .appendingPathComponent("Sources/AstronomicalMenuCore/AstronomicalMenuApplication.swift"),
      encoding: .utf8)

    let launchCallStart = try XCTUnwrap(
      applicationSource.range(of: "registerThinTalkOpenHandler()"))
    let handlerRegistrationStart = try XCTUnwrap(
      applicationSource.range(of: "forEventClass: AEEventClass(kInternetEventClass)"))
    let matcherUsageStart = try XCTUnwrap(
      applicationSource.range(of: "isThinTalkChatOpenRequest(openedURL)"))

    XCTAssertLessThan(launchCallStart.lowerBound, handlerRegistrationStart.lowerBound)
    XCTAssertLessThan(handlerRegistrationStart.lowerBound, matcherUsageStart.lowerBound)
  }
}
