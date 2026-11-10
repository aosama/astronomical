import Foundation
import XCTest

@testable import AstronomicalShared

/// The Observatory console links chat through its channel's scheme; these tests
/// pin the link contract so the web console and the packaged app cannot drift
/// apart on what opens the conversation window, and so no channel can ever open
/// its chat window from another channel's link.
final class ChatLinkContractTests: XCTestCase {
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

  func test_should_refuse_the_stable_link_in_development() {
    XCTAssertFalse(
      isThinTalkChatOpenRequest(URL(string: "astronomical://chat")!, channel: .development))
  }

  func test_should_refuse_the_development_link_in_stable_and_the_app_store() {
    XCTAssertFalse(isThinTalkChatOpenRequest(URL(string: "astronomical-dev://chat")!, channel: .stable))
    XCTAssertFalse(
      isThinTalkChatOpenRequest(URL(string: "astronomical-dev://chat")!, channel: .appStore))
  }

  func test_should_accept_the_chat_link_regardless_of_scheme_case() {
    XCTAssertTrue(
      isThinTalkChatOpenRequest(URL(string: "ASTRONOMICAL-DEV://chat")!, channel: .development))
  }

  func test_should_refuse_links_asking_for_something_else() {
    for otherLink in ["astronomical-dev://settings", "astronomical-dev://chat/1234", "astronomical-dev://"] {
      XCTAssertFalse(isThinTalkChatOpenRequest(URL(string: otherLink)!, channel: .development), otherLink)
    }
  }

  func test_should_refuse_links_from_other_schemes() {
    XCTAssertFalse(isThinTalkChatOpenRequest(URL(string: "https://chat")!, channel: .development))
    XCTAssertFalse(
      isThinTalkChatOpenRequest(URL(string: "astronomical-copy://chat")!, channel: .development))
  }

  /// The matcher is only half the contract: if the main app's delegate stops
  /// registering the GURL Apple-event handler before launch finishes, a link
  /// from the Observatory or from the menu-bar agent silently dead-ends, and
  /// no behavioral test can catch it without a real event loop. This
  /// structural test pins the wiring the way the other source-scanning
  /// contracts do.
  func test_should_register_the_link_handler_in_the_main_app_before_launch_finishes() throws {
    let packageDirectoryURL = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
    let applicationSource = try String(
      contentsOf: packageDirectoryURL
        .appendingPathComponent("Sources/AstronomicalAppCore/AstronomicalApplication.swift"),
      encoding: .utf8)

    let willFinishLaunchingStart = try XCTUnwrap(
      applicationSource.range(of: "func applicationWillFinishLaunching"))
    let handlerRegistrationStart = try XCTUnwrap(
      applicationSource.range(of: "forEventClass: AEEventClass(kInternetEventClass)"))
    let didFinishLaunchingStart = try XCTUnwrap(
      applicationSource.range(of: "func applicationDidFinishLaunching"))
    let matcherUsageStart = try XCTUnwrap(
      applicationSource.range(of: "ApplicationLink.parse(openedURL, channel: applicationIdentity.channel)"))

    XCTAssertLessThan(willFinishLaunchingStart.lowerBound, handlerRegistrationStart.lowerBound)
    XCTAssertLessThan(handlerRegistrationStart.lowerBound, didFinishLaunchingStart.lowerBound)
    XCTAssertLessThan(didFinishLaunchingStart.lowerBound, matcherUsageStart.lowerBound)
  }

  /// The agent has no window: it must ask the main app through a link, never
  /// build a conversation window of its own.
  func test_should_open_chat_from_the_agent_through_a_link_and_link_no_chat_window() throws {
    let packageDirectoryURL = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let agentSource = try String(
      contentsOf: packageDirectoryURL
        .appendingPathComponent("Sources/AstronomicalMenuCore/AstronomicalMenuApplication.swift"),
      encoding: .utf8)

    XCTAssertTrue(agentSource.contains("openInMainApplication(.chat)"))
    XCTAssertFalse(agentSource.contains("import ThinTalk"))
    XCTAssertFalse(agentSource.contains("setActivationPolicy(.regular)"), "the agent never becomes a regular app")
  }
}
