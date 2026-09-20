import XCTest

@testable import AstronomicalMenuCore

/// Contracts for how the menu application maps its channel identity onto the
/// conversation surface's wire identity. The chat client validates the
/// supervisor's channel and state directory during its handshake, so every
/// mapping below is what keeps the Chat menu item talking to the instance the
/// menu app itself started.
final class ChatLaunchIdentityTests: XCTestCase {
  private func menuIdentity(
    channel: ApplicationChannel
  ) -> ApplicationIdentity {
    ApplicationIdentity(
      channel: channel,
      supervisorPort: channel.defaultSupervisorPort,
      stateDirectoryName: channel.stateDirectoryName,
      version: "9.9",
      buildNumber: "999",
      commit: "test-commit",
      isDirty: false)
  }

  func test_development_channel_maps_to_the_development_conversation_identity() {
    let mapped = ChatLaunchIdentity.chatIdentity(
      from: menuIdentity(channel: .development))

    XCTAssertEqual(mapped.channel, .development)
    XCTAssertEqual(mapped.supervisorPort, 6733)
    XCTAssertEqual(mapped.stateDirectoryName, ".astronomical-dev")
    XCTAssertEqual(mapped.version, "9.9")
    XCTAssertEqual(mapped.commit, "test-commit")
  }

  func test_stable_channel_maps_to_the_stable_conversation_identity() {
    let mapped = ChatLaunchIdentity.chatIdentity(from: menuIdentity(channel: .stable))

    XCTAssertEqual(mapped.channel, .stable)
    XCTAssertEqual(mapped.supervisorPort, 6732)
    XCTAssertEqual(mapped.stateDirectoryName, ".astronomical")
    XCTAssertEqual(mapped.buildNumber, "999")
  }

  func test_app_store_channel_presents_the_stable_conversation_identity() {
    // The App Store channel has no state directory name of its own; the chat
    // surface falls back to the Stable conversation channel's directory, which
    // is the identity the bundled Stable daemon advertises.
    let mapped = ChatLaunchIdentity.chatIdentity(from: menuIdentity(channel: .appStore))

    XCTAssertEqual(mapped.channel, .stable)
    XCTAssertEqual(mapped.supervisorPort, 6732)
    XCTAssertEqual(mapped.stateDirectoryName, ".astronomical")
  }
}
