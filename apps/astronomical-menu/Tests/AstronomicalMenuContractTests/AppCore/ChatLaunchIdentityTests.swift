import XCTest

@testable import AstronomicalAppCore

/// Contracts for how the menu application maps its channel identity onto the
/// conversation surface's wire identity. The chat client validates the
/// daemon's channel and state directory during its handshake, so every
/// mapping below is what keeps the Chat menu item talking to the instance the
/// menu app itself started.
final class ChatLaunchIdentityTests: XCTestCase {
  private func menuIdentity(
    channel: ApplicationChannel
  ) -> ApplicationIdentity {
    ApplicationIdentity(
      channel: channel,
      daemonPort: channel.defaultDaemonPort,
      stateDirectoryName: channel.stateDirectoryName,
      version: "9.9",
      buildNumber: "999",
      commit: "test-commit",
      isDirty: false)
  }

  func test_should_map_the_development_channel_to_the_development_conversation_identity() {
    let mapped = ChatLaunchIdentity.chatIdentity(
      from: menuIdentity(channel: .development))

    XCTAssertEqual(mapped.channel, .development)
    XCTAssertEqual(mapped.daemonPort, 6733)
    XCTAssertEqual(mapped.stateDirectoryName, ".astronomical-dev")
    XCTAssertEqual(mapped.version, "9.9")
    XCTAssertEqual(mapped.commit, "test-commit")
  }

  /// A Development identity must never reach Stable's directory, even when the
  /// bundle names no directory at all.
  func test_should_keep_development_on_its_own_directory_without_a_bundled_name() {
    let unnamedDevelopmentIdentity = ApplicationIdentity(
      channel: .development,
      daemonPort: ApplicationChannel.development.defaultDaemonPort,
      stateDirectoryName: nil,
      version: "9.9",
      buildNumber: "999",
      commit: "test-commit",
      isDirty: false)

    let mapped = ChatLaunchIdentity.chatIdentity(from: unnamedDevelopmentIdentity)

    XCTAssertEqual(mapped.channel, .development)
    XCTAssertEqual(mapped.daemonPort, 6733)
    XCTAssertEqual(mapped.stateDirectoryName, ".astronomical-dev")
  }

  func test_should_map_the_stable_channel_to_the_stable_conversation_identity() {
    let mapped = ChatLaunchIdentity.chatIdentity(from: menuIdentity(channel: .stable))

    XCTAssertEqual(mapped.channel, .stable)
    XCTAssertEqual(mapped.daemonPort, 6732)
    XCTAssertEqual(mapped.stateDirectoryName, ".astronomical")
    XCTAssertEqual(mapped.buildNumber, "999")
  }

  func test_should_present_the_app_store_channel_as_the_stable_conversation_identity() {
    // The App Store channel has no state directory name of its own; the chat
    // surface falls back to the Stable conversation channel's name for its
    // own session files. What the engine advertises is checked separately,
    // through the expected state directory label.
    let mapped = ChatLaunchIdentity.chatIdentity(from: menuIdentity(channel: .appStore))

    XCTAssertEqual(mapped.channel, .stable)
    XCTAssertEqual(mapped.daemonPort, 6732)
    XCTAssertEqual(mapped.stateDirectoryName, ".astronomical")
  }

  /// The chat window and the menu-bar agent check the same engine, so the
  /// chat window must expect the agent's state directory label for every
  /// channel. When each had its own definition the store build's two halves
  /// disagreed about one engine (issue #2188).
  func test_should_expect_the_same_state_directory_label_as_the_agent_for_every_channel() {
    for channel in [ApplicationChannel.stable, .development, .appStore] {
      let menuIdentity = menuIdentity(channel: channel)

      let mapped = ChatLaunchIdentity.chatIdentity(from: menuIdentity)

      XCTAssertEqual(
        mapped.expectedServerStateDirectory, menuIdentity.expectedServerStateDirectory,
        "channel \(channel.rawValue)")
    }
  }

  func test_should_expect_the_group_container_for_a_store_build_with_an_app_group() {
    let groupedStoreIdentity = ApplicationIdentity(
      channel: .appStore,
      daemonPort: ApplicationChannel.appStore.defaultDaemonPort,
      stateDirectoryName: nil,
      version: "9.9",
      buildNumber: "999",
      commit: "test-commit",
      isDirty: false,
      applicationGroupIdentifier: "TEAM123456.astronomical")

    let mapped = ChatLaunchIdentity.chatIdentity(from: groupedStoreIdentity)

    XCTAssertEqual(
      mapped.expectedServerStateDirectory,
      "~/Library/Group Containers/TEAM123456.astronomical/Astronomical")
  }

  func test_should_keep_the_home_dot_folder_label_for_the_direct_channels() {
    XCTAssertEqual(
      ChatLaunchIdentity.chatIdentity(from: menuIdentity(channel: .stable))
        .expectedServerStateDirectory, "~/.astronomical")
    XCTAssertEqual(
      ChatLaunchIdentity.chatIdentity(from: menuIdentity(channel: .development))
        .expectedServerStateDirectory, "~/.astronomical-dev")
  }

  /// The engine knows two instances; the store build must name Stable's, not
  /// its distribution channel, both when it starts the engine and when it
  /// checks the engine's answer.
  func test_should_name_the_stable_runtime_instance_for_the_store_channel() {
    XCTAssertEqual(ApplicationChannel.appStore.runtimeInstanceName, "stable")
    XCTAssertEqual(ApplicationChannel.stable.runtimeInstanceName, "stable")
    XCTAssertEqual(ApplicationChannel.development.runtimeInstanceName, "development")
  }
}
