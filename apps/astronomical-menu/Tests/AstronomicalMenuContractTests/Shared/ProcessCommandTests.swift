import Foundation
import XCTest

@testable import AstronomicalShared

/// The main app and its agent reach each other only through these commands.
/// Stable and Development can run side by side, so a command of one must
/// never be heard by the other.
@MainActor
final class ProcessCommandTests: XCTestCase {
  func test_should_name_every_command_differently_within_a_channel() {
    let commandNames = ProcessCommand.allCases.map { $0.notificationName(channel: .stable).rawValue }

    XCTAssertEqual(Set(commandNames).count, ProcessCommand.allCases.count)
  }

  func test_should_never_share_a_command_name_between_channels() {
    let channelNameSets = [ApplicationChannel.stable, .development, .appStore].map { applicationChannel in
      Set(ProcessCommand.allCases.map { $0.notificationName(channel: applicationChannel).rawValue })
    }

    XCTAssertTrue(channelNameSets[0].isDisjoint(with: channelNameSets[1]))
    XCTAssertTrue(channelNameSets[0].isDisjoint(with: channelNameSets[2]))
    XCTAssertTrue(channelNameSets[1].isDisjoint(with: channelNameSets[2]))
  }

  func test_should_start_command_names_with_the_channels_bundle_identifier() {
    XCTAssertEqual(
      ProcessCommand.quitAgent.notificationName(channel: .development).rawValue,
      "dev.astronomical.app.development.agent.quit")
  }

  /// A helper's identifier must extend its main app's, which is also what the
  /// main app registers as its login item.
  func test_should_derive_the_agent_identifier_from_the_application_identifier() {
    for applicationChannel in [ApplicationChannel.stable, .development, .appStore] {
      XCTAssertEqual(
        applicationChannel.agentBundleIdentifier, applicationChannel.applicationBundleIdentifier + ".agent")
    }
    XCTAssertEqual(ApplicationChannel.stable.applicationBundleIdentifier, "dev.astronomical.app")
    XCTAssertEqual(ApplicationChannel.appStore.applicationBundleIdentifier, "app.astronomical.appstore")
  }

  /// Posted under a prefix made for this run: distributed notifications reach
  /// every process, and a real channel's name would command a running app.
  func test_should_deliver_a_posted_command_with_its_detail_to_an_observer() {
    let commandChannel = ProcessCommandChannel(
      commandNamePrefix: "dev.astronomical.tests.\(UUID().uuidString)")
    let commandArrived = expectation(description: "the observer receives the command")
    var receivedDetail: String?
    let observation = commandChannel.observe(.checkAgentBuild) { detail in
      receivedDetail = detail
      commandArrived.fulfill()
    }
    defer { DistributedNotificationCenter.default().removeObserver(observation) }

    commandChannel.post(.checkAgentBuild, detail: "1.2.3|45|abc|false")

    wait(for: [commandArrived], timeout: 5)
    XCTAssertEqual(receivedDetail, "1.2.3|45|abc|false")
  }

  /// The guard against the defect above: nothing in the test suite may build a
  /// command channel for a real instance.
  func test_should_keep_every_test_off_the_real_channels() throws {
    let realChannelConstruction = "ProcessCommandChannel(" + "channel:"
    for testFileURL in try PackageSources.swiftFileURLs(under: "Tests") {
      let testSource = try String(contentsOf: testFileURL, encoding: .utf8)
      XCTAssertFalse(testSource.contains(realChannelConstruction), testFileURL.lastPathComponent)
    }
  }

  func test_should_change_the_build_fingerprint_with_any_part_of_the_build() {
    func fingerprint(version: String = "1.0.0", buildNumber: String = "7", commit: String = "abc", isDirty: Bool = false)
      -> String
    {
      ApplicationIdentity(
        channel: .stable, daemonPort: 6732, stateDirectoryName: ".astronomical", version: version,
        buildNumber: buildNumber, commit: commit, isDirty: isDirty
      ).buildFingerprint
    }

    let baselineFingerprint = fingerprint()
    XCTAssertEqual(baselineFingerprint, fingerprint())
    XCTAssertNotEqual(baselineFingerprint, fingerprint(version: "1.0.1"))
    XCTAssertNotEqual(baselineFingerprint, fingerprint(buildNumber: "8"))
    XCTAssertNotEqual(baselineFingerprint, fingerprint(commit: "abd"))
    XCTAssertNotEqual(baselineFingerprint, fingerprint(isDirty: true))
  }
}
