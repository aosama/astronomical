import Foundation
import ThinTalkCore
import XCTest

final class ApplicationIdentityTests: XCTestCase {
  func test_should_keep_each_bundled_channel_port_and_state_directory_together() {
    for channel in [ThinTalkChannel.stable, .development] {
      let identityBundle = ApplicationIdentityBundle(identityFields: [
        "AstronomicalChannel": channel.rawValue,
        "AstronomicalDaemonPort": channel.defaultDaemonPort,
        "AstronomicalStateDirectoryName": channel.stateDirectoryName,
        "CFBundleShortVersionString": "1.2.3",
        "CFBundleVersion": "42",
        "AstronomicalBuildCommit": "bundle-commit",
        "AstronomicalBuildDirty": true,
      ])

      let identity = ThinTalkApplicationIdentity.current(bundle: identityBundle)

      XCTAssertEqual(identity.channel, channel)
      XCTAssertEqual(identity.daemonPort, channel.defaultDaemonPort)
      XCTAssertEqual(identity.stateDirectoryName, channel.stateDirectoryName)
      XCTAssertEqual(identity.version, "1.2.3")
      XCTAssertEqual(identity.buildNumber, "42")
      XCTAssertEqual(identity.commit, "bundle-commit")
      XCTAssertTrue(identity.isDirty)
    }
  }

  func test_should_use_channel_defaults_when_bundle_connection_fields_are_missing() {
    for channel in [ThinTalkChannel.stable, .development] {
      let identityBundle = ApplicationIdentityBundle(identityFields: [
        "AstronomicalChannel": channel.rawValue,
      ])

      let identity = ThinTalkApplicationIdentity.current(bundle: identityBundle)

      XCTAssertEqual(identity.channel, channel)
      XCTAssertEqual(identity.daemonPort, channel.defaultDaemonPort)
      XCTAssertEqual(identity.stateDirectoryName, channel.stateDirectoryName)
    }
  }

  func test_should_fall_back_to_development_for_missing_or_unknown_bundle_channels() {
    let invalidBundleIdentityFields: [[String: Any]] = [
      [:],
      ["AstronomicalChannel": "unsupported"],
      ["AstronomicalChannel": 7],
    ]
    for identityFields in invalidBundleIdentityFields {
      let identity = ThinTalkApplicationIdentity.current(
        bundle: ApplicationIdentityBundle(identityFields: identityFields))

      XCTAssertEqual(identity.channel, .development)
      XCTAssertEqual(identity.daemonPort, ThinTalkChannel.development.defaultDaemonPort)
      XCTAssertEqual(identity.stateDirectoryName, ThinTalkChannel.development.stateDirectoryName)
      XCTAssertEqual(identity.version, "unknown")
      XCTAssertEqual(identity.buildNumber, "0")
      XCTAssertEqual(identity.commit, "unknown")
      XCTAssertFalse(identity.isDirty)
    }
  }

  func test_should_use_channel_defaults_for_mistyped_bundle_connection_fields() {
    let identityBundle = ApplicationIdentityBundle(identityFields: [
      "AstronomicalChannel": ThinTalkChannel.stable.rawValue,
      "AstronomicalDaemonPort": "not-a-port",
      "AstronomicalStateDirectoryName": 7,
    ])

    let identity = ThinTalkApplicationIdentity.current(bundle: identityBundle)

    XCTAssertEqual(identity.channel, .stable)
    XCTAssertEqual(identity.daemonPort, ThinTalkChannel.stable.defaultDaemonPort)
    XCTAssertEqual(identity.stateDirectoryName, ThinTalkChannel.stable.stateDirectoryName)
  }

  func test_should_preserve_explicit_bundle_connection_fields() {
    let bundledDaemonPort = ThinTalkChannel.development.defaultDaemonPort + 1
    let identityBundle = ApplicationIdentityBundle(identityFields: [
      "AstronomicalChannel": ThinTalkChannel.development.rawValue,
      "AstronomicalDaemonPort": bundledDaemonPort,
      "AstronomicalStateDirectoryName": ".bundle-test-state",
    ])

    let identity = ThinTalkApplicationIdentity.current(bundle: identityBundle)

    XCTAssertEqual(identity.channel, .development)
    XCTAssertEqual(identity.daemonPort, bundledDaemonPort)
    XCTAssertEqual(identity.stateDirectoryName, ".bundle-test-state")
  }

  func test_should_keep_the_explicit_initializer_available_for_isolated_tests() {
    let explicitDaemonPort = ThinTalkChannel.stable.defaultDaemonPort + 1
    let identity = ThinTalkApplicationIdentity(channel: .stable, daemonPort: explicitDaemonPort)

    XCTAssertEqual(identity.channel, .stable)
    XCTAssertEqual(identity.daemonPort, explicitDaemonPort)
    XCTAssertEqual(identity.stateDirectoryName, ThinTalkChannel.stable.stateDirectoryName)
  }
}

private final class ApplicationIdentityBundle: Bundle, @unchecked Sendable {
  private let identityFields: [String: Any]

  init(identityFields: [String: Any]) {
    self.identityFields = identityFields
    super.init()
  }

  override func object(forInfoDictionaryKey key: String) -> Any? {
    identityFields[key]
  }
}
