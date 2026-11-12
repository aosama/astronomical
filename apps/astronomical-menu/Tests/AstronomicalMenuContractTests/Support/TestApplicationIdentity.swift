@testable import AstronomicalShared

/// The identity of a build on `channel` with that channel's own port and state
/// directory, for tests whose subject is not the identity itself.
func testApplicationIdentity(channel: ApplicationChannel) -> ApplicationIdentity {
  ApplicationIdentity(
    channel: channel,
    daemonPort: channel.defaultDaemonPort,
    stateDirectoryName: channel.stateDirectoryName,
    version: "1.0.0",
    buildNumber: "1",
    commit: "abcdef0",
    isDirty: false)
}
