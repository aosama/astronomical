import Foundation
import ThinTalkCore

/// Maps the menu application's channel identity onto the chat surface's wire
/// identity. The chat client validates the daemon's channel and state
/// directory during its handshake, so this mapping has to agree exactly with
/// what the bundled daemon advertises for the same instance. The App Store
/// build presents the Stable product identity (see ApplicationChannel), so it
/// maps to the Stable conversation channel. Its engine keeps its state in the
/// app group container, not the home dot-folder, so the expected state
/// directory label is handed over from the one definition the menu-bar agent
/// also checks against; two definitions let the chat window and the agent
/// disagree about the same engine (issue #2188).
enum ChatLaunchIdentity {
  static func chatIdentity(
    from applicationIdentity: ApplicationIdentity
  ) -> ThinTalkCore.ThinTalkApplicationIdentity {
    let channel: ThinTalkCore.ThinTalkChannel =
      applicationIdentity.channel == .development ? .development : .stable
    return ThinTalkCore.ThinTalkApplicationIdentity(
      channel: channel,
      daemonPort: applicationIdentity.daemonPort,
      stateDirectoryName: chatStateDirectoryName(
        channel: channel, channelOverride: applicationIdentity.stateDirectoryName),
      version: applicationIdentity.version,
      buildNumber: applicationIdentity.buildNumber,
      commit: applicationIdentity.commit,
      isDirty: applicationIdentity.isDirty,
      expectedServerStateDirectory: applicationIdentity.expectedServerStateDirectory)
  }

  /// Only the App Store build has no bundled name. The fallback follows the chat
  /// channel, so a Development identity can never resolve to Stable's directory.
  private static func chatStateDirectoryName(
    channel: ThinTalkCore.ThinTalkChannel, channelOverride: String?
  ) -> String {
    guard let channelOverride else { return channel.stateDirectoryName }
    return channelOverride
  }
}
