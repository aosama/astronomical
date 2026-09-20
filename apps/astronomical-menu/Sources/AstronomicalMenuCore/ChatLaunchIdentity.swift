import Foundation
import ThinTalkCore

/// Maps the menu application's channel identity onto the chat surface's wire
/// identity. The chat client validates the supervisor's channel and state
/// directory during its handshake, so this mapping has to agree exactly with
/// what the bundled daemon advertises for the same instance. The App Store
/// build presents the Stable product identity (see ApplicationChannel), so it
/// maps to the Stable conversation channel.
enum ChatLaunchIdentity {
  static func chatIdentity(
    from applicationIdentity: ApplicationIdentity
  ) -> ThinTalkCore.ThinTalkApplicationIdentity {
    let channel: ThinTalkCore.ThinTalkChannel =
      applicationIdentity.channel == .development ? .development : .stable
    return ThinTalkCore.ThinTalkApplicationIdentity(
      channel: channel,
      supervisorPort: applicationIdentity.supervisorPort,
      stateDirectoryName: chatStateDirectoryName(
        channelOverride: applicationIdentity.stateDirectoryName),
      version: applicationIdentity.version,
      buildNumber: applicationIdentity.buildNumber,
      commit: applicationIdentity.commit,
      isDirty: applicationIdentity.isDirty)
  }

  private static func chatStateDirectoryName(channelOverride: String?) -> String {
    guard let channelOverride else { return ThinTalkCore.ThinTalkChannel.stable.stateDirectoryName }
    return channelOverride
  }
}
