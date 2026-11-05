// Decides whether a URL handed to the app by macOS asks for this channel's
// Thin Talk conversation window. The Observatory console links chat here through
// its channel's scheme (see ApplicationChannel.chatURLScheme), so the link
// contract between the daemon's web console and this app lives in one pure,
// testable place.

import Foundation

/// Refuses another channel's scheme, so an app that macOS routes the wrong link
/// to can never open a chat window attached to a different instance.
func isThinTalkChatOpenRequest(_ openedURL: URL, channel: ApplicationChannel) -> Bool {
  guard openedURL.scheme?.lowercased() == channel.chatURLScheme else { return false }
  // For a custom scheme, URL parses astronomical://chat with "chat" as the
  // host, not as a path component; the path form is accepted too so a
  // hand-written variant of the link still opens the conversation window.
  let hasChatHost = openedURL.host?.lowercased() == "chat"
  let trailingPathComponents = openedURL.pathComponents.filter { $0 != "/" }
  if hasChatHost {
    return trailingPathComponents.isEmpty
  }
  return trailingPathComponents == ["chat"]
}
