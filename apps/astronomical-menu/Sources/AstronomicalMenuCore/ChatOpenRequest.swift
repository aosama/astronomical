// Decides whether a URL handed to the app by macOS asks for the Thin Talk
// conversation window. The Observatory console links chat here through
// astronomical://chat (see its chat view), so the link contract between the
// daemon's web console and this app lives in one pure, testable place.

import Foundation

/// The URL scheme the packaged app registers so a browser (the Observatory)
/// can hand a user to the native conversation window.
let thinTalkURLScheme = "astronomical"

func isThinTalkChatOpenRequest(_ openedURL: URL) -> Bool {
  guard openedURL.scheme?.lowercased() == thinTalkURLScheme else { return false }
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
