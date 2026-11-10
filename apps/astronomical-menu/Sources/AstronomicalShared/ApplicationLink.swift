// The links macOS hands to the main app through its channel's URL scheme. The
// Observatory console opens chat this way, and so does the menu-bar agent,
// which has no window of its own: opening a link is how it asks the main app
// to show one, and macOS starts the main app first when it is not running.
// The contract lives in one pure, testable place.

import Foundation

public enum ApplicationLink: String, CaseIterable, Sendable {
  /// The conversation window.
  case chat
  /// The conversation window, opened on Your Astro.
  case yourAstro = "your-astro"
  /// An update check. The main app owns the updater, so the agent's popover
  /// asks for a check this way. The link only ever shows the updater's own
  /// prompt, which is why it is safe for anything on the Mac to open it.
  case checkForUpdates = "check-for-updates"

  public func url(channel: ApplicationChannel) -> URL? {
    URL(string: "\(channel.chatURLScheme)://\(rawValue)")
  }

  /// Refuses another channel's scheme, so an app that macOS routes the wrong
  /// link to can never open a window attached to a different instance.
  public static func parse(_ openedURL: URL, channel: ApplicationChannel) -> ApplicationLink? {
    guard openedURL.scheme?.lowercased() == channel.chatURLScheme else { return nil }
    // For a custom scheme, URL parses astronomical://chat with "chat" as the
    // host, not as a path component; the path form is accepted too so a
    // hand-written variant of the link still opens the window.
    let trailingPathComponents = openedURL.pathComponents.filter { $0 != "/" }
    let destinationName: String?
    if let hostName = openedURL.host, !hostName.isEmpty {
      destinationName = trailingPathComponents.isEmpty ? hostName.lowercased() : nil
    } else {
      destinationName = trailingPathComponents.count == 1 ? trailingPathComponents[0].lowercased() : nil
    }
    guard let destinationName else { return nil }
    return ApplicationLink(rawValue: destinationName)
  }
}

public func isThinTalkChatOpenRequest(_ openedURL: URL, channel: ApplicationChannel) -> Bool {
  ApplicationLink.parse(openedURL, channel: channel) == .chat
}
