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

/// A link that puts a draft in the Development chat's message field:
/// `astronomical-dev://draft?text=...`.
///
/// It exists so a developer, or an agent testing the app, can type a question
/// without taking the keyboard from whoever is using the Mac. Three limits
/// keep it harmless: only the Development channel answers it, so a web page
/// cannot reach the app people actually use; it only fills the field, so
/// nothing is sent to a model until someone presses Send; and the text is
/// bounded.
public enum DraftLink {
  public static let destinationName = "draft"
  public static let textParameterName = "text"
  /// The page refuses a longer draft (`MAXIMUM_DRAFT_CHARACTERS` in
  /// HostCommand.ts); refusing here too keeps a huge link out of the bridge.
  public static let maximumDraftCharacters = 4000

  public static func url(draftText: String, channel: ApplicationChannel) -> URL? {
    var linkComponents = URLComponents()
    linkComponents.scheme = channel.chatURLScheme
    linkComponents.host = destinationName
    linkComponents.queryItems = [URLQueryItem(name: textParameterName, value: draftText)]
    return linkComponents.url
  }

  /// The draft a link carries, or nil when the link is not a draft link for
  /// this channel, the channel is not Development, or the text is empty or
  /// too long.
  public static func draftText(from openedURL: URL, channel: ApplicationChannel) -> String? {
    guard channel == .development, openedURL.scheme?.lowercased() == channel.chatURLScheme,
      openedURL.host?.lowercased() == destinationName,
      let linkComponents = URLComponents(url: openedURL, resolvingAgainstBaseURL: false),
      let draftText = linkComponents.queryItems?.first(where: { $0.name == textParameterName })?.value
    else { return nil }
    let hasText = !draftText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    return hasText && draftText.count <= maximumDraftCharacters ? draftText : nil
  }
}

/// A link that tells the Development chat where to put its next export, so
/// that one export is saved without the save panel.
///
/// The save panel is a system window that comes to the front, so checking an
/// export on a live build took the screen from whoever was using the Mac.
/// `cargo xtask app export-to <folder>` opens this link in the background
/// before the export is saved from the sheet. Only the Development channel
/// answers it: in the app people use, the person always chooses where a file
/// goes, and a web page that opened such a link must not be able to choose
/// for them.
public enum ExportFolderLink {
  public static let destinationName = "export-folder"
  public static let pathParameterName = "path"

  public static func url(folderPath: String, channel: ApplicationChannel) -> URL? {
    var linkComponents = URLComponents()
    linkComponents.scheme = channel.chatURLScheme
    linkComponents.host = destinationName
    linkComponents.queryItems = [URLQueryItem(name: pathParameterName, value: folderPath)]
    return linkComponents.url
  }

  /// The folder a link names, or nil when the link is not an export-folder
  /// link for this channel, the channel is not Development, or the path is
  /// not an absolute one. Whether the folder exists is checked when the
  /// export is written, since it may be made in between.
  public static func folderURL(from openedURL: URL, channel: ApplicationChannel) -> URL? {
    guard channel == .development, openedURL.scheme?.lowercased() == channel.chatURLScheme,
      openedURL.host?.lowercased() == destinationName,
      let linkComponents = URLComponents(url: openedURL, resolvingAgainstBaseURL: false),
      let folderPath = linkComponents.queryItems?.first(where: { $0.name == pathParameterName })?.value,
      folderPath.hasPrefix("/"), folderPath.count > 1
    else { return nil }
    return URL(fileURLWithPath: folderPath, isDirectory: true).standardizedFileURL
  }
}
