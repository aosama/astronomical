// Reads the public update feed, the same file the main app's updater reads.
//
// The agent reads it only to notice that a newer version exists. It never
// downloads or installs one, and it does not check the feed's signature: the
// worst a wrong feed can do here is show a notice, after which the main app's
// updater, which does verify signatures, reports that there is nothing to
// install.

import Foundation

/// One release in the feed, reduced to what deciding "is this newer?" needs.
struct UpdateFeedItem: Equatable {
  /// The feed's `sparkle:version`: the bundle's build number, which is what
  /// the updater compares. The display version is only shown to people.
  let buildNumber: Int
  let displayVersion: String
  /// The release channel, or nil for a Stable release.
  let channelName: String?
  let minimumSystemVersion: String?
}

enum UpdateFeedError: Error, Equatable {
  case unreadableFeed
}

enum UpdateFeedParser {
  /// The fields a release may carry either as child elements or, in feeds
  /// written by older tools, as attributes of its enclosure.
  private static let buildNumberField = "sparkle:version"
  private static let displayVersionField = "sparkle:shortVersionString"
  private static let channelField = "sparkle:channel"
  private static let minimumSystemVersionField = "sparkle:minimumSystemVersion"

  /// The releases a feed lists. A feed that is not well-formed XML throws, so
  /// a broken download is never mistaken for "no update exists".
  ///
  /// The whole document is read at once: a feed is a few kilobytes, and a
  /// tree is simpler to get right than a streaming reader's bookkeeping.
  static func items(fromFeedData feedData: Data) throws -> [UpdateFeedItem] {
    let feedDocument: XMLDocument
    do {
      // The feed needs nothing from outside itself, so nothing is fetched on
      // its say-so.
      feedDocument = try XMLDocument(data: feedData, options: [.nodeLoadExternalEntitiesNever])
    } catch {
      throw UpdateFeedError.unreadableFeed
    }
    guard let feedRoot = feedDocument.rootElement() else { throw UpdateFeedError.unreadableFeed }
    var feedItems: [UpdateFeedItem] = []
    for channelElement in feedRoot.elements(forName: "channel") {
      for itemElement in channelElement.elements(forName: "item") {
        if let feedItem = feedItem(from: itemElement) { feedItems.append(feedItem) }
      }
    }
    return feedItems
  }

  /// A release without a numeric build number cannot be compared with the
  /// installed build, so it is left out rather than guessed at.
  private static func feedItem(from itemElement: XMLElement) -> UpdateFeedItem? {
    guard let buildNumberText = fieldText(buildNumberField, in: itemElement),
      let buildNumber = Int(buildNumberText)
    else { return nil }
    return UpdateFeedItem(
      buildNumber: buildNumber,
      displayVersion: fieldText(displayVersionField, in: itemElement) ?? buildNumberText,
      channelName: fieldText(channelField, in: itemElement),
      minimumSystemVersion: fieldText(minimumSystemVersionField, in: itemElement))
  }

  /// A field's text from the release's own child element, or failing that
  /// from the attribute of the same name on its enclosure.
  private static func fieldText(_ fieldName: String, in itemElement: XMLElement) -> String? {
    if let fieldElement = itemElement.elements(forName: fieldName).first,
      let elementText = fieldElement.stringValue
    {
      return elementText.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    for enclosureElement in itemElement.elements(forName: "enclosure") {
      if let attributeText = enclosureElement.attribute(forName: fieldName)?.stringValue {
        return attributeText.trimmingCharacters(in: .whitespacesAndNewlines)
      }
    }
    return nil
  }
}
