// Holds what the agent knows about an available update, for the popover.

import Foundation

/// Downloads the update feed. A protocol so tests supply a feed without a
/// network.
protocol UpdateFeedFetching: Sendable {
  func feedData(from feedURL: URL) async throws -> Data
}

struct URLSessionUpdateFeedFetcher: UpdateFeedFetching {
  private static let requestTimeoutSeconds: TimeInterval = 30

  func feedData(from feedURL: URL) async throws -> Data {
    // The stored copy is skipped on purpose: a release is announced by
    // replacing this one file, and a cached answer would hide it for as long
    // as the cache lives.
    var feedRequest = URLRequest(
      url: feedURL, cachePolicy: .reloadIgnoringLocalCacheData,
      timeoutInterval: Self.requestTimeoutSeconds)
    feedRequest.httpMethod = "GET"
    let (feedData, feedResponse) = try await URLSession.shared.data(for: feedRequest)
    if let httpResponse = feedResponse as? HTTPURLResponse, httpResponse.statusCode != 200 {
      throw UpdateFeedError.unreadableFeed
    }
    return feedData
  }
}

@MainActor
final class UpdateNoticeStore: ObservableObject {
  /// The update the popover mentions, or nil when there is nothing to say.
  @Published private(set) var availableUpdate: AvailableUpdate?

  private let installedBuildNumber: Int
  private let settingsReader: any UpdateSettingsReading
  private let feedFetcher: any UpdateFeedFetching
  private let operatingSystemVersion: OperatingSystemVersion
  /// The feed as last downloaded, kept so that a choice changed in the main
  /// app (skipping a version, turning automatic checks off) can be applied
  /// without asking the network again.
  private var latestFeedItems: [UpdateFeedItem] = []

  init(
    installedBuildNumber: Int,
    settingsReader: any UpdateSettingsReading,
    feedFetcher: any UpdateFeedFetching = URLSessionUpdateFeedFetcher(),
    operatingSystemVersion: OperatingSystemVersion = ProcessInfo.processInfo.operatingSystemVersion
  ) {
    self.installedBuildNumber = installedBuildNumber
    self.settingsReader = settingsReader
    self.feedFetcher = feedFetcher
    self.operatingSystemVersion = operatingSystemVersion
  }

  /// Downloads the feed and decides again. With automatic checks off nothing
  /// is downloaded. A download that fails keeps what was known: one bad
  /// connection must not make a real update vanish from the popover.
  func checkFeed() async {
    guard let updateSettings = settingsReader.currentUpdateSettings() else {
      forgetFeed()
      return
    }
    guard updateSettings.preferences.checksAutomatically else {
      forgetFeed()
      return
    }
    do {
      let feedData = try await feedFetcher.feedData(from: updateSettings.feedURL)
      latestFeedItems = try UpdateFeedParser.items(fromFeedData: feedData)
    } catch {
      NSLog("Could not read the update feed: \(error.localizedDescription)")
    }
    applyCurrentSettings()
  }

  /// Decides again from the feed already downloaded and the settings as they
  /// are now. Costs no request, so the popover calls it each time it opens.
  func applyCurrentSettings() {
    guard let updateSettings = settingsReader.currentUpdateSettings() else {
      forgetFeed()
      return
    }
    availableUpdate = AvailableUpdatePolicy.newestApplicableUpdate(
      feedItems: latestFeedItems,
      installedBuildNumber: installedBuildNumber,
      preferences: updateSettings.preferences,
      operatingSystemVersion: operatingSystemVersion)
  }

  private func forgetFeed() {
    latestFeedItems = []
    availableUpdate = nil
  }
}
