import XCTest

@testable import AstronomicalMenuCore

// Stand-ins shared by the update notice tests.

/// A feed the test decides, counting how often it was asked for.
final class StubUpdateFeedFetcher: UpdateFeedFetching, @unchecked Sendable {
  private let stateLock = NSLock()
  private var feedText: String?
  private var requestCount = 0

  init(feedText: String?) { self.feedText = feedText }

  /// nil makes the next downloads fail, as a lost connection would.
  func serve(_ nextFeedText: String?) {
    stateLock.withLock { feedText = nextFeedText }
  }

  var fetchCount: Int { stateLock.withLock { requestCount } }

  func feedData(from feedURL: URL) async throws -> Data {
    let servedFeedText: String? = stateLock.withLock {
      requestCount += 1
      return feedText
    }
    guard let servedFeedText else { throw URLError(.notConnectedToInternet) }
    return Data(servedFeedText.utf8)
  }
}

@MainActor
final class StubUpdateSettingsReader: UpdateSettingsReading {
  var updateSettings: MainApplicationUpdateSettings?

  init(checksAutomatically: Bool = true, skippedBuildNumber: Int? = nil) {
    setPreferences(checksAutomatically: checksAutomatically, skippedBuildNumber: skippedBuildNumber)
  }

  func setPreferences(checksAutomatically: Bool = true, skippedBuildNumber: Int? = nil) {
    var defaultsDomain: [String: Any] = ["SUEnableAutomaticChecks": checksAutomatically]
    if let skippedBuildNumber { defaultsDomain["SUSkippedVersion"] = String(skippedBuildNumber) }
    let settingsDefaults = UserDefaults(suiteName: "UpdateNoticeTests.settings")!
    settingsDefaults.removePersistentDomain(forName: "UpdateNoticeTests.settings")
    settingsDefaults.setPersistentDomain(defaultsDomain, forName: "UpdateNoticeTests.settings")
    updateSettings = MainApplicationUpdateSettings(
      mainApplicationInfo: ["SUFeedURL": "https://example.invalid/appcast.xml"],
      mainApplicationDefaults: settingsDefaults, applicationChannel: .stable)
  }

  func currentUpdateSettings() -> MainApplicationUpdateSettings? { updateSettings }
}

@MainActor
final class StubUpdateNoticePresenter: UpdateNoticePresenting {
  var isNotificationAccepted = true
  private(set) var presentedUpdates: [AvailableUpdate] = []

  func present(_ availableUpdate: AvailableUpdate) async -> Bool {
    guard isNotificationAccepted else { return false }
    presentedUpdates.append(availableUpdate)
    return true
  }
}

@MainActor
final class StubMainApplicationPresence: MainApplicationPresenceChecking {
  var isMainApplicationRunning = false
}

func updateFeedText(buildNumber: Int, displayVersion: String) -> String {
  """
  <rss xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle"><channel><item>
    <sparkle:version>\(buildNumber)</sparkle:version>
    <sparkle:shortVersionString>\(displayVersion)</sparkle:shortVersionString>
  </item></channel></rss>
  """
}
