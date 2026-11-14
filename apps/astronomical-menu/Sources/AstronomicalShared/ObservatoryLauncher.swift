import AppKit

/// Opens a page of the Observatory, the local web console the engine serves,
/// in the default browser. Both the main app's menu and the agent's popover
/// offer it, so the one way to open it lives here.
@MainActor
public enum ObservatoryLauncher {
  /// `path` is the page's path on the instance's local server, such as "/"
  /// for the Overview. A failure is shown to the person as an alert.
  public static func open(path: String, applicationIdentity: ApplicationIdentity) {
    do {
      let observatoryURL = try applicationIdentity.endpointURL(path: path)
      guard NSWorkspace.shared.open(observatoryURL) else {
        throw ObservatoryLaunchError.defaultBrowserUnavailable
      }
    } catch {
      NSApp.presentError(error)
    }
  }

  /// Opens a page as `open` does, but first waits for the engine to answer.
  ///
  /// A link can be what starts the app. The engine then needs a moment before
  /// it serves any page, and a browser opened sooner shows "cannot connect",
  /// which reads as a broken app to someone who has just clicked a link. The
  /// wait is bounded: after it, the page opens anyway, so a slow start still
  /// ends in a page the person can reload.
  public static func openOnceEngineAnswers(path: String, applicationIdentity: ApplicationIdentity) {
    Task { @MainActor in
      if let statusURL = try? applicationIdentity.endpointURL(path: "/v1/status") {
        await waitUntilReachable(statusURL)
      }
      open(path: path, applicationIdentity: applicationIdentity)
    }
  }

  private static func waitUntilReachable(_ statusURL: URL) async {
    var statusRequest = URLRequest(url: statusURL)
    statusRequest.timeoutInterval = 1
    for _ in 0..<engineAnswerAttemptLimit {
      if let (_, statusResponse) = try? await URLSession.shared.data(for: statusRequest),
        (statusResponse as? HTTPURLResponse)?.statusCode == 200
      {
        return
      }
      try? await Task.sleep(nanoseconds: engineAnswerRetryNanoseconds)
    }
  }

  /// Forty tries half a second apart: about twenty seconds, longer than a
  /// cold start of the engine takes.
  private static let engineAnswerAttemptLimit = 40
  private static let engineAnswerRetryNanoseconds: UInt64 = 500_000_000
}

enum ObservatoryLaunchError: LocalizedError {
  case defaultBrowserUnavailable

  var errorDescription: String? {
    "The Observatory could not be opened in the default browser"
  }
}
