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
}

enum ObservatoryLaunchError: LocalizedError {
  case defaultBrowserUnavailable

  var errorDescription: String? {
    "The Observatory could not be opened in the default browser"
  }
}
