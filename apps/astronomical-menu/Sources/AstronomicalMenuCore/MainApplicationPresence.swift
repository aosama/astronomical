import AppKit

/// Says whether the main app is running. While it is, its own updater shows
/// its dialog on schedule, and a notification from the agent on top of that
/// would be the same news twice.
@MainActor
protocol MainApplicationPresenceChecking {
  var isMainApplicationRunning: Bool { get }
}

@MainActor
struct RunningMainApplicationPresence: MainApplicationPresenceChecking {
  let mainApplicationBundleIdentifier: String

  var isMainApplicationRunning: Bool {
    !NSRunningApplication.runningApplications(withBundleIdentifier: mainApplicationBundleIdentifier).isEmpty
  }
}
