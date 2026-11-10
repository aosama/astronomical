import AppKit
import os

/// Starts the menu-bar agent bundled inside the main app, and waits for it to
/// leave when an update is about to replace the bundle.
///
/// The agent owns the engine, so the main app makes sure it is running at
/// launch; when the agent already started at login this does nothing.
@MainActor
struct AgentLauncher {
  /// How long a quitting agent gets before an update goes ahead. Its quit
  /// stops the engine first, which may still be releasing a model's memory.
  static let agentExitTimeout = Duration.seconds(10)
  private static let agentExitPollInterval: TimeInterval = 0.1

  private let agentBundleURL: URL?
  private let agentBundleIdentifier: String
  private let logger = Logger(subsystem: "dev.astronomical.app", category: "agent-launcher")

  init(applicationChannel: ApplicationChannel, applicationBundle: Bundle = .main) {
    // A bare executable run has no bundle, so there is no agent to start; the
    // developer starts the pieces by hand.
    self.init(
      agentBundleIdentifier: applicationChannel.agentBundleIdentifier,
      agentBundleURL: applicationBundle.bundleIdentifier == nil
        ? nil : ApplicationBundleLayout.agentBundleURL(inApplicationBundleAt: applicationBundle.bundleURL))
  }

  /// Names the agent directly, so tests can stand a known process in for it.
  init(agentBundleIdentifier: String, agentBundleURL: URL?) {
    self.agentBundleIdentifier = agentBundleIdentifier
    self.agentBundleURL = agentBundleURL
  }

  var isAgentRunning: Bool {
    !NSRunningApplication.runningApplications(withBundleIdentifier: agentBundleIdentifier).isEmpty
  }

  func launchAgentIfNeeded() {
    guard let agentBundleURL, !isAgentRunning else { return }
    let launchConfiguration = NSWorkspace.OpenConfiguration()
    // The agent has no window; starting it must not take focus from chat.
    launchConfiguration.activates = false
    launchConfiguration.addsToRecentItems = false
    let logger = logger
    NSWorkspace.shared.openApplication(at: agentBundleURL, configuration: launchConfiguration) {
      _, launchError in
      if let launchError {
        logger.error("The menu-bar agent did not start: \(launchError.localizedDescription, privacy: .public)")
      }
    }
  }

  /// Returns whether the agent left within `timeout`. The run loop keeps
  /// turning while waiting, so the app stays responsive to the update's own
  /// progress window.
  func waitForAgentExit(timeout: Duration = AgentLauncher.agentExitTimeout) -> Bool {
    let exitDeadline = ContinuousClock.now + timeout
    while isAgentRunning {
      if ContinuousClock.now >= exitDeadline { return false }
      RunLoop.current.run(until: Date(timeIntervalSinceNow: Self.agentExitPollInterval))
    }
    return true
  }
}
