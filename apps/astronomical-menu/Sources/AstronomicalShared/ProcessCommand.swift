// The few things the main app and its menu-bar agent ask of each other.
//
// Distributed notifications carry them: both processes are sandbox-compatible
// with that (a sandboxed app may post and observe them as long as it sends no
// userInfo), and neither has to be running for the other to post. Names start
// with the channel's bundle identifier, so Stable and Development, which can
// run side by side, never hear each other's commands.

import Foundation

public enum ProcessCommand: String, CaseIterable, Sendable {
  /// Main app to agent: restart the engine.
  case restartEngine = "agent.restart-engine"
  /// Main app to agent: quit, which stops the engine the agent owns.
  case quitAgent = "agent.quit"
  /// Main app to agent: the detail is the sender's build fingerprint; an
  /// agent from a different build relaunches itself.
  case checkAgentBuild = "agent.build-check"
  /// Agent to main app: quit, so the menu-bar Quit still stops everything.
  case quitApplication = "application.quit"

  public func notificationName(channel: ApplicationChannel) -> Notification.Name {
    notificationName(commandNamePrefix: channel.applicationBundleIdentifier)
  }

  func notificationName(commandNamePrefix: String) -> Notification.Name {
    Notification.Name("\(commandNamePrefix).\(rawValue)")
  }
}

@MainActor
public struct ProcessCommandChannel {
  private let commandNamePrefix: String
  private let notificationCenter: DistributedNotificationCenter

  public init(channel: ApplicationChannel, notificationCenter: DistributedNotificationCenter = .default()) {
    self.init(commandNamePrefix: channel.applicationBundleIdentifier, notificationCenter: notificationCenter)
  }

  /// A channel under a prefix no installed app uses. Distributed notifications
  /// reach every process on the Mac, so a test that posted on a real channel's
  /// names would command a running Astronomical: a build check from a test
  /// once made the Development agent quit and stop its engine.
  init(commandNamePrefix: String, notificationCenter: DistributedNotificationCenter = .default()) {
    self.commandNamePrefix = commandNamePrefix
    self.notificationCenter = notificationCenter
  }

  /// `detail` travels as the notification's object, the one payload a
  /// sandboxed process may send.
  public func post(_ command: ProcessCommand, detail: String? = nil) {
    notificationCenter.postNotificationName(
      command.notificationName(commandNamePrefix: commandNamePrefix), object: detail, userInfo: nil,
      deliverImmediately: true)
  }

  /// The returned token keeps the observation alive; hold it for as long as
  /// the command should be answered.
  public func observe(
    _ command: ProcessCommand, perform: @escaping @MainActor (_ detail: String?) -> Void
  ) -> NSObjectProtocol {
    notificationCenter.addObserver(
      forName: command.notificationName(commandNamePrefix: commandNamePrefix), object: nil, queue: .main
    ) { notification in
      let detail = notification.object as? String
      MainActor.assumeIsolated { perform(detail) }
    }
  }
}
