import Foundation

/// Lets exactly one menu-bar agent of a channel run.
///
/// macOS can start the agent twice at the same moment: registering it as a
/// login item makes launchd start it at once, while the main app opens it
/// through Launch Services, and the two launch paths do not know about each
/// other. Two agents mean two menu-bar items and two owners of one engine.
/// A check of the running applications cannot settle this, because both
/// agents start before either is listed; an exclusive file lock can, since
/// the kernel grants it to one process only and releases it when that
/// process ends, however it ends.
public final class AgentInstanceLock {
  static let lockFileName = "menu-agent.lock"

  private let lockFileDescriptor: Int32

  private init(lockFileDescriptor: Int32) {
    self.lockFileDescriptor = lockFileDescriptor
  }

  deinit {
    close(lockFileDescriptor)
  }

  /// Returns the lock when this process is the only agent using
  /// `stateDirectoryURL`, and nil when another agent already holds it. A
  /// state directory or lock file that cannot be created throws instead, so
  /// a disk fault is never mistaken for a second agent.
  public static func acquire(inStateDirectory stateDirectoryURL: URL) throws -> AgentInstanceLock? {
    try FileManager.default.createDirectory(at: stateDirectoryURL, withIntermediateDirectories: true)
    let lockFilePath = stateDirectoryURL.appendingPathComponent(lockFileName).path
    let lockFileDescriptor = open(lockFilePath, O_CREAT | O_RDWR | O_CLOEXEC, 0o600)
    guard lockFileDescriptor >= 0 else {
      throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
    guard flock(lockFileDescriptor, LOCK_EX | LOCK_NB) == 0 else {
      let lockErrorNumber = errno
      close(lockFileDescriptor)
      if lockErrorNumber == EWOULDBLOCK { return nil }
      throw POSIXError(POSIXErrorCode(rawValue: lockErrorNumber) ?? .EIO)
    }
    return AgentInstanceLock(lockFileDescriptor: lockFileDescriptor)
  }
}

/// Holds the agent's lock for the life of the process, or ends a second agent
/// before it shows a menu-bar item or touches the engine. A lock that cannot
/// be taken for any other reason lets the agent run: one agent with a
/// possible duplicate is better than no engine at all.
@MainActor
public enum SingleAgentGuard {
  private static var heldInstanceLock: AgentInstanceLock?

  public static func endThisProcessIfAnotherAgentRuns(
    applicationIdentity: ApplicationIdentity = .current()
  ) {
    do {
      guard let instanceLock = try AgentInstanceLock.acquire(
        inStateDirectory: applicationIdentity.stateDirectoryURL())
      else { exit(EXIT_SUCCESS) }
      heldInstanceLock = instanceLock
    } catch {
      NSLog("Astronomical agent could not take its single-instance lock: %@", "\(error)")
    }
  }
}
