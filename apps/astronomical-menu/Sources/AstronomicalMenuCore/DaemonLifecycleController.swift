import Darwin
import Foundation

private let daemonShutdownTimeout = Duration.seconds(3)

struct DaemonOwnershipRecord: Codable, Equatable {
  let menuProcessIdentifier: Int32
  let menuExecutablePath: String
  let daemonProcessIdentifier: Int32
  let daemonExecutablePath: String

  func matchesExpectedExecutables(menuExecutablePath: String, daemonExecutablePath: String) -> Bool
  {
    self.menuExecutablePath == menuExecutablePath
      && self.daemonExecutablePath == daemonExecutablePath
  }
}

struct DaemonOwnershipStore {
  let ownershipRecordURL: URL

  func load() throws -> DaemonOwnershipRecord {
    try JSONDecoder().decode(DaemonOwnershipRecord.self, from: Data(contentsOf: ownershipRecordURL))
  }

  func persist(_ ownershipRecord: DaemonOwnershipRecord) throws {
    let ownershipDirectoryURL = ownershipRecordURL.deletingLastPathComponent()
    try FileManager.default.createDirectory(
      at: ownershipDirectoryURL, withIntermediateDirectories: true)
    let temporaryRecordURL = ownershipRecordURL.appendingPathExtension("tmp")
    try JSONEncoder().encode(ownershipRecord).write(to: temporaryRecordURL, options: .atomic)
    if FileManager.default.fileExists(atPath: ownershipRecordURL.path) {
      _ = try FileManager.default.replaceItemAt(ownershipRecordURL, withItemAt: temporaryRecordURL)
    } else {
      try FileManager.default.moveItem(at: temporaryRecordURL, to: ownershipRecordURL)
    }
  }

  func remove() { try? FileManager.default.removeItem(at: ownershipRecordURL) }
}

final class OwnedDaemonProcess {
  let processIdentifier: pid_t
  private var hasBeenReaped = false

  init(processIdentifier: pid_t) { self.processIdentifier = processIdentifier }

  func reapIfExited() -> Bool {
    if hasBeenReaped { return true }
    var childProcessStatus: Int32 = 0
    let waitResult = waitpid(processIdentifier, &childProcessStatus, WNOHANG)
    if waitResult == processIdentifier || (waitResult == -1 && errno == ECHILD) {
      hasBeenReaped = true
      return true
    }
    return false
  }

  func waitUntilExit() {
    guard !hasBeenReaped else { return }
    var childProcessStatus: Int32 = 0
    while waitpid(processIdentifier, &childProcessStatus, 0) == -1 && errno == EINTR {}
    hasBeenReaped = true
  }
}

func launchOwnedDaemonProcess(
  executableURL: URL,
  arguments: [String] = [],
  stderrCapturePath: String? = nil
) throws -> OwnedDaemonProcess {
  var spawnAttributes: posix_spawnattr_t?
  var spawnFileActions: posix_spawn_file_actions_t?
  guard posix_spawnattr_init(&spawnAttributes) == 0,
    posix_spawn_file_actions_init(&spawnFileActions) == 0
  else {
    throw DaemonLifecycleError.cannotInitializeProcessSpawn
  }
  defer {
    posix_spawnattr_destroy(&spawnAttributes)
    posix_spawn_file_actions_destroy(&spawnFileActions)
  }
  // The daemon explains pre-logging exits (lock conflicts, configuration
  // failures) only through stderr. Discarding that stream made every early
  // death invisible to this controller, so stderr is captured to a file in
  // the instance state directory and replayed into the operator diagnostic.
  let stderrOpenSucceeded: Bool
  if let stderrCapturePath {
    stderrOpenSucceeded = posix_spawn_file_actions_addopen(
      &spawnFileActions, STDERR_FILENO, stderrCapturePath,
      O_WRONLY | O_CREAT | O_APPEND, 0o644) == 0
  } else {
    stderrOpenSucceeded = posix_spawn_file_actions_addopen(
      &spawnFileActions, STDERR_FILENO, "/dev/null", O_WRONLY, 0) == 0
  }
  guard posix_spawnattr_setflags(&spawnAttributes, Int16(POSIX_SPAWN_SETPGROUP)) == 0,
    posix_spawnattr_setpgroup(&spawnAttributes, 0) == 0,
    posix_spawn_file_actions_addopen(&spawnFileActions, STDIN_FILENO, "/dev/null", O_RDONLY, 0)
      == 0,
    posix_spawn_file_actions_addopen(&spawnFileActions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0)
      == 0,
    stderrOpenSucceeded
  else {
    throw DaemonLifecycleError.cannotConfigureProcessSpawn
  }

  let executablePath = executableURL.path
  let processArguments = [executablePath] + arguments
  let mutableArgumentPointers = processArguments.map { strdup($0) } + [nil]
  defer {
    for mutableArgumentPointer in mutableArgumentPointers.compactMap({ $0 }) {
      free(mutableArgumentPointer)
    }
  }
  var daemonProcessIdentifier: pid_t = 0
  let spawnStatus = mutableArgumentPointers.withUnsafeBufferPointer { argumentBuffer in
    posix_spawn(
      &daemonProcessIdentifier,
      executablePath,
      &spawnFileActions,
      &spawnAttributes,
      UnsafeMutablePointer(mutating: argumentBuffer.baseAddress),
      environ
    )
  }
  guard spawnStatus == 0 else { throw DaemonLifecycleError.cannotLaunchDaemon(spawnStatus) }
  return OwnedDaemonProcess(processIdentifier: daemonProcessIdentifier)
}

enum DaemonLifecycleError: LocalizedError {
  case cannotInitializeProcessSpawn
  case cannotConfigureProcessSpawn
  case cannotLaunchDaemon(Int32)
  case bundledDaemonUnavailable
  case unownedDaemonDidNotStop
  case existingDaemonNotReady(String)
  case daemonExitedBeforeReady(String, String?)
  case readinessTimedOut(String)

  var errorDescription: String? {
    switch self {
    case .cannotInitializeProcessSpawn, .cannotConfigureProcessSpawn, .cannotLaunchDaemon:
      return "The bundled server could not be started; retry or reinstall Astronomical"
    case .bundledDaemonUnavailable:
      return "The bundled server executable is unavailable"
    case .unownedDaemonDidNotStop:
      return "The existing server did not stop; quit it and retry"
    case let .existingDaemonNotReady(configurationFileLabel):
      return "A server is running but is not ready with the effective configuration. Check or reveal \(configurationFileLabel), then retry"
    case let .daemonExitedBeforeReady(configurationFileLabel, failureExplanation):
      var description =
        "The server exited before becoming ready. Correct or reveal \(configurationFileLabel), then retry"
      if let failureExplanation, !failureExplanation.isEmpty {
        description += ". \(failureExplanation)"
      }
      return description
    case let .readinessTimedOut(configurationFileLabel):
      return "The server did not become ready in time. Check or reveal \(configurationFileLabel), then retry"
    }
  }
}

/// Detects the daemon's lock-conflict exit in a captured explanation so the
/// restart path can retry a residual race instead of surfacing it.
func lockConflictIsIndicated(by explanation: String?) -> Bool {
  explanation?.contains("already running") == true
}

/// Probes whether the daemon instance flock is free, i.e. no daemon process
/// is alive holding the state directory. Kept as a free function so restart
/// recovery is unit-testable without spawning anything.
func instanceLockIsFree(lockPath: String) -> Bool {
  let lockFileDescriptor = open(lockPath, O_RDWR)
  guard lockFileDescriptor >= 0 else { return true }
  defer { close(lockFileDescriptor) }
  if flock(lockFileDescriptor, LOCK_EX | LOCK_NB) == 0 {
    flock(lockFileDescriptor, LOCK_UN)
    return true
  }
  return false
}

/// Menu-side mirror of the daemon's `last-startup-failure.json` payload.
struct StartupFailureSnapshot: Decodable {
  let stage: String
  let message: String
  let context: String?
}

@MainActor
final class DaemonLifecycleController {
  private let applicationIdentity: ApplicationIdentity
  private let daemonClient: any DaemonClient
  private let ownershipStore: DaemonOwnershipStore
  private let readinessTimeout: Duration
  private let readinessPollInterval: Duration
  private let unownedDaemonStopPollCount: Int
  private let unownedDaemonStopPollInterval: Duration
  private let configuredMenuExecutableURL: URL?
  private let configuredDaemonExecutableURL: URL?
  private let configuredDaemonArguments: [String]?
  private var ownedDaemonProcess: OwnedDaemonProcess?
  private let injectedOwnershipRecordURL: URL?
  private(set) var ownsDaemon = false

  init(
    daemonClient: any DaemonClient,
    applicationIdentity: ApplicationIdentity = .current(),
    readinessTimeout: Duration = .seconds(8),
    readinessPollInterval: Duration = .milliseconds(100),
    unownedDaemonStopPollCount: Int = 30,
    unownedDaemonStopPollInterval: Duration = .milliseconds(100),
    menuExecutableURL: URL? = nil,
    daemonExecutableURL: URL? = nil,
    ownershipRecordURL: URL? = nil,
    daemonArguments: [String]? = nil
  ) {
    self.daemonClient = daemonClient
    self.applicationIdentity = applicationIdentity
    self.readinessTimeout = readinessTimeout
    self.readinessPollInterval = readinessPollInterval
    self.unownedDaemonStopPollCount = unownedDaemonStopPollCount
    self.unownedDaemonStopPollInterval = unownedDaemonStopPollInterval
    configuredMenuExecutableURL = menuExecutableURL
    configuredDaemonExecutableURL = daemonExecutableURL
    configuredDaemonArguments = daemonArguments
    let resolvedOwnershipRecordURL: URL
    if let ownershipRecordURL {
      resolvedOwnershipRecordURL = ownershipRecordURL
    } else {
      resolvedOwnershipRecordURL = applicationIdentity.daemonOwnershipURL()
    }
    ownershipStore = DaemonOwnershipStore(
      ownershipRecordURL: resolvedOwnershipRecordURL
    )
    injectedOwnershipRecordURL = ownershipRecordURL
  }

  func startDaemonIfNeeded() async throws {
    if await daemonClient.healthIsAvailable() {
      try await waitForExistingInstanceReadiness()
      return
    }
    guard let daemonExecutableURL = bundledDaemonExecutableURL(),
      let menuExecutableURL = canonicalMenuExecutableURL()
    else { throw DaemonLifecycleError.bundledDaemonUnavailable }
    recoverStaleOwnershipRecord(
      menuExecutableURL: menuExecutableURL, daemonExecutableURL: daemonExecutableURL)
    if await daemonClient.healthIsAvailable() {
      try await waitForExistingInstanceReadiness()
      return
    }
    try startOwnedDaemon(menuExecutableURL: menuExecutableURL, daemonExecutableURL: daemonExecutableURL)
    try throwIfOwnedDaemonExited()
    try await waitForExpectedInstanceReadiness()
  }

  func restartDaemon() async throws -> String {
    if await daemonClient.healthIsAvailable() {
      try await daemonClient.requestShutdown()
      for _ in 0..<unownedDaemonStopPollCount where await daemonClient.healthIsAvailable() {
        try await Task.sleep(for: unownedDaemonStopPollInterval)
      }
      if await daemonClient.healthIsAvailable(), !ownsDaemon {
        throw DaemonLifecycleError.unownedDaemonDidNotStop
      }
    }
    if ownsDaemon { stopOwnedDaemon() }
    if await daemonClient.healthIsAvailable() {
      throw DaemonLifecycleError.unownedDaemonDidNotStop
    }
    guard let daemonExecutableURL = bundledDaemonExecutableURL(),
      let menuExecutableURL = canonicalMenuExecutableURL()
    else {
      throw DaemonLifecycleError.bundledDaemonUnavailable
    }
    // The previous daemon stops answering /health while it is still tearing
    // down GPU memory; its instance flock survives until process exit. A
    // spawn during that window dies on the lock, so wait for the lock first.
    await waitForInstanceLockRelease(timeout: .seconds(5))
    // A residual race (another supervisor respawning, lock holder discovered
    // late) is retried briefly instead of red-bannering immediately.
    for spawnAttemptIndex in 1...3 {
      do {
        try startOwnedDaemon(menuExecutableURL: menuExecutableURL, daemonExecutableURL: daemonExecutableURL)
        try throwIfOwnedDaemonExited()
        try await waitForExpectedInstanceReadiness()
        return "Server restarted"
      } catch let lifecycleError as DaemonLifecycleError {
        guard case .daemonExitedBeforeReady = lifecycleError,
          spawnAttemptIndex < 3,
          lockConflictIsIndicated(by: capturedStartupFailureExplanation())
        else { throw lifecycleError }
        try? await Task.sleep(for: .milliseconds(300))
      }
    }
    throw DaemonLifecycleError.readinessTimedOut(applicationIdentity.expectedConfigurationFile)
  }

  func stopOwnedDaemon() {
    guard ownsDaemon, let ownedDaemonProcess else { return }
    if !ownedDaemonProcess.reapIfExited() {
      _ = kill(-ownedDaemonProcess.processIdentifier, SIGTERM)
      let shutdownDeadline = ContinuousClock.now + daemonShutdownTimeout
      while ContinuousClock.now < shutdownDeadline, !ownedDaemonProcess.reapIfExited() {
        usleep(25_000)
      }
      if !ownedDaemonProcess.reapIfExited() {
        _ = kill(-ownedDaemonProcess.processIdentifier, SIGKILL)
        ownedDaemonProcess.waitUntilExit()
      }
    }
    clearOwnedDaemonState()
  }

  private func startOwnedDaemon(menuExecutableURL: URL, daemonExecutableURL: URL) throws {
    prepareDaemonStderrCapture()
    let daemonArguments: [String]
    if let configuredDaemonArguments {
      daemonArguments = configuredDaemonArguments
    } else {
      daemonArguments = applicationIdentity.daemonArguments
    }
    let daemonProcess = try launchOwnedDaemonProcess(
      executableURL: daemonExecutableURL,
      arguments: daemonArguments,
      stderrCapturePath: daemonStderrCapturePath
    )
    let ownershipRecord = DaemonOwnershipRecord(
      menuProcessIdentifier: getpid(),
      menuExecutablePath: menuExecutableURL.path,
      daemonProcessIdentifier: daemonProcess.processIdentifier,
      daemonExecutablePath: daemonExecutableURL.path
    )
    do {
      try persistOwnershipRecord(ownershipRecord)
      ownedDaemonProcess = daemonProcess
      ownsDaemon = true
    } catch {
      terminateProcessGroup(processIdentifier: daemonProcess.processIdentifier)
      daemonProcess.waitUntilExit()
      throw error
    }
  }

  private func waitForExpectedInstanceReadiness() async throws {
    let readinessClock = ContinuousClock()
    let readinessDeadline = readinessClock.now.advanced(by: readinessTimeout)
    while readinessClock.now < readinessDeadline {
      // A dead child must win over a health poll. Otherwise a fast exit can be
      // reported as a readiness timeout after SIGTERM reaps a zombie.
      try throwIfOwnedDaemonExited()
      if await daemonClient.expectedInstanceIsHealthy() { return }
      try throwIfOwnedDaemonExited()
      try await Task.sleep(for: readinessPollInterval)
    }
    if await daemonClient.expectedInstanceIsHealthy() { return }
    try throwIfOwnedDaemonExited()
    if reapOwnedDaemonWithin(.milliseconds(50)) {
      try throwIfOwnedDaemonExited()
    }
    await stopUnreadyOwnedDaemon()
    throw DaemonLifecycleError.readinessTimedOut(applicationIdentity.expectedConfigurationFile)
  }

  private func throwIfOwnedDaemonExited() throws {
    guard ownedDaemonProcess?.reapIfExited() == true else { return }
    clearOwnedDaemonState()
    throw DaemonLifecycleError.daemonExitedBeforeReady(
      applicationIdentity.expectedConfigurationFile,
      capturedStartupFailureExplanation())
  }

  private func reapOwnedDaemonWithin(_ gracePeriod: Duration) -> Bool {
    guard let ownedDaemonProcess else { return false }
    if ownedDaemonProcess.reapIfExited() { return true }
    let reapClock = ContinuousClock()
    let reapDeadline = reapClock.now.advanced(by: gracePeriod)
    while reapClock.now < reapDeadline {
      if ownedDaemonProcess.reapIfExited() { return true }
      usleep(1_000)
    }
    return ownedDaemonProcess.reapIfExited()
  }

  private func waitForExistingInstanceReadiness() async throws {
    let readinessClock = ContinuousClock()
    let readinessDeadline = readinessClock.now.advanced(by: readinessTimeout)
    while readinessClock.now < readinessDeadline {
      if await daemonClient.expectedInstanceIsHealthy() { return }
      try await Task.sleep(for: readinessPollInterval)
    }
    if await daemonClient.expectedInstanceIsHealthy() { return }
    throw DaemonLifecycleError.existingDaemonNotReady(
      applicationIdentity.expectedConfigurationFile)
  }

  private func stopUnreadyOwnedDaemon() async {
    guard ownsDaemon, let ownedDaemonProcess else { return }
    _ = kill(-ownedDaemonProcess.processIdentifier, SIGTERM)
    let shutdownClock = ContinuousClock()
    let shutdownDeadline = shutdownClock.now.advanced(by: daemonShutdownTimeout)
    while shutdownClock.now < shutdownDeadline, !ownedDaemonProcess.reapIfExited() {
      try? await Task.sleep(for: .milliseconds(25))
    }
    if !ownedDaemonProcess.reapIfExited() {
      _ = kill(-ownedDaemonProcess.processIdentifier, SIGKILL)
      while !ownedDaemonProcess.reapIfExited() { await Task.yield() }
    }
    clearOwnedDaemonState()
  }

  /// The state directory this controller actually operates on. Tests inject
  /// an ownership record inside a temporary directory; deriving the state
  /// root from it keeps lock probes and stderr captures off the operator's
  /// real home during unit runs.
  private func effectiveStateDirectoryURL() -> URL {
    if let injectedOwnershipRecordURL {
      return injectedOwnershipRecordURL.deletingLastPathComponent()
    }
    return applicationIdentity.stateDirectoryURL()
  }

  private var daemonStderrCapturePath: String {
    effectiveStateDirectoryURL()
      .appendingPathComponent("logs", isDirectory: true)
      .appendingPathComponent("daemon-spawn-stderr.log")
      .path
  }

  private func prepareDaemonStderrCapture() {
    try? FileManager.default.createDirectory(
      at: effectiveStateDirectoryURL().appendingPathComponent("logs", isDirectory: true),
      withIntermediateDirectories: true)
    // Each spawn starts from a clean capture so the tail we replay always
    // belongs to the attempt that just died.
    FileManager.default.createFile(atPath: daemonStderrCapturePath, contents: nil)
  }

  /// Stops the channel daemon at application-quit time: an owned daemon
  /// through the process-group teardown, and a healthy-but-unowned daemon
  /// (started outside this menu process, for example by a development build
  /// script) through a verified single-process termination. Quit is the last
  /// moment anything can stop a daemon the menu bar app was left controlling,
  /// so a daemon whose executable matches this bundle is stopped either way;
  /// a process that is not this bundle's daemon is never signalled.
  func stopChannelDaemonOnApplicationQuit() {
    if ownsDaemon {
      stopOwnedDaemon()
      return
    }
    guard let daemonExecutableURL = bundledDaemonExecutableURL(),
      let lockHolderProcessIdentifier = instanceLockHolderProcessIdentifier(),
      processMatches(
        processIdentifier: lockHolderProcessIdentifier,
        executablePath: daemonExecutableURL.path,
        processGroupIdentifier: nil)
    else { return }
    terminateVerifiedDaemonProcess(processIdentifier: lockHolderProcessIdentifier)
  }

  /// Single-process termination for a daemon this menu process did not spawn.
  /// A script-started daemon shares the spawning script's process group, so a
  /// group kill here could reach unrelated processes; only the verified daemon
  /// pid is signalled. The daemon treats SIGTERM as its graceful shutdown, so
  /// the normal path lets it tear itself and its worker down. If SIGKILL is
  /// needed, the daemon dies before its runtime can reap the worker, so
  /// orphaned worker processes carrying this bundle's worker executable are
  /// reaped afterwards.
  private func terminateVerifiedDaemonProcess(processIdentifier: pid_t) {
    _ = kill(processIdentifier, SIGTERM)
    let shutdownDeadline = ContinuousClock.now + daemonShutdownTimeout
    while ContinuousClock.now < shutdownDeadline, processIsAlive(processIdentifier) {
      usleep(50_000)
    }
    guard processIsAlive(processIdentifier) else { return }
    _ = kill(processIdentifier, SIGKILL)
    let killDeadline = ContinuousClock.now + daemonShutdownTimeout
    while ContinuousClock.now < killDeadline, processIsAlive(processIdentifier) {
      usleep(50_000)
    }
    reapOrphanedWorkerProcesses()
  }

  private func reapOrphanedWorkerProcesses() {
    guard let bundledWorkerExecutablePath else { return }
    for orphanedWorkerProcessIdentifier
    in orphanedProcessIdentifiers(matchingExecutablePath: bundledWorkerExecutablePath) {
      _ = kill(orphanedWorkerProcessIdentifier, SIGKILL)
    }
  }

  private var bundledWorkerExecutablePath: String? {
    guard let daemonExecutableURL = bundledDaemonExecutableURL() else { return nil }
    return daemonExecutableURL.deletingLastPathComponent()
      .appendingPathComponent("astronomical-inference-worker").path
  }

  private func instanceLockHolderProcessIdentifier() -> pid_t? {
    daemonProcessIdentifierFromInstanceLockNote(
      try? String(contentsOf: instanceLockURL, encoding: .utf8))
  }

  private var instanceLockURL: URL {
    effectiveStateDirectoryURL().appendingPathComponent("instance.lock")
  }

  /// Waits until the daemon instance flock is free, i.e. the previous
  /// process fully exited and released the state directory for a respawn.
  private func waitForInstanceLockRelease(timeout: Duration) async {
    let lockDeadline = ContinuousClock().now.advanced(by: timeout)
    while ContinuousClock().now < lockDeadline {
      if instanceLockIsFree(lockPath: instanceLockURL.path) { return }
      try? await Task.sleep(for: .milliseconds(50))
    }
  }

  /// Composes the operator-facing failure story from the two durable homes
  /// the daemon keeps: its captured stderr and the startup-failure record.
  func capturedStartupFailureExplanation() -> String? {
    var explanationFragments: [String] = []
    if let stderrTail = capturedDaemonStderrTail() {
      explanationFragments.append("server said: \(stderrTail)")
    }
    if let failureRecordSummary = lastStartupFailureRecordSummary() {
      explanationFragments.append(failureRecordSummary)
    }
    return explanationFragments.isEmpty ? nil : explanationFragments.joined(separator: " — ")
  }

  private func capturedDaemonStderrTail() -> String? {
    guard let stderrBytes = FileManager.default.contents(atPath: daemonStderrCapturePath),
      !stderrBytes.isEmpty
    else { return nil }
    let significantSuffix = stderrBytes.count > 300
      ? stderrBytes.suffix(300) : stderrBytes[...]
    return String(data: significantSuffix, encoding: .utf8)?
      .trimmingCharacters(in: .whitespacesAndNewlines)
  }

  private func lastStartupFailureRecordSummary() -> String? {
    let recordURL = effectiveStateDirectoryURL()
      .appendingPathComponent("last-startup-failure.json")
    guard let recordBytes = FileManager.default.contents(atPath: recordURL.path),
      let failureRecord = try? JSONDecoder().decode(
        StartupFailureSnapshot.self, from: recordBytes)
    else { return nil }
    var summary = "last start failed [\(failureRecord.stage)]: \(failureRecord.message)"
    if let failureContext = failureRecord.context {
      summary += " (holder: \(failureContext))"
    }
    return summary
  }

  private func clearOwnedDaemonState() {
    ownedDaemonProcess = nil
    ownsDaemon = false
    removeOwnershipRecord()
  }

  private func recoverStaleOwnershipRecord(menuExecutableURL: URL, daemonExecutableURL: URL) {
    guard let ownershipRecord = try? ownershipStore.load() else {
      removeOwnershipRecord()
      return
    }
    guard
      ownershipRecord.matchesExpectedExecutables(
        menuExecutablePath: menuExecutableURL.path,
        daemonExecutablePath: daemonExecutableURL.path
      )
    else {
      removeOwnershipRecord()
      return
    }
    guard
      !processMatches(
        processIdentifier: ownershipRecord.menuProcessIdentifier,
        executablePath: ownershipRecord.menuExecutablePath,
        processGroupIdentifier: nil
      )
    else { return }
    guard
      processMatches(
        processIdentifier: ownershipRecord.daemonProcessIdentifier,
        executablePath: ownershipRecord.daemonExecutablePath,
        processGroupIdentifier: ownershipRecord.daemonProcessIdentifier
      )
    else {
      removeOwnershipRecord()
      return
    }
    terminateProcessGroup(processIdentifier: ownershipRecord.daemonProcessIdentifier)
    removeOwnershipRecord()
  }

  private func processMatches(
    processIdentifier: Int32, executablePath: String, processGroupIdentifier: Int32?
  ) -> Bool {
    let processInspection = Process()
    processInspection.executableURL = URL(fileURLWithPath: "/bin/ps")
    processInspection.arguments = [
      "-p", String(processIdentifier), "-o", "pgid=", "-o", "state=", "-o", "comm=",
    ]
    let standardOutput = Pipe()
    processInspection.standardOutput = standardOutput
    processInspection.standardError = FileHandle.nullDevice
    guard (try? processInspection.run()) != nil else { return false }
    processInspection.waitUntilExit()
    guard processInspection.terminationStatus == 0,
      let processDescription = String(
        data: standardOutput.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)
    else { return false }
    let descriptionFields = processDescription.split(whereSeparator: { $0.isWhitespace })
    // comm is the tail, not the third token: bundle paths like
    // "Astronomical Development.app" contain spaces, so splitting on
    // whitespace breaks the executable path across fields and every
    // Development-channel match would silently fail.
    guard descriptionFields.count >= 3,
      let observedProcessGroupIdentifier = Int32(descriptionFields[0]),
      !descriptionFields[1].hasPrefix("Z")
    else { return false }
    if let processGroupIdentifier, processGroupIdentifier != observedProcessGroupIdentifier {
      return false
    }
    let observedExecutablePath = descriptionFields.dropFirst(2).joined(separator: " ")
    return observedExecutablePath == executablePath
  }

  private func terminateProcessGroup(processIdentifier: Int32) {
    _ = kill(-processIdentifier, SIGTERM)
    let deadline = ContinuousClock.now + daemonShutdownTimeout
    while ContinuousClock.now < deadline, kill(processIdentifier, 0) == 0 { usleep(25_000) }
    if kill(processIdentifier, 0) == 0 { _ = kill(-processIdentifier, SIGKILL) }
  }

  private func persistOwnershipRecord(_ ownershipRecord: DaemonOwnershipRecord) throws {
    try ownershipStore.persist(ownershipRecord)
  }

  private func removeOwnershipRecord() { ownershipStore.remove() }

  private func canonicalMenuExecutableURL() -> URL? {
    if let configuredMenuExecutableURL { return configuredMenuExecutableURL }
    return URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
      .resolvingSymlinksInPath()
  }

  private func bundledDaemonExecutableURL() -> URL? {
    if let configuredDaemonExecutableURL { return configuredDaemonExecutableURL }
    let daemonExecutableURL = URL(fileURLWithPath: CommandLine.arguments[0])
      .deletingLastPathComponent().appendingPathComponent("astronomicald")
      .standardizedFileURL.resolvingSymlinksInPath()
    return FileManager.default.isExecutableFile(atPath: daemonExecutableURL.path)
      ? daemonExecutableURL : nil
  }
}
