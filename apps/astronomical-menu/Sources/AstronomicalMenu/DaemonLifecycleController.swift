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

  init(processIdentifier: pid_t) { self.processIdentifier = processIdentifier }

  func waitUntilExit() {
    var childProcessStatus: Int32 = 0
    while waitpid(processIdentifier, &childProcessStatus, 0) == -1 && errno == EINTR {}
  }
}

func launchOwnedDaemonProcess(executableURL: URL, arguments: [String] = []) throws
  -> OwnedDaemonProcess
{
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
  guard posix_spawnattr_setflags(&spawnAttributes, Int16(POSIX_SPAWN_SETPGROUP)) == 0,
    posix_spawnattr_setpgroup(&spawnAttributes, 0) == 0,
    posix_spawn_file_actions_addopen(&spawnFileActions, STDIN_FILENO, "/dev/null", O_RDONLY, 0)
      == 0,
    posix_spawn_file_actions_addopen(&spawnFileActions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0)
      == 0
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

  var errorDescription: String? {
    switch self {
    case .cannotInitializeProcessSpawn, .cannotConfigureProcessSpawn, .cannotLaunchDaemon:
      "The bundled server could not be started"
    case .bundledDaemonUnavailable:
      "The bundled server executable is unavailable"
    case .unownedDaemonDidNotStop:
      "The existing server did not stop; quit it and retry"
    }
  }
}

@MainActor
final class DaemonLifecycleController {
  private let applicationIdentity: ApplicationIdentity
  private let supervisorClient: any SupervisorClient
  private let ownershipStore: DaemonOwnershipStore
  private var ownedDaemonProcess: OwnedDaemonProcess?
  private(set) var ownsDaemon = false

  init(
    supervisorClient: any SupervisorClient,
    applicationIdentity: ApplicationIdentity = .current()
  ) {
    self.supervisorClient = supervisorClient
    self.applicationIdentity = applicationIdentity
    ownershipStore = DaemonOwnershipStore(
      ownershipRecordURL: applicationIdentity.daemonOwnershipURL()
    )
  }

  func startDaemonIfNeeded() async {
    guard !(await supervisorClient.healthIsAvailable()),
      let daemonExecutableURL = bundledDaemonExecutableURL(),
      let menuExecutableURL = canonicalMenuExecutableURL()
    else { return }
    recoverStaleOwnershipRecord(
      menuExecutableURL: menuExecutableURL, daemonExecutableURL: daemonExecutableURL)
    guard !(await supervisorClient.healthIsAvailable()) else { return }
    try? startOwnedDaemon(menuExecutableURL: menuExecutableURL, daemonExecutableURL: daemonExecutableURL)
  }

  func restartDaemon() async throws -> String {
    if await supervisorClient.healthIsAvailable() {
      try await supervisorClient.requestShutdown()
      for _ in 0..<30 where await supervisorClient.healthIsAvailable() {
        try? await Task.sleep(for: .milliseconds(100))
      }
      if await supervisorClient.healthIsAvailable(), !ownsDaemon {
        throw DaemonLifecycleError.unownedDaemonDidNotStop
      }
    }
    if ownsDaemon { stopOwnedDaemon() }
    guard let daemonExecutableURL = bundledDaemonExecutableURL(),
      let menuExecutableURL = canonicalMenuExecutableURL()
    else {
      throw DaemonLifecycleError.bundledDaemonUnavailable
    }
    try startOwnedDaemon(menuExecutableURL: menuExecutableURL, daemonExecutableURL: daemonExecutableURL)
    return "Server restarted"
  }

  func stopOwnedDaemon() {
    guard ownsDaemon, let ownedDaemonProcess else { return }
    terminateProcessGroup(processIdentifier: ownedDaemonProcess.processIdentifier)
    ownedDaemonProcess.waitUntilExit()
    self.ownedDaemonProcess = nil
    ownsDaemon = false
    removeOwnershipRecord()
  }

  private func startOwnedDaemon(menuExecutableURL: URL, daemonExecutableURL: URL) throws {
    let daemonProcess = try launchOwnedDaemonProcess(
      executableURL: daemonExecutableURL,
      arguments: applicationIdentity.daemonArguments
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
    guard descriptionFields.count >= 3,
      let observedProcessGroupIdentifier = Int32(descriptionFields[0]),
      !descriptionFields[1].hasPrefix("Z")
    else { return false }
    if let processGroupIdentifier, processGroupIdentifier != observedProcessGroupIdentifier {
      return false
    }
    return String(descriptionFields[2]) == executablePath
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
    URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL.resolvingSymlinksInPath()
  }

  private func bundledDaemonExecutableURL() -> URL? {
    let daemonExecutableURL = URL(fileURLWithPath: CommandLine.arguments[0])
      .deletingLastPathComponent().appendingPathComponent("astronomicald")
      .standardizedFileURL.resolvingSymlinksInPath()
    return FileManager.default.isExecutableFile(atPath: daemonExecutableURL.path)
      ? daemonExecutableURL : nil
  }
}
