import Darwin
import Foundation

/// Parses the daemon's best-effort instance lock note
/// (`pid=N instance=astronomicald`). The flock is the real lock; the pid note
/// only names the holder for diagnostics, so callers must verify process
/// liveness and executable identity before acting on the parsed value.
func daemonProcessIdentifierFromInstanceLockNote(_ lockNote: String?) -> pid_t? {
  guard let lockNote else { return nil }
  for lockNoteField in lockNote.split(separator: " ") {
    let noteKeyValue = lockNoteField.split(separator: "=", maxSplits: 1)
    if noteKeyValue.count == 2, noteKeyValue[0] == "pid",
      let holderProcessIdentifier = Int32(noteKeyValue[1])
    {
      return holderProcessIdentifier
    }
  }
  return nil
}

/// Matches a process by identifier, executable, and optional process group
/// using ps output. A zombie never matches: it is dead for every
/// termination decision even though `kill(pid, 0)` still succeeds.
func processMatches(
  processIdentifier: Int32, executablePath: String, processGroupIdentifier: Int32?
) -> Bool {
  let processDescriptionFields = psOutputFields(
    arguments: ["-p", String(processIdentifier), "-o", "pgid=", "-o", "state=", "-o", "comm="])
  // comm is the tail, not the third token: bundle paths like
  // "Astronomical Development.app" contain spaces, so splitting on
  // whitespace breaks the executable path across fields and every
  // Development-channel match would silently fail.
  guard processDescriptionFields.count >= 3,
    let observedProcessGroupIdentifier = Int32(processDescriptionFields[0]),
    !processDescriptionFields[1].hasPrefix("Z")
  else { return false }
  if let processGroupIdentifier, processGroupIdentifier != observedProcessGroupIdentifier {
    return false
  }
  let observedExecutablePath = processDescriptionFields.dropFirst(2).joined(separator: " ")
  return observedExecutablePath == executablePath
}

/// Liveness for a process the caller cannot `waitpid` on. A zombie is dead
/// for termination purposes — no signal changes its state — so quit must not
/// wait on one; treating it as alive would hang on a parent that never reaps.
func processIsAlive(_ processIdentifier: Int32) -> Bool {
  guard kill(processIdentifier, 0) == 0 else { return false }
  let processStateFields = psOutputFields(
    arguments: ["-p", String(processIdentifier), "-o", "state="])
  guard let observedProcessState = processStateFields.first else { return true }
  return !observedProcessState.hasPrefix("Z")
}

/// Processes whose executable matches the given path and that have been
/// reparented to launchd: after a daemon pid is confirmed dead, a worker
/// still alive with parent pid 1 lost its supervisor mid-shutdown.
func orphanedProcessIdentifiers(matchingExecutablePath executablePath: String) -> [pid_t] {
  let inventoryFieldsByLine = psOutputFieldsByLine(arguments: ["-axo", "pid=,ppid=,comm="])
  var orphanedProcessIdentifiers: [pid_t] = []
  for inventoryFields in inventoryFieldsByLine {
    // comm is the tail, joined back across its spaces: worker paths inside
    // "Astronomical Development.app" split into several whitespace fields.
    guard inventoryFields.count >= 3,
      let candidateProcessIdentifier = Int32(inventoryFields[0]),
      let candidateParentProcessIdentifier = Int32(inventoryFields[1]),
      candidateParentProcessIdentifier == 1,
      String(inventoryFields.dropFirst(2).joined(separator: " ")) == executablePath
    else { continue }
    orphanedProcessIdentifiers.append(candidateProcessIdentifier)
  }
  return orphanedProcessIdentifiers
}

private func psOutputFields(arguments: [String]) -> [Substring] {
  if let outputFields = psOutputFieldsByLine(arguments: arguments).first {
    return outputFields
  }
  return []
}

private func psOutputFieldsByLine(arguments: [String]) -> [[Substring]] {
  let processInspection = Process()
  processInspection.executableURL = URL(fileURLWithPath: "/bin/ps")
  processInspection.arguments = arguments
  let standardOutput = Pipe()
  processInspection.standardOutput = standardOutput
  processInspection.standardError = FileHandle.nullDevice
  guard (try? processInspection.run()) != nil else { return [] }
  processInspection.waitUntilExit()
  guard processInspection.terminationStatus == 0,
    let inspectionOutput = String(
      data: standardOutput.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)
  else { return [] }
  return inspectionOutput
    .split(separator: "\n")
    .map { $0.split(whereSeparator: { $0.isWhitespace }) }
}
