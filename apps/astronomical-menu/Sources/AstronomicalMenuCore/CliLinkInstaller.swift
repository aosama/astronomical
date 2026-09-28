import Foundation
import os

/// Installs, repairs, and removes the `astronomical` terminal command symlink.
///
/// The link lives at `/usr/local/bin/astronomical` and points into this app
/// bundle, so every app update replaces the CLI atomically with the daemon it
/// talks to. Creating or replacing that link needs root because the directory
/// is root-owned on Apple Silicon; the privileged step runs through an
/// AppleScript administrator prompt, once per action, with nothing piped
/// through the shell from untrusted input — both paths are constants joined
/// from this bundle's own location.
@MainActor
struct CliLinkInstaller {
  private let channel: ApplicationChannel
  private let bundleCliURL: URL?
  private let destinationDirectory = URL(fileURLWithPath: "/usr/local/bin")
  private let logger = Logger(subsystem: "dev.astronomical.app", category: "cli-link")

  init(channel: ApplicationChannel, bundle: Bundle = .main) {
    self.channel = channel
    // The menu binary and the CLI live in the same Contents/MacOS directory.
    self.bundleCliURL = bundle.executableURL?.deletingLastPathComponent()
      .appendingPathComponent("astronomical")
  }

  var destinationURL: URL {
    guard let linkName = CliLinkPlan.linkName(for: channel) else {
      // Unreachable for supported channels; the plan refuses development.
      return destinationDirectory.appendingPathComponent("astronomical")
    }
    return destinationDirectory.appendingPathComponent(linkName)
  }

  var isSupportedForChannel: Bool {
    CliLinkPlan.linkName(for: channel) != nil
  }

  func installTerminalCommand() async throws {
    try await apply()
  }

  func removeTerminalCommand() async throws {
    guard isSupportedForChannel else { return }
    guard let bundleCliURL else { throw CliLinkError.bundleCliUnavailable }
    let status = CliLinkInstaller.observedStatus(at: destinationURL)
    let action = CliLinkPlan.decide(
      status: status, channel: channel, bundleCliPath: bundleCliURL.path)
    guard case .alreadyCurrent = action else {
      // Uninstall must remove only the link this app created (contract #3):
      // a missing link is already gone, and a foreign destination is not ours.
      if !status.exists { return }
      throw CliLinkError.foreignDestination
    }
    try await runPrivilegedRemoval(destination: destinationURL)
  }

  /// Full auto-repair (owner decision on issue #825): on every launch, make
  /// the terminal command current for this bundle without asking. A missing
  /// link is only created by the explicit install action; repair covers a
  /// link this app owns that dangles after the bundle moved.
  func repairOnLaunchIfNeeded() async {
    guard isSupportedForChannel, let bundleCliURL else { return }
    let status = CliLinkInstaller.observedStatus(at: destinationURL)
    let action = CliLinkPlan.decide(
      status: status, channel: channel, bundleCliPath: bundleCliURL.path)
    switch action {
    case .alreadyCurrent, .installMissing, .refuseForeignDestination, .notSupportedForChannel:
      return
    case .repairOwnedDanglingLink:
      do {
        try await recreateOwnedLink(bundleCliURL: bundleCliURL)
        logger.info("repaired terminal command link")
      } catch {
        // Repair is best-effort: a declined admin prompt must never block
        // the application from launching.
        logger.error("terminal command repair failed: \(error.localizedDescription)")
      }
    }
  }

  private func apply() async throws {
    guard isSupportedForChannel else { throw CliLinkError.unsupportedChannel }
    guard let bundleCliURL else { throw CliLinkError.bundleCliUnavailable }
    let status = CliLinkInstaller.observedStatus(at: destinationURL)
    let action = CliLinkPlan.decide(
      status: status, channel: channel, bundleCliPath: bundleCliURL.path)
    switch action {
    case .alreadyCurrent:
      return
    case .installMissing, .repairOwnedDanglingLink:
      try await recreateOwnedLink(bundleCliURL: bundleCliURL)
    case .refuseForeignDestination:
      throw CliLinkError.foreignDestination
    case .notSupportedForChannel:
      throw CliLinkError.unsupportedChannel
    }
  }

  private func recreateOwnedLink(bundleCliURL: URL) async throws {
    try await runPrivilegedLinkCreation(
      target: bundleCliURL.path, destination: destinationURL.path)
  }

  private func runPrivilegedLinkCreation(target: String, destination: String) async throws {
    let script =
      "do shell script \"ln -sfn '\(escaped(target))' '\(escaped(destination))'\" "
      + "with administrator privileges"
    try await runAppleScript(script)
  }

  private func runPrivilegedRemoval(destination: URL) async throws {
    let script =
      "do shell script \"rm -f '\(escaped(destination.path))'\" "
      + "with administrator privileges"
    try await runAppleScript(script)
  }

  // Single quotes break out of the sh single-quoted context, so they are the
  // one character that must be neutralized before the paths reach the shell.
  private func escaped(_ path: String) -> String {
    path.replacingOccurrences(of: "'", with: "'\\''")
  }

  private func runAppleScript(_ source: String) async throws {
    try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
      let process = Process()
      process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
      process.arguments = ["-e", source]
      let pipe = Pipe()
      process.standardOutput = pipe
      process.standardError = pipe
      process.terminationHandler = { osascript in
        if osascript.terminationStatus == 0 {
          continuation.resume()
        } else {
          let failureOutput = String(
            data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)
          logger.error("osascript failed: \(failureOutput ?? "", privacy: .public)")
          continuation.resume(throwing: CliLinkError.privilegedExecutionUnavailable)
        }
      }
      do {
        try process.run()
      } catch {
        continuation.resume(throwing: CliLinkError.privilegedExecutionUnavailable)
      }
    }
  }

  private static func observedStatus(at url: URL) -> CliLinkStatus {
    let fileManager = FileManager.default
    var isSymlink: ObjCBool = false
    let exists = fileManager.fileExists(atPath: url.path, isDirectory: &isSymlink)
    guard exists else {
      return CliLinkStatus(
        exists: false, isSymlink: false, symlinkDestinationPath: nil,
        symlinkDestinationExists: false)
    }
    guard (try? fileManager.destinationOfSymbolicLink(atPath: url.path)) != nil else {
      return CliLinkStatus(
        exists: true, isSymlink: false, symlinkDestinationPath: nil,
        symlinkDestinationExists: false)
    }
    let destinationPath = (try? fileManager.destinationOfSymbolicLink(atPath: url.path))
    let destinationExists = destinationPath.map {
      fileManager.fileExists(atPath: $0)
    } ?? false
    return CliLinkStatus(
      exists: true, isSymlink: true, symlinkDestinationPath: destinationPath,
      symlinkDestinationExists: destinationExists)
  }
}

enum CliLinkError: LocalizedError {
  case bundleCliUnavailable
  case foreignDestination
  case unsupportedChannel
  case privilegedExecutionUnavailable

  var errorDescription: String? {
    switch self {
    case .bundleCliUnavailable:
      return "The astronomical command-line tool could not be located in the app bundle"
    case .foreignDestination:
      return "A file this app did not create already exists at the destination path"
    case .unsupportedChannel:
      return "This Astronomical channel does not own the terminal command"
    case .privilegedExecutionUnavailable:
      return "Administrator approval could not complete the terminal command change"
    }
  }
}
