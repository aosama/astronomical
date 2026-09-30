import Foundation

/// One entry in the session list the sidebar renders.
public struct SessionFileSummary: Equatable, Sendable {
  public let id: String
  public let title: String
  public let updatedAt: String

  public init(id: String, title: String, updatedAt: String) {
    self.id = id
    self.title = title
    self.updatedAt = updatedAt
  }
}

/// Errors the session file store reports across the bridge.
public enum SessionFileStoreError: Error, Equatable, Sendable {
  /// The page asked for an id the store refuses to address on disk.
  case invalidIdentifier(String)
  /// The stored file is not the JSON object the bridge contract defines.
  case unreadableDocument(String)
}

/// Durable storage for Thin Talk sessions and preferences.
///
/// The page never touches the filesystem (the web view uses a non-persistent
/// data store), so every durable read and write crosses the session bridge into
/// this store. Documents are stored exactly as the page serialized them: the
/// store moves bytes and never interprets the conversation schema, which keeps
/// one schema owner (the web client) instead of two.
///
/// Writes are atomic (temporary file plus rename) so a crash mid-save leaves the
/// previous document intact rather than a truncated one.
public final class SessionFileStore: @unchecked Sendable {
  /// Characters an identifier may contain. Session ids are UUIDs the page
  /// generates, but they arrive over a bridge from a web context, so they are
  /// treated as untrusted input: the allowlist keeps a crafted id from escaping
  /// the sessions directory.
  private static let allowedIdentifierCharacters = Set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")

  private let rootDirectory: URL
  private let fileManager: FileManager
  private let queue = DispatchQueue(label: "astronomical.thintalk.sessionfilestore", qos: .userInitiated)

  /// - Parameters:
  ///   - directory: The channel's thin-talk state directory. Sessions live in
  ///     `sessions/` beneath it and preferences in `preferences.json`.
  ///   - fileManager: Injected for tests.
  public init(directory: URL, fileManager: FileManager = .default) {
    self.rootDirectory = directory
    self.fileManager = fileManager
  }

  public var sessionsDirectory: URL { rootDirectory.appendingPathComponent("sessions", isDirectory: true) }
  public var preferencesFileURL: URL { rootDirectory.appendingPathComponent("preferences.json") }

  // MARK: - Sessions

  /// Lists every stored session, newest first. A file whose header cannot be
  /// read is skipped rather than failing the whole list, so one corrupt file
  /// cannot hide the others.
  public func list() throws -> [SessionFileSummary] {
    try queue.sync { try listSynced() }
  }

  /// The raw document bytes for one session, or nil when no file exists under
  /// that id. The bytes are passed to the page unmodified; the page owns the
  /// document schema and validates it.
  public func load(id: String) throws -> Data? {
    try queue.sync {
      guard Self.isValidIdentifier(id) else { throw SessionFileStoreError.invalidIdentifier(id) }
      let fileURL = sessionFileURL(id: id)
      guard fileManager.fileExists(atPath: fileURL.path) else { return nil }
      return try Data(contentsOf: fileURL)
    }
  }

  /// Stores one session document atomically.
  public func save(id: String, documentData: Data) throws {
    try queue.sync {
      guard Self.isValidIdentifier(id) else { throw SessionFileStoreError.invalidIdentifier(id) }
      try createDirectoriesIfNeeded()
      let fileURL = sessionFileURL(id: id)
      try documentData.write(to: fileURL, options: [.atomic])
    }
  }

  public func delete(id: String) throws {
    try queue.sync {
      guard Self.isValidIdentifier(id) else { throw SessionFileStoreError.invalidIdentifier(id) }
      let fileURL = sessionFileURL(id: id)
      guard fileManager.fileExists(atPath: fileURL.path) else { return }
      try fileManager.removeItem(at: fileURL)
    }
  }

  /// Rewrites only the title of one stored session, leaving the rest of the
  /// document bytes untouched.
  public func rename(id: String, title: String) throws {
    try queue.sync {
      guard Self.isValidIdentifier(id) else { throw SessionFileStoreError.invalidIdentifier(id) }
      let fileURL = sessionFileURL(id: id)
      let rawDocument = try JSONSerialization.jsonObject(with: Data(contentsOf: fileURL))
      guard var document = rawDocument as? [String: Any] else {
        throw SessionFileStoreError.unreadableDocument(id)
      }
      document["title"] = title
      let renamedData = try JSONSerialization.data(withJSONObject: document)
      try renamedData.write(to: fileURL, options: [.atomic])
    }
  }

  // MARK: - Preferences

  public func loadPreferences() throws -> Data? {
    try queue.sync {
      guard fileManager.fileExists(atPath: preferencesFileURL.path) else { return nil }
      return try Data(contentsOf: preferencesFileURL)
    }
  }

  public func savePreferences(_ preferencesData: Data) throws {
    try queue.sync {
      try createDirectoriesIfNeeded()
      try preferencesData.write(to: preferencesFileURL, options: [.atomic])
    }
  }

  // MARK: - Internals

  private func listSynced() throws -> [SessionFileSummary] {
    try createDirectoriesIfNeeded()
    let contents = try fileManager.contentsOfDirectory(at: sessionsDirectory, includingPropertiesForKeys: [.contentModificationDateKey])
    let sessionFiles = contents.filter { $0.pathExtension == "json" }
    let summaries = sessionFiles.compactMap { fileURL -> SessionFileSummary? in
      guard let header = try? readHeader(of: fileURL) else { return nil }
      return header
    }
    return summaries.sorted { $0.updatedAt > $1.updatedAt }
  }

  /// Reads the three fields the sidebar needs without loading the transcript.
  private func readHeader(of fileURL: URL) throws -> SessionFileSummary? {
    let rawDocument = try JSONSerialization.jsonObject(with: Data(contentsOf: fileURL))
    guard let document = rawDocument as? [String: Any] else {
      throw SessionFileStoreError.unreadableDocument(fileURL.lastPathComponent)
    }
    guard let id = document["id"] as? String, !id.isEmpty else { return nil }
    let title = document["title"] as? String ?? ""
    let updatedAt = document["updatedAt"] as? String ?? ""
    return SessionFileSummary(id: id, title: title, updatedAt: updatedAt)
  }

  private func sessionFileURL(id: String) -> URL {
    sessionsDirectory.appendingPathComponent("\(id).json")
  }

  private func createDirectoriesIfNeeded() throws {
    if !fileManager.fileExists(atPath: sessionsDirectory.path) {
      try fileManager.createDirectory(at: sessionsDirectory, withIntermediateDirectories: true)
    }
  }

  static func isValidIdentifier(_ id: String) -> Bool {
    !id.isEmpty && id.count <= 64 && id.allSatisfy { Self.allowedIdentifierCharacters.contains($0) }
  }
}
