import Foundation
import ThinTalkCore
import XCTest

/// The session file store is the only durable state Thin Talk owns, so its
/// contract is exercised directly: atomic writes, id validation, corrupt-file
/// tolerance in the list, and preferences round-tripping.
final class SessionFileStoreTests: XCTestCase {
  private var storeDirectory: URL!

  override func setUpWithError() throws {
    try super.setUpWithError()
    storeDirectory = FileManager.default.temporaryDirectory
      .appendingPathComponent("thintalk-store-tests-\(UUID().uuidString)", isDirectory: true)
  }

  override func tearDownWithError() throws {
    if let storeDirectory {
      try? FileManager.default.removeItem(at: storeDirectory)
    }
    storeDirectory = nil
    try super.tearDownWithError()
  }

  private func makeStore() -> SessionFileStore {
    SessionFileStore(directory: storeDirectory)
  }

  private func documentData(id: String, title: String, updatedAt: String) -> Data {
    let document: [String: Any] = [
      "schemaVersion": 1,
      "id": id,
      "title": title,
      "createdAt": "2026-01-01T00:00:00Z",
      "updatedAt": updatedAt,
      "messages": [
        ["id": "m1", "role": "user", "content": "hello", "reasoning": "", "state": "complete"]
      ],
    ]
    return try! JSONSerialization.data(withJSONObject: document)
  }

  func test_save_then_load_round_trips_document_bytes() throws {
    let store = makeStore()
    let document = documentData(id: "session-1", title: "Romeo", updatedAt: "2026-01-02T03:04:05Z")
    try store.save(id: "session-1", documentData: document)

    let loaded = try XCTUnwrap(try store.load(id: "session-1"))
    XCTAssertEqual(loaded, document)
  }

  func test_load_returns_nil_for_unknown_session() throws {
    let store = makeStore()
    XCTAssertNil(try store.load(id: "missing"))
  }

  func test_list_returns_summaries_newest_first() throws {
    let store = makeStore()
    try store.save(id: "older", documentData: documentData(id: "older", title: "Older", updatedAt: "2026-01-01T00:00:00Z"))
    try store.save(id: "newer", documentData: documentData(id: "newer", title: "Newer", updatedAt: "2026-01-02T00:00:00Z"))

    let summaries = try store.list()
    XCTAssertEqual(summaries.map(\.id), ["newer", "older"])
    XCTAssertEqual(summaries.map(\.title), ["Newer", "Older"])
  }

  func test_list_skips_files_without_a_readable_header() throws {
    let store = makeStore()
    try store.save(id: "good", documentData: documentData(id: "good", title: "Good", updatedAt: "2026-01-01T00:00:00Z"))
    try FileManager.default.createDirectory(at: store.sessionsDirectory, withIntermediateDirectories: true)
    try Data("not json".utf8).write(to: store.sessionsDirectory.appendingPathComponent("corrupt.json"))

    let summaries = try store.list()
    XCTAssertEqual(summaries.map(\.id), ["good"])
  }

  func test_rename_updates_only_the_title() throws {
    let store = makeStore()
    let original = documentData(id: "session-1", title: "Before", updatedAt: "2026-01-02T03:04:05Z")
    try store.save(id: "session-1", documentData: original)

    try store.rename(id: "session-1", title: "After")

    let renamed = try JSONSerialization.jsonObject(with: XCTUnwrap(try store.load(id: "session-1"))) as? [String: Any]
    XCTAssertEqual(renamed?["title"] as? String, "After")
    XCTAssertEqual(renamed?["updatedAt"] as? String, "2026-01-02T03:04:05Z")
    XCTAssertEqual((renamed?["messages"] as? [[String: Any]])?.count, 1)
  }

  func test_delete_removes_the_document() throws {
    let store = makeStore()
    try store.save(id: "session-1", documentData: documentData(id: "session-1", title: "t", updatedAt: "u"))
    try store.delete(id: "session-1")
    XCTAssertNil(try store.load(id: "session-1"))
  }

  func test_delete_of_unknown_session_is_a_no_op() throws {
    let store = makeStore()
    XCTAssertNoThrow(try store.delete(id: "missing"))
  }

  func test_identifiers_cannot_escape_the_sessions_directory() throws {
    let store = makeStore()
    for hostile in ["", "../escape", "a/b", "a b", "a.b", String(repeating: "x", count: 65)] {
      XCTAssertThrowsError(try store.save(id: hostile, documentData: Data())) { error in
        guard case SessionFileStoreError.invalidIdentifier(let rejected) = error else {
          return XCTFail("expected invalidIdentifier for \(hostile)")
        }
        XCTAssertEqual(rejected, hostile)
      }
    }
  }

  func test_preferences_round_trip() throws {
    let store = makeStore()
    XCTAssertNil(try store.loadPreferences())

    let preferences = Data(#"{"thinkingEffort":"balanced","composerFontSize":16}"#.utf8)
    try store.savePreferences(preferences)
    XCTAssertEqual(try store.loadPreferences(), preferences)

    let updated = Data(#"{"thinkingEffort":"high","composerFontSize":12}"#.utf8)
    try store.savePreferences(updated)
    XCTAssertEqual(try store.loadPreferences(), updated)
  }
}
