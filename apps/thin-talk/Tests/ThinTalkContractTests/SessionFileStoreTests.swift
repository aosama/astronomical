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

  func test_should_load_the_same_document_bytes_it_saved() throws {
    let store = makeStore()
    let document = documentData(id: "session-1", title: "Romeo", updatedAt: "2026-01-02T03:04:05Z")
    try store.save(id: "session-1", documentData: document)

    let loaded = try XCTUnwrap(try store.load(id: "session-1"))
    XCTAssertEqual(loaded, document)
  }

  func test_should_load_nothing_for_an_unknown_session() throws {
    let store = makeStore()
    XCTAssertNil(try store.load(id: "missing"))
  }

  func test_should_list_summaries_newest_first() throws {
    let store = makeStore()
    try store.save(id: "older", documentData: documentData(id: "older", title: "Older", updatedAt: "2026-01-01T00:00:00Z"))
    try store.save(id: "newer", documentData: documentData(id: "newer", title: "Newer", updatedAt: "2026-01-02T00:00:00Z"))

    let summaries = try store.list()
    XCTAssertEqual(summaries.map(\.id), ["newer", "older"])
    XCTAssertEqual(summaries.map(\.title), ["Newer", "Older"])
  }

  func test_should_skip_files_without_a_readable_header_when_listing() throws {
    let store = makeStore()
    try store.save(id: "good", documentData: documentData(id: "good", title: "Good", updatedAt: "2026-01-01T00:00:00Z"))
    try FileManager.default.createDirectory(at: store.sessionsDirectory, withIntermediateDirectories: true)
    try Data("not json".utf8).write(to: store.sessionsDirectory.appendingPathComponent("corrupt.json"))

    let summaries = try store.list()
    XCTAssertEqual(summaries.map(\.id), ["good"])
  }

  func test_should_change_only_the_title_on_rename() throws {
    let store = makeStore()
    let original = documentData(id: "session-1", title: "Before", updatedAt: "2026-01-02T03:04:05Z")
    try store.save(id: "session-1", documentData: original)

    try store.rename(id: "session-1", title: "After")

    let renamed = try JSONSerialization.jsonObject(with: XCTUnwrap(try store.load(id: "session-1"))) as? [String: Any]
    XCTAssertEqual(renamed?["title"] as? String, "After")
    XCTAssertEqual(renamed?["updatedAt"] as? String, "2026-01-02T03:04:05Z")
    XCTAssertEqual((renamed?["messages"] as? [[String: Any]])?.count, 1)
  }

  func test_should_remove_the_document_on_delete() throws {
    let store = makeStore()
    try store.save(id: "session-1", documentData: documentData(id: "session-1", title: "t", updatedAt: "u"))
    try store.delete(id: "session-1")
    XCTAssertNil(try store.load(id: "session-1"))
  }

  func test_should_do_nothing_when_deleting_an_unknown_session() throws {
    let store = makeStore()
    XCTAssertNoThrow(try store.delete(id: "missing"))
  }

  func test_should_refuse_identifiers_that_escape_the_sessions_directory() throws {
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

  func test_should_load_the_preferences_it_saved() throws {
    let store = makeStore()
    XCTAssertNil(try store.loadPreferences())

    let preferences = Data(#"{"modelID":"model-alpha","sidebarWidthPixels":300}"#.utf8)
    try store.savePreferences(preferences)
    XCTAssertEqual(try store.loadPreferences(), preferences)

    let updated = Data(#"{"modelID":"model-beta","sidebarWidthPixels":260}"#.utf8)
    try store.savePreferences(updated)
    XCTAssertEqual(try store.loadPreferences(), updated)
  }

  func test_should_pass_a_chats_saved_skill_through_the_list_and_nil_for_older_files() throws {
    let store = makeStore()
    try store.save(id: "drawn", documentData: Data(#"{"id":"drawn","title":"Moon","updatedAt":"2026-11-08T10:00:00Z","skill":"pictures","messages":[]}"#.utf8))
    try store.save(id: "older", documentData: Data(#"{"id":"older","title":"Hi","updatedAt":"2026-11-07T10:00:00Z","messages":[]}"#.utf8))
    try store.save(id: "odd", documentData: Data(#"{"id":"odd","title":"Odd","updatedAt":"2026-11-06T10:00:00Z","skill":7,"messages":[]}"#.utf8))

    let skillByID = Dictionary(uniqueKeysWithValues: try store.list().map { ($0.id, $0.skill) })
    XCTAssertEqual(skillByID["drawn"], "pictures")
    XCTAssertEqual(skillByID["older"], .some(nil), "a file saved before skills were recorded lists without one")
    XCTAssertEqual(skillByID["odd"], .some(nil), "a skill that is not text is ignored")
  }

  func test_should_return_nil_before_astro_has_a_profile_and_round_trip_it_after() throws {
    let store = makeStore()
    XCTAssertNil(try store.loadAstroProfile())

    let firstProfile = Data(#"{"nickname":"Maya","style":"explorer","hasFinishedFirstFlight":true}"#.utf8)
    try store.saveAstroProfile(firstProfile)
    XCTAssertEqual(try store.loadAstroProfile(), firstProfile)

    let changedProfile = Data(#"{"nickname":"","style":"pilot","hasFinishedFirstFlight":true}"#.utf8)
    try store.saveAstroProfile(changedProfile)
    XCTAssertEqual(try store.loadAstroProfile(), changedProfile)
  }

  func test_should_keep_astro_profile_and_chat_preferences_in_separate_files() throws {
    let store = makeStore()
    let preferences = Data(#"{"modelID":"model-alpha"}"#.utf8)
    let astroProfile = Data(#"{"style":"pilot"}"#.utf8)
    try store.savePreferences(preferences)
    try store.saveAstroProfile(astroProfile)

    XCTAssertNotEqual(store.preferencesFileURL, store.astroProfileFileURL)
    XCTAssertEqual(try store.loadPreferences(), preferences, "saving Astro's profile must not touch chat preferences")
    XCTAssertEqual(try store.loadAstroProfile(), astroProfile)
  }
}
