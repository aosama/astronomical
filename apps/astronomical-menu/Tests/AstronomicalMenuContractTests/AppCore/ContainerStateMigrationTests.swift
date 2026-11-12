import Foundation
import XCTest

@testable import AstronomicalAppCore

/// An App Store build is two sandboxed applications. These tests pin where
/// its state lives (the group container both can open), what the engine is
/// told, and the one-time move of state an earlier single-application build
/// left in the main app's own container.
@MainActor
final class ContainerStateMigrationTests: XCTestCase {
  private var scratchDirectoryURL: URL!
  private var homeDirectoryURL: URL { scratchDirectoryURL.appendingPathComponent("home", isDirectory: true) }
  private var groupContainerURL: URL {
    scratchDirectoryURL.appendingPathComponent("Group Containers/TEAM123456.astronomical", isDirectory: true)
  }

  override func setUp() async throws {
    try await super.setUp()
    scratchDirectoryURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("astronomical-container-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: homeDirectoryURL, withIntermediateDirectories: true)
  }

  override func tearDown() async throws {
    try? FileManager.default.removeItem(at: scratchDirectoryURL)
    try await super.tearDown()
  }

  private func storeIdentity(applicationGroupIdentifier: String?) -> ApplicationIdentity {
    ApplicationIdentity(
      channel: .appStore, daemonPort: 6732, stateDirectoryName: nil, version: "1.0.0", buildNumber: "7",
      commit: "abc", isDirty: false, applicationGroupIdentifier: applicationGroupIdentifier)
  }

  private func resolver() -> ApplicationIdentity.GroupContainerResolver {
    let resolvedGroupContainerURL = groupContainerURL
    return { _ in resolvedGroupContainerURL }
  }

  private func writeLegacyState() throws -> URL {
    let legacyStateDirectoryURL = storeIdentity(applicationGroupIdentifier: nil)
      .legacyContainerStateDirectoryURL(homeDirectoryURL: homeDirectoryURL)
    try FileManager.default.createDirectory(
      at: legacyStateDirectoryURL.appendingPathComponent("models"), withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: legacyStateDirectoryURL.appendingPathComponent("config.json"))
    return legacyStateDirectoryURL
  }

  func test_should_keep_store_state_in_the_group_container_when_the_build_has_a_group() {
    let groupedIdentity = storeIdentity(applicationGroupIdentifier: "TEAM123456.astronomical")

    XCTAssertEqual(
      groupedIdentity.stateDirectoryURL(homeDirectoryURL: homeDirectoryURL, groupContainerURL: resolver()).path,
      groupContainerURL.appendingPathComponent("Astronomical").path)
    XCTAssertEqual(
      groupedIdentity.expectedServerStateDirectory,
      "~/Library/Group Containers/TEAM123456.astronomical/Astronomical")
  }

  func test_should_keep_application_support_for_a_store_build_without_a_group() {
    let ungroupedIdentity = storeIdentity(applicationGroupIdentifier: nil)

    XCTAssertEqual(
      ungroupedIdentity.stateDirectoryURL(homeDirectoryURL: homeDirectoryURL, groupContainerURL: resolver()).path,
      homeDirectoryURL.appendingPathComponent("Library/Application Support/Astronomical").path)
    XCTAssertNil(ungroupedIdentity.applicationGroupContainerURL(groupContainerURL: resolver()))
  }

  /// A direct build is not sandboxed: a group identifier that reached its
  /// plist by mistake must not move its state out of the home dot-folder.
  func test_should_ignore_a_group_on_a_direct_channel_build() {
    let directIdentity = ApplicationIdentity(
      channel: .stable, daemonPort: 6732, stateDirectoryName: ".astronomical", version: "1.0.0",
      buildNumber: "7", commit: "abc", isDirty: false, applicationGroupIdentifier: "TEAM123456.astronomical")

    XCTAssertEqual(
      directIdentity.stateDirectoryURL(homeDirectoryURL: homeDirectoryURL, groupContainerURL: resolver()).path,
      homeDirectoryURL.appendingPathComponent(".astronomical").path)
    XCTAssertEqual(directIdentity.daemonArguments(groupContainerURL: resolver()), ["--instance", "stable"])
  }

  /// Only this side can resolve the container, so the engine is told it.
  func test_should_tell_the_engine_the_group_container_only_when_there_is_one() {
    XCTAssertEqual(
      storeIdentity(applicationGroupIdentifier: "TEAM123456.astronomical")
        .daemonArguments(groupContainerURL: resolver()),
      ["--instance", "stable", "--application-group-container", groupContainerURL.path])
    XCTAssertEqual(
      storeIdentity(applicationGroupIdentifier: nil).daemonArguments(groupContainerURL: resolver()),
      ["--instance", "stable"])
  }

  func test_should_fall_back_when_the_platform_cannot_resolve_the_group() {
    let unresolvableIdentity = storeIdentity(applicationGroupIdentifier: "TEAM123456.astronomical")

    XCTAssertEqual(unresolvableIdentity.daemonArguments(groupContainerURL: { _ in nil }), ["--instance", "stable"])
    XCTAssertEqual(
      unresolvableIdentity.stateDirectoryURL(homeDirectoryURL: homeDirectoryURL, groupContainerURL: { _ in nil })
        .path,
      homeDirectoryURL.appendingPathComponent("Library/Application Support/Astronomical").path)
  }

  func test_should_move_existing_state_into_the_group_container_once() throws {
    let legacyStateDirectoryURL = try writeLegacyState()
    let groupedIdentity = storeIdentity(applicationGroupIdentifier: "TEAM123456.astronomical")

    let firstOutcome = moveContainerStateIntoGroupContainerIfNeeded(
      applicationIdentity: groupedIdentity, groupContainerURL: resolver(), homeDirectoryURL: homeDirectoryURL)
    let secondOutcome = moveContainerStateIntoGroupContainerIfNeeded(
      applicationIdentity: groupedIdentity, groupContainerURL: resolver(), homeDirectoryURL: homeDirectoryURL)

    XCTAssertEqual(firstOutcome, .moved)
    XCTAssertEqual(secondOutcome, .nothingToMove)
    XCTAssertFalse(FileManager.default.fileExists(atPath: legacyStateDirectoryURL.path))
    XCTAssertTrue(
      FileManager.default.fileExists(
        atPath: groupContainerURL.appendingPathComponent("Astronomical/config.json").path))
    XCTAssertTrue(
      FileManager.default.fileExists(atPath: groupContainerURL.appendingPathComponent("Astronomical/models").path))
  }

  func test_should_move_nothing_for_a_fresh_install() {
    let outcome = moveContainerStateIntoGroupContainerIfNeeded(
      applicationIdentity: storeIdentity(applicationGroupIdentifier: "TEAM123456.astronomical"),
      groupContainerURL: resolver(), homeDirectoryURL: homeDirectoryURL)

    XCTAssertEqual(outcome, .nothingToMove)
    XCTAssertFalse(FileManager.default.fileExists(atPath: groupContainerURL.path))
  }

  /// The group container is what the engine is using; state already there is
  /// never overwritten or merged, and the old state is left untouched.
  func test_should_never_replace_state_already_in_the_group_container() throws {
    let legacyStateDirectoryURL = try writeLegacyState()
    let groupStateDirectoryURL = groupContainerURL.appendingPathComponent("Astronomical")
    try FileManager.default.createDirectory(at: groupStateDirectoryURL, withIntermediateDirectories: true)
    try Data("current".utf8).write(to: groupStateDirectoryURL.appendingPathComponent("config.json"))

    let outcome = moveContainerStateIntoGroupContainerIfNeeded(
      applicationIdentity: storeIdentity(applicationGroupIdentifier: "TEAM123456.astronomical"),
      groupContainerURL: resolver(), homeDirectoryURL: homeDirectoryURL)

    XCTAssertEqual(outcome, .alreadyInGroupContainer)
    XCTAssertEqual(
      try String(contentsOf: groupStateDirectoryURL.appendingPathComponent("config.json"), encoding: .utf8),
      "current")
    XCTAssertTrue(FileManager.default.fileExists(atPath: legacyStateDirectoryURL.path))
  }

  /// An agent that started before the move leaves a state directory holding
  /// only its lock file. That is not state: the old state must still move in,
  /// beside the lock the running agent keeps open.
  func test_should_move_state_into_a_group_directory_that_holds_only_the_agent_lock() throws {
    let legacyStateDirectoryURL = try writeLegacyState()
    let groupStateDirectoryURL = groupContainerURL.appendingPathComponent("Astronomical")
    try FileManager.default.createDirectory(at: groupStateDirectoryURL, withIntermediateDirectories: true)
    let agentLockFileURL = groupStateDirectoryURL.appendingPathComponent(
      ApplicationIdentity.agentInstanceLockFileName)
    try Data().write(to: agentLockFileURL)

    let outcome = moveContainerStateIntoGroupContainerIfNeeded(
      applicationIdentity: storeIdentity(applicationGroupIdentifier: "TEAM123456.astronomical"),
      groupContainerURL: resolver(), homeDirectoryURL: homeDirectoryURL)

    XCTAssertEqual(outcome, .moved)
    XCTAssertTrue(FileManager.default.fileExists(atPath: groupStateDirectoryURL.appendingPathComponent("config.json").path))
    XCTAssertTrue(FileManager.default.fileExists(atPath: groupStateDirectoryURL.appendingPathComponent("models").path))
    XCTAssertTrue(FileManager.default.fileExists(atPath: agentLockFileURL.path), "the agent's lock stays in place")
    XCTAssertFalse(FileManager.default.fileExists(atPath: legacyStateDirectoryURL.path))
  }

  /// One file beside the lock is state the engine wrote, and is never merged.
  func test_should_treat_any_file_beside_the_agent_lock_as_existing_state() throws {
    let legacyStateDirectoryURL = try writeLegacyState()
    let groupStateDirectoryURL = groupContainerURL.appendingPathComponent("Astronomical")
    try FileManager.default.createDirectory(at: groupStateDirectoryURL, withIntermediateDirectories: true)
    try Data().write(
      to: groupStateDirectoryURL.appendingPathComponent(ApplicationIdentity.agentInstanceLockFileName))
    try FileManager.default.createDirectory(
      at: groupStateDirectoryURL.appendingPathComponent("logs"), withIntermediateDirectories: true)

    let outcome = moveContainerStateIntoGroupContainerIfNeeded(
      applicationIdentity: storeIdentity(applicationGroupIdentifier: "TEAM123456.astronomical"),
      groupContainerURL: resolver(), homeDirectoryURL: homeDirectoryURL)

    XCTAssertEqual(outcome, .alreadyInGroupContainer)
    XCTAssertTrue(FileManager.default.fileExists(atPath: legacyStateDirectoryURL.path))
  }

  func test_should_move_state_into_an_empty_group_directory() throws {
    _ = try writeLegacyState()
    let groupStateDirectoryURL = groupContainerURL.appendingPathComponent("Astronomical")
    try FileManager.default.createDirectory(at: groupStateDirectoryURL, withIntermediateDirectories: true)

    let outcome = moveContainerStateIntoGroupContainerIfNeeded(
      applicationIdentity: storeIdentity(applicationGroupIdentifier: "TEAM123456.astronomical"),
      groupContainerURL: resolver(), homeDirectoryURL: homeDirectoryURL)

    XCTAssertEqual(outcome, .moved)
    XCTAssertTrue(FileManager.default.fileExists(atPath: groupStateDirectoryURL.appendingPathComponent("config.json").path))
  }

  func test_should_move_nothing_for_a_build_without_a_group() throws {
    let legacyStateDirectoryURL = try writeLegacyState()

    let outcome = moveContainerStateIntoGroupContainerIfNeeded(
      applicationIdentity: storeIdentity(applicationGroupIdentifier: nil),
      groupContainerURL: resolver(), homeDirectoryURL: homeDirectoryURL)

    XCTAssertEqual(outcome, .nothingToMove)
    XCTAssertTrue(FileManager.default.fileExists(atPath: legacyStateDirectoryURL.path))
  }

  /// A move that cannot happen is reported, never thrown: launch carries on
  /// and the old state stays where it was.
  func test_should_report_a_move_that_fails_and_leave_the_old_state_in_place() throws {
    let legacyStateDirectoryURL = try writeLegacyState()
    let blockedParentURL = scratchDirectoryURL.appendingPathComponent("blocked")
    try Data("a file where a folder is needed".utf8).write(to: blockedParentURL)
    let migration = ContainerStateMigration(
      legacyStateDirectoryURL: legacyStateDirectoryURL,
      groupStateDirectoryURL: blockedParentURL.appendingPathComponent("container/Astronomical"))

    guard case .failed = migration.moveLegacyStateIntoGroupContainer() else {
      return XCTFail("a blocked destination must be reported as a failure")
    }
    XCTAssertTrue(FileManager.default.fileExists(atPath: legacyStateDirectoryURL.path))
  }

  func test_should_report_no_group_for_a_bundle_whose_plist_names_none() {
    XCTAssertNil(ApplicationIdentity.current(bundle: Bundle(for: Self.self)).applicationGroupIdentifier)
  }

  /// The packager writes the group into each application's plist; this is the
  /// only way either process learns which container they share.
  func test_should_read_the_group_identifier_the_packager_wrote_into_the_plist() throws {
    let bundleURL = scratchDirectoryURL.appendingPathComponent("Store.app", isDirectory: true)
    let contentsURL = bundleURL.appendingPathComponent("Contents", isDirectory: true)
    try FileManager.default.createDirectory(at: contentsURL, withIntermediateDirectories: true)
    let storePlist: [String: Any] = [
      "CFBundleIdentifier": "app.astronomical.appstore",
      "AstronomicalChannel": "app-store",
      "AstronomicalApplicationGroupIdentifier": "TEAM123456.astronomical",
    ]
    try PropertyListSerialization.data(fromPropertyList: storePlist, format: .xml, options: 0)
      .write(to: contentsURL.appendingPathComponent("Info.plist"))
    let storeBundle = try XCTUnwrap(Bundle(url: bundleURL))

    let storeIdentity = ApplicationIdentity.current(bundle: storeBundle)

    XCTAssertEqual(storeIdentity.channel, .appStore)
    XCTAssertEqual(storeIdentity.applicationGroupIdentifier, "TEAM123456.astronomical")
    XCTAssertNil(storeIdentity.stateDirectoryName, "a store build names no home dot-folder")
  }
}
