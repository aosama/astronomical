import XCTest

@testable import AstronomicalMenuCore

/// Only one menu-bar agent of a channel may run: the lock is what stops a
/// second one launched at the same moment by another launch path.
final class AgentInstanceLockTests: XCTestCase {
  private var stateDirectoryURL: URL!

  override func setUpWithError() throws {
    stateDirectoryURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("agent-instance-lock-\(UUID().uuidString)", isDirectory: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: stateDirectoryURL)
  }

  func test_should_grant_the_lock_to_the_first_agent() throws {
    XCTAssertNotNil(try AgentInstanceLock.acquire(inStateDirectory: stateDirectoryURL))
  }

  func test_should_refuse_a_second_agent_while_the_first_holds_the_lock() throws {
    let firstAgentLock = try AgentInstanceLock.acquire(inStateDirectory: stateDirectoryURL)

    XCTAssertNotNil(firstAgentLock)
    XCTAssertNil(try AgentInstanceLock.acquire(inStateDirectory: stateDirectoryURL))
  }

  /// An agent that quit, or crashed, must not keep the next one out.
  func test_should_grant_the_lock_again_once_the_holder_is_gone() throws {
    var firstAgentLock = try AgentInstanceLock.acquire(inStateDirectory: stateDirectoryURL)
    XCTAssertNotNil(firstAgentLock)
    firstAgentLock = nil

    XCTAssertNotNil(try AgentInstanceLock.acquire(inStateDirectory: stateDirectoryURL))
  }

  /// A first launch has no state directory yet; the agent starts before the
  /// engine creates it.
  func test_should_create_a_missing_state_directory() throws {
    let nestedStateDirectoryURL = stateDirectoryURL.appendingPathComponent("not/yet/there", isDirectory: true)

    XCTAssertNotNil(try AgentInstanceLock.acquire(inStateDirectory: nestedStateDirectoryURL))
    XCTAssertTrue(FileManager.default.fileExists(atPath: nestedStateDirectoryURL.path))
  }

  func test_should_keep_channels_apart_by_their_state_directories() throws {
    let otherChannelStateDirectoryURL = stateDirectoryURL.appendingPathComponent("other-channel")
    let stableAgentLock = try AgentInstanceLock.acquire(inStateDirectory: stateDirectoryURL)

    XCTAssertNotNil(stableAgentLock)
    XCTAssertNotNil(try AgentInstanceLock.acquire(inStateDirectory: otherChannelStateDirectoryURL))
  }

  /// A state location that cannot hold a lock file is a fault to report, not
  /// a second agent: answering nil here would make a lone agent quit.
  func test_should_throw_when_the_state_directory_cannot_be_created() throws {
    let blockingFileURL = stateDirectoryURL!
    try Data().write(to: blockingFileURL)

    XCTAssertThrowsError(
      try AgentInstanceLock.acquire(inStateDirectory: blockingFileURL.appendingPathComponent("state")))
  }
}
