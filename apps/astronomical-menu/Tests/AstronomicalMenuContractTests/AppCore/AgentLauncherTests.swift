import Foundation
import XCTest

@testable import AstronomicalAppCore

/// The main app starts its bundled agent and, before an update, waits for it
/// to leave. These tests cover what can be decided without launching one, so
/// they name agents that cannot be the Development agent a contributor has
/// running.
@MainActor
final class AgentLauncherTests: XCTestCase {
  private static let absentAgentIdentifier = "dev.astronomical.tests.absent-agent"

  /// Like a bare `swift run`, a host with no agent bundle has nothing to start.
  func test_should_do_nothing_when_the_host_carries_no_agent_bundle() {
    let launcher = AgentLauncher(agentBundleIdentifier: Self.absentAgentIdentifier, agentBundleURL: nil)

    launcher.launchAgentIfNeeded()

    XCTAssertFalse(launcher.isAgentRunning)
  }

  /// The test bundle has no Contents/Library/LoginItems, so the launcher made
  /// for it finds no agent and launching stays a no-op.
  func test_should_find_no_agent_in_a_bundle_without_login_items() {
    XCTAssertNil(ApplicationBundleLayout.agentBundleURL(inApplicationBundleAt: Bundle(for: Self.self).bundleURL))
  }

  func test_should_report_an_agent_that_is_not_running_as_already_gone() {
    let launcher = AgentLauncher(agentBundleIdentifier: Self.absentAgentIdentifier, agentBundleURL: nil)
    let waitStartedAt = ContinuousClock.now

    let agentLeft = launcher.waitForAgentExit(timeout: .seconds(5))

    XCTAssertTrue(agentLeft)
    XCTAssertLessThan(ContinuousClock.now - waitStartedAt, .seconds(1), "there is nothing to wait for")
  }

  /// An update must not hang on an agent that ignores the quit: the wait gives
  /// up at its deadline and says the agent is still there. Finder stands in
  /// for a process that will not leave.
  func test_should_give_up_at_the_deadline_when_the_agent_stays() throws {
    let stayingLauncher = AgentLauncher(agentBundleIdentifier: "com.apple.finder", agentBundleURL: nil)
    try XCTSkipUnless(stayingLauncher.isAgentRunning, "Finder is not running on this machine")
    let waitStartedAt = ContinuousClock.now

    let agentLeft = stayingLauncher.waitForAgentExit(timeout: .milliseconds(300))

    XCTAssertFalse(agentLeft)
    XCTAssertGreaterThanOrEqual(ContinuousClock.now - waitStartedAt, .milliseconds(300))
  }
}
