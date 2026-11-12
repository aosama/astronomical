import XCTest

@testable import AstronomicalAppCore

/// A link can launch the main app and ask for something the app cannot do
/// until launching has finished; the request must wait rather than be lost.
@MainActor
final class LaunchDeferredActionTests: XCTestCase {
  private var runCount = 0

  private func makeDeferredAction() -> LaunchDeferredAction {
    LaunchDeferredAction { [weak self] in self?.runCount += 1 }
  }

  func test_should_run_at_once_when_asked_after_launching() {
    let deferredAction = makeDeferredAction()
    deferredAction.becomeReady()

    deferredAction.request()

    XCTAssertEqual(runCount, 1)
  }

  func test_should_hold_a_request_made_while_launching_until_ready() {
    let deferredAction = makeDeferredAction()

    deferredAction.request()
    XCTAssertEqual(runCount, 0, "nothing runs before the app is ready")

    deferredAction.becomeReady()
    XCTAssertEqual(runCount, 1)
  }

  func test_should_run_once_for_repeated_requests_made_while_launching() {
    let deferredAction = makeDeferredAction()
    deferredAction.request()
    deferredAction.request()

    deferredAction.becomeReady()

    XCTAssertEqual(runCount, 1)
  }

  func test_should_not_run_when_nothing_was_requested() {
    let deferredAction = makeDeferredAction()

    deferredAction.becomeReady()

    XCTAssertEqual(runCount, 0)
  }

  func test_should_run_every_request_made_after_launching() {
    let deferredAction = makeDeferredAction()
    deferredAction.request()
    deferredAction.becomeReady()

    deferredAction.request()

    XCTAssertEqual(runCount, 2)
  }
}
