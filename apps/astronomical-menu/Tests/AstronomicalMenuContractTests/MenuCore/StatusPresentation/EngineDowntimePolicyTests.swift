import XCTest

@testable import AstronomicalMenuCore

final class EngineDowntimePolicyTests: XCTestCase {
  private let firstPollTime = Date(timeIntervalSinceReferenceDate: 1_000)

  private var readyStatusDocument: DaemonStatusDocument {
    DaemonStatusDocument(
      status: "ready", activity: "idle", readyModelIdentifier: nil, progress: nil,
      expertMemoryMode: nil, mlxMemoryCeilingBytes: 0, servingSession: .empty)
  }

  func test_should_not_report_a_stopped_engine_within_the_grace_period() {
    var engineDowntimePolicy = EngineDowntimePolicy()
    XCTAssertFalse(engineDowntimePolicy.engineIsDown(for: .unavailable, at: firstPollTime))
    XCTAssertFalse(
      engineDowntimePolicy.engineIsDown(
        for: .unavailable, at: firstPollTime.addingTimeInterval(engineDowntimeGraceSeconds - 0.1)))
  }

  func test_should_report_a_stopped_engine_once_the_grace_period_has_passed() {
    var engineDowntimePolicy = EngineDowntimePolicy()
    _ = engineDowntimePolicy.engineIsDown(for: .unavailable, at: firstPollTime)
    XCTAssertTrue(
      engineDowntimePolicy.engineIsDown(
        for: .unavailable, at: firstPollTime.addingTimeInterval(engineDowntimeGraceSeconds)))
  }

  func test_should_start_the_grace_period_again_after_the_engine_answers() {
    var engineDowntimePolicy = EngineDowntimePolicy()
    _ = engineDowntimePolicy.engineIsDown(for: .unavailable, at: firstPollTime)
    let answerTime = firstPollTime.addingTimeInterval(engineDowntimeGraceSeconds + 1)
    XCTAssertFalse(engineDowntimePolicy.engineIsDown(for: readyStatusDocument, at: answerTime))
    XCTAssertFalse(
      engineDowntimePolicy.engineIsDown(for: .unavailable, at: answerTime.addingTimeInterval(1)))
  }
}
