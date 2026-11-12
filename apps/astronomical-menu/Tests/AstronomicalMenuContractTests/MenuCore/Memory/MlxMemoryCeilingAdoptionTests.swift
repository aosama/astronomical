import XCTest

@testable import AstronomicalMenuCore

final class MlxMemoryCeilingAdoptionTests: XCTestCase {
  /// The wired claim the worker logged for this machine, and the 36.50 GB
  /// ceiling a 37 GB request installs against it.
  private let incidentWiredResidencyBudgetBytes: UInt64 = 40_200_896_512
  private let incidentPlannableWiredResidencyBytes: UInt64 = 39_195_874_112
  private let requestedThirtySevenGigabyteCeilingBytes: UInt64 = 37 * mlxMemoryBytesPerGigabyte
  private let bootTwentyOneGigabyteCeilingBytes: UInt64 = 21 * mlxMemoryBytesPerGigabyte

  func test_should_adopt_a_37_gb_request_when_the_live_ceiling_is_the_plannable_wired_budget() {
    XCTAssertEqual(
      mlxMemoryCeilingTheEngineWillInstall(
        requestedCeilingBytes: requestedThirtySevenGigabyteCeilingBytes,
        wiredResidencyBudgetBytes: incidentWiredResidencyBudgetBytes
      ),
      incidentPlannableWiredResidencyBytes
    )
    XCTAssertTrue(
      liveMlxMemoryCeilingAdoptedRequest(
        liveCeilingBytes: incidentPlannableWiredResidencyBytes,
        requestedCeilingBytes: requestedThirtySevenGigabyteCeilingBytes,
        wiredResidencyBudgetBytes: incidentWiredResidencyBudgetBytes
      )
    )
  }

  func test_should_adopt_a_request_that_fits_under_the_wired_budget() {
    let requestedCeilingBytes = 32 * mlxMemoryBytesPerGigabyte

    XCTAssertTrue(
      liveMlxMemoryCeilingAdoptedRequest(
        liveCeilingBytes: requestedCeilingBytes,
        requestedCeilingBytes: requestedCeilingBytes,
        wiredResidencyBudgetBytes: incidentWiredResidencyBudgetBytes
      )
    )
  }

  func test_should_not_adopt_while_the_live_ceiling_is_still_the_previous_boot_ceiling() {
    XCTAssertFalse(
      liveMlxMemoryCeilingAdoptedRequest(
        liveCeilingBytes: bootTwentyOneGigabyteCeilingBytes,
        requestedCeilingBytes: requestedThirtySevenGigabyteCeilingBytes,
        wiredResidencyBudgetBytes: incidentWiredResidencyBudgetBytes
      )
    )
  }

  func test_should_not_treat_the_raw_37_gb_request_as_adopted_when_the_engine_must_clamp_it() {
    XCTAssertFalse(
      liveMlxMemoryCeilingAdoptedRequest(
        liveCeilingBytes: requestedThirtySevenGigabyteCeilingBytes,
        requestedCeilingBytes: requestedThirtySevenGigabyteCeilingBytes,
        wiredResidencyBudgetBytes: incidentWiredResidencyBudgetBytes
      )
    )
  }

  func test_should_require_the_raw_request_when_no_wired_budget_was_reported() {
    XCTAssertTrue(
      liveMlxMemoryCeilingAdoptedRequest(
        liveCeilingBytes: requestedThirtySevenGigabyteCeilingBytes,
        requestedCeilingBytes: requestedThirtySevenGigabyteCeilingBytes,
        wiredResidencyBudgetBytes: nil
      )
    )
    XCTAssertFalse(
      liveMlxMemoryCeilingAdoptedRequest(
        liveCeilingBytes: incidentPlannableWiredResidencyBytes,
        requestedCeilingBytes: requestedThirtySevenGigabyteCeilingBytes,
        wiredResidencyBudgetBytes: nil
      )
    )
  }

  func test_should_adopt_when_the_request_equals_the_plannable_wired_budget() {
    XCTAssertTrue(
      liveMlxMemoryCeilingAdoptedRequest(
        liveCeilingBytes: incidentPlannableWiredResidencyBytes,
        requestedCeilingBytes: incidentPlannableWiredResidencyBytes,
        wiredResidencyBudgetBytes: incidentWiredResidencyBudgetBytes
      )
    )
  }

  func test_should_keep_a_sub_thousand_byte_wired_claim_unchanged() {
    XCTAssertEqual(
      mlxMemoryCeilingTheEngineWillInstall(
        requestedCeilingBytes: 10_000,
        wiredResidencyBudgetBytes: 999
      ),
      999
    )
  }

  func test_should_reserve_the_margin_once_the_wired_claim_reaches_one_thousand_bytes() {
    XCTAssertEqual(
      mlxMemoryCeilingTheEngineWillInstall(
        requestedCeilingBytes: 10_000,
        wiredResidencyBudgetBytes: 1_000
      ),
      975
    )
  }

  func test_should_install_nothing_when_the_wired_claim_is_zero() {
    XCTAssertEqual(
      mlxMemoryCeilingTheEngineWillInstall(
        requestedCeilingBytes: requestedThirtySevenGigabyteCeilingBytes,
        wiredResidencyBudgetBytes: 0
      ),
      0
    )
    XCTAssertTrue(
      liveMlxMemoryCeilingAdoptedRequest(
        liveCeilingBytes: 0,
        requestedCeilingBytes: requestedThirtySevenGigabyteCeilingBytes,
        wiredResidencyBudgetBytes: 0
      )
    )
  }
}
