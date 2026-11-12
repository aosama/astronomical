import XCTest

@testable import AstronomicalMenuCore

final class MaximumMlxMemoryControlPresentationTests: XCTestCase {
  // A 48 GB Mac: a 40 GiB machine ceiling and the 40,200,896,512-byte wired
  // claim measured in issue #1297, which the engine plans 2.5% below
  // (39,195,874,112 bytes, 36.50 GB).
  private let machineCeilingBytes: UInt64 = 42_949_672_960
  private let wiredResidencyBudgetBytes: UInt64 = 40_200_896_512
  private let plannableWiredResidencyBytes: UInt64 = 39_195_874_112

  private func statusDocument(
    liveCeilingBytes: UInt64,
    wiredResidencyBudgetBytes: UInt64?,
    configuredMaximumGigabytes: UInt64?,
    pendingCeilingBytes: UInt64? = nil
  ) -> DaemonStatusDocument {
    DaemonStatusDocument(
      status: "ready", activity: "idle", readyModelIdentifier: nil, progress: nil,
      expertMemoryMode: nil,
      mlxMemoryCeilingBytes: liveCeilingBytes,
      machineMlxMemoryCeilingBytes: machineCeilingBytes,
      wiredResidencyBudgetBytes: wiredResidencyBudgetBytes,
      configuredMaximumMlxMemoryGigabytes: configuredMaximumGigabytes,
      pendingMlxMemoryCeilingBytes: pendingCeilingBytes,
      servingSession: .empty
    )
  }

  func test_should_show_only_the_reachable_limit_when_the_request_is_served() {
    let servedStatusDocument = statusDocument(
      liveCeilingBytes: 34 * mlxMemoryBytesPerGigabyte,
      wiredResidencyBudgetBytes: wiredResidencyBudgetBytes,
      configuredMaximumGigabytes: 34
    )

    let controlPresentation = MaximumMlxMemoryControlPresentation(
      statusDocument: servedStatusDocument, editableMaximumWholeGigabytes: 34)

    XCTAssertEqual(controlPresentation.reachableMaximumCaption, "of 36 GB")
    XCTAssertFalse(controlPresentation.shouldOfferApply)
    XCTAssertTrue(controlPresentation.shouldOfferMacMaximum)
    XCTAssertNil(controlPresentation.cappedCeilingCaption)
  }

  func test_should_offer_apply_only_after_the_stepper_moves() {
    let servedStatusDocument = statusDocument(
      liveCeilingBytes: 34 * mlxMemoryBytesPerGigabyte,
      wiredResidencyBudgetBytes: wiredResidencyBudgetBytes,
      configuredMaximumGigabytes: 34
    )

    let editedPresentation = MaximumMlxMemoryControlPresentation(
      statusDocument: servedStatusDocument, editableMaximumWholeGigabytes: 30)

    XCTAssertTrue(editedPresentation.shouldOfferApply)
  }

  func test_should_hide_use_max_and_the_cap_while_the_mac_maximum_is_in_force() {
    // "Use Max" asks for whatever the Mac allows; serving the plannable wired
    // budget below the machine ceiling honours it and must not read as a cap.
    let macMaximumStatusDocument = statusDocument(
      liveCeilingBytes: plannableWiredResidencyBytes,
      wiredResidencyBudgetBytes: wiredResidencyBudgetBytes,
      configuredMaximumGigabytes: nil
    )

    let controlPresentation = MaximumMlxMemoryControlPresentation(
      statusDocument: macMaximumStatusDocument, editableMaximumWholeGigabytes: 36)

    XCTAssertFalse(controlPresentation.shouldOfferMacMaximum)
    XCTAssertFalse(controlPresentation.shouldOfferApply)
    XCTAssertNil(controlPresentation.cappedCeilingCaption)
  }

  func test_should_never_offer_a_stepper_value_the_engine_would_cap() {
    let wiredStatusDocument = statusDocument(
      liveCeilingBytes: plannableWiredResidencyBytes,
      wiredResidencyBudgetBytes: wiredResidencyBudgetBytes,
      configuredMaximumGigabytes: nil
    )
    let reachableWholeGigabytes = reachableMaximumMlxMemoryWholeGigabytes(for: wiredStatusDocument)

    XCTAssertEqual(reachableWholeGigabytes, 36)
    XCTAssertEqual(
      mlxMemoryCeilingTheEngineWillInstall(
        requestedCeilingBytes: reachableWholeGigabytes * mlxMemoryBytesPerGigabyte,
        wiredResidencyBudgetBytes: wiredResidencyBudgetBytes
      ),
      reachableWholeGigabytes * mlxMemoryBytesPerGigabyte
    )
  }

  func test_should_reach_the_machine_ceiling_when_serving_is_pageable() {
    let pageableStatusDocument = statusDocument(
      liveCeilingBytes: machineCeilingBytes,
      wiredResidencyBudgetBytes: nil,
      configuredMaximumGigabytes: nil
    )

    XCTAssertEqual(reachableMaximumMlxMemoryWholeGigabytes(for: pageableStatusDocument), 40)
  }

  func test_should_keep_the_metal_margin_when_the_wired_claim_equals_the_machine_ceiling() {
    let fullClaimStatusDocument = statusDocument(
      liveCeilingBytes: machineCeilingBytes,
      wiredResidencyBudgetBytes: machineCeilingBytes,
      configuredMaximumGigabytes: nil
    )

    XCTAssertEqual(reachableMaximumMlxMemoryWholeGigabytes(for: fullClaimStatusDocument), 39)
  }

  func test_should_report_a_cap_when_an_explicit_request_exceeds_the_live_ceiling() {
    // Only the configuration file can ask for 40 GB here; the stepper stops at 36.
    let cappedStatusDocument = statusDocument(
      liveCeilingBytes: plannableWiredResidencyBytes,
      wiredResidencyBudgetBytes: wiredResidencyBudgetBytes,
      configuredMaximumGigabytes: 40
    )

    let controlPresentation = MaximumMlxMemoryControlPresentation(
      statusDocument: cappedStatusDocument,
      editableMaximumWholeGigabytes: appliedMaximumMlxMemoryWholeGigabytes(for: cappedStatusDocument)
    )

    XCTAssertEqual(controlPresentation.cappedCeilingCaption, "Capped at 36.50 GB on this Mac")
    XCTAssertEqual(appliedMaximumMlxMemoryWholeGigabytes(for: cappedStatusDocument), 36)
    // The stepper already rests on 36, so Apply must still be offered or the
    // cap notice has no one-click way out.
    XCTAssertTrue(controlPresentation.shouldOfferApply)
  }

  func test_should_not_report_a_cap_while_the_requested_change_is_queued() {
    let queuedStatusDocument = statusDocument(
      liveCeilingBytes: 30 * mlxMemoryBytesPerGigabyte,
      wiredResidencyBudgetBytes: wiredResidencyBudgetBytes,
      configuredMaximumGigabytes: 34,
      pendingCeilingBytes: 34 * mlxMemoryBytesPerGigabyte
    )

    let controlPresentation = MaximumMlxMemoryControlPresentation(
      statusDocument: queuedStatusDocument, editableMaximumWholeGigabytes: 34)

    XCTAssertNil(controlPresentation.cappedCeilingCaption)
  }

  func test_should_report_a_cap_when_the_configured_request_overflows_bytes() {
    let overflowingStatusDocument = statusDocument(
      liveCeilingBytes: plannableWiredResidencyBytes,
      wiredResidencyBudgetBytes: wiredResidencyBudgetBytes,
      configuredMaximumGigabytes: UInt64.max
    )

    let controlPresentation = MaximumMlxMemoryControlPresentation(
      statusDocument: overflowingStatusDocument, editableMaximumWholeGigabytes: 36)

    XCTAssertEqual(controlPresentation.cappedCeilingCaption, "Capped at 36.50 GB on this Mac")
  }

  func test_should_not_report_a_cap_before_the_engine_publishes_a_ceiling() {
    let unmeasuredStatusDocument = statusDocument(
      liveCeilingBytes: 0,
      wiredResidencyBudgetBytes: nil,
      configuredMaximumGigabytes: 34
    )

    let controlPresentation = MaximumMlxMemoryControlPresentation(
      statusDocument: unmeasuredStatusDocument, editableMaximumWholeGigabytes: 34)

    XCTAssertNil(controlPresentation.cappedCeilingCaption)
  }
}
