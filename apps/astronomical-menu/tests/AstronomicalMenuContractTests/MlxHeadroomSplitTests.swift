import XCTest

@testable import AstronomicalMenuCore

final class MlxHeadroomSplitTests: XCTestCase {
  func test_should_keep_undifferentiated_headroom_when_the_engine_omits_the_split() {
    let headroomSplit = MlxHeadroomSplit.from(utilization: nil, availableByteCount: 5_400_000_000)

    XCTAssertFalse(headroomSplit.enginePublishedTheSplit)
    XCTAssertEqual(headroomSplit.remainderByteCount, 5_400_000_000)
    XCTAssertEqual(headroomSplit.unusedBudgetByteCount, 0)
    XCTAssertEqual(headroomSplit.paintedByteCount, 5_400_000_000)
  }

  func test_should_scale_named_owners_onto_the_unused_capacity_width() {
    let utilization = MlxMemoryCeilingUtilization(
      unusedHeadroomBytes: 4_970_000_000,
      reservedModelCoreSlackBytes: 670_000_000,
      reservedContextGrowthBytes: 1_300_000_000,
      reservedActivationAndWorkspaceBytes: 30_000_000,
      unseatedExpertEntitlementBytes: 2_970_000_000,
      speculativeDraftPayloadBytes: 0,
      unexplainedHeadroomBytes: 0,
      ownerOverrunBytes: 0
    )
    let availableByteCount: UInt64 = 4_970_000_000
    let headroomSplit = MlxHeadroomSplit.from(
      utilization: utilization, availableByteCount: availableByteCount)

    XCTAssertTrue(headroomSplit.enginePublishedTheSplit)
    XCTAssertEqual(headroomSplit.paintedByteCount, availableByteCount)
    XCTAssertEqual(
      headroomSplit.unusedBudgetByteCount
        + headroomSplit.reservedContextGrowthByteCount
        + headroomSplit.reservedActivationByteCount
        + headroomSplit.reservedModelCoreSlackByteCount
        + headroomSplit.unexplainedHeadroomByteCount
        + headroomSplit.remainderByteCount,
      availableByteCount
    )
    XCTAssertGreaterThan(headroomSplit.unusedBudgetByteCount, 0)
    XCTAssertGreaterThan(headroomSplit.reservedContextGrowthByteCount, 0)
    XCTAssertEqual(headroomSplit.ownerOverrunByteCount, 0)
  }

  func test_should_decode_engine_utilization_from_the_status_snapshot() throws {
    let statusDocument = try JSONDecoder().decode(
      SupervisorStatusDocument.self,
      from: Data(
        """
        {"status":"ready","activity":"generating","mlx_memory_ceiling_bytes":23000000000,"mlx_memory_snapshot":{"source":"decode_submitted","active_memory_bytes":18030000000,"allocator_cache_memory_bytes":0,"peak_memory_bytes":19570000000,"expert_payload_bytes":14950000000,"model_core_payload_bytes":2600000000,"context_state_payload_bytes":1120000000,"speculative_prefill_draft_memory_bytes":0,"memory_ceiling_utilization":{"unused_headroom_bytes":4970000000,"reserved_model_core_slack_bytes":670000000,"reserved_context_growth_bytes":1300000000,"reserved_activation_and_workspace_bytes":30000000,"unseated_expert_entitlement_bytes":2970000000,"speculative_draft_payload_bytes":0,"unexplained_headroom_bytes":0,"owner_overrun_bytes":0}}}
        """.utf8)
    )

    let utilization = try XCTUnwrap(statusDocument.mlxMemorySnapshot?.memoryCeilingUtilization)
    XCTAssertEqual(utilization.unseatedExpertEntitlementBytes, 2_970_000_000)
    XCTAssertEqual(statusDocument.mlxHeadroomSplit.enginePublishedTheSplit, true)
    XCTAssertEqual(
      statusDocument.mlxHeadroomSplit.paintedByteCount,
      statusDocument.mlxMemoryBreakdown.availableByteCount
    )
  }

  func test_should_keep_legacy_snapshots_on_the_single_headroom_row() throws {
    let statusDocument = try JSONDecoder().decode(
      SupervisorStatusDocument.self,
      from: Data(
        """
        {"status":"ready","activity":"idle","mlx_memory_ceiling_bytes":23000000000,"mlx_memory_snapshot":{"source":"idle_poll","active_memory_bytes":17600000000,"allocator_cache_memory_bytes":0,"peak_memory_bytes":19480000000,"expert_payload_bytes":15000000000,"model_core_payload_bytes":2600000000,"context_state_payload_bytes":0}}
        """.utf8)
    )

    XCTAssertNil(statusDocument.mlxMemorySnapshot?.memoryCeilingUtilization)
    XCTAssertFalse(statusDocument.mlxHeadroomSplit.enginePublishedTheSplit)
    XCTAssertEqual(statusDocument.mlxHeadroomSplit.remainderByteCount, 5_400_000_000)
  }

  func test_should_give_rounding_remainder_to_a_positive_owner() {
    let allocated = allocateProportions([2, 1], onto: 10)
    XCTAssertEqual(allocated.reduce(0, +), 10)
    XCTAssertEqual(allocated[0], 6)
    XCTAssertEqual(allocated[1], 4)
  }
}
