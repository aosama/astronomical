import Foundation
import XCTest

@testable import AstronomicalMenu

final class SupervisorStatusCompatibilityTests: XCTestCase {
  func test_should_decode_image_generation_progress_from_the_public_status_contract() throws {
    let statusDocument = try JSONDecoder().decode(
      SupervisorStatusDocument.self,
      from: Data(
        #"{"status":"ready","activity":"image_generation","ready_model_id":"fictional/flux-model","progress":{"phase":"denoising","completed_steps":2,"total_steps":4,"elapsed_ms":1000}}"#.utf8
      )
    )

    XCTAssertEqual(statusDocument.readyModelIdentifier, "fictional/flux-model")
    XCTAssertEqual(statusDocument.progress?.completedUnitCount, 2)
    XCTAssertEqual(statusDocument.progress?.totalUnitCount, 4)
    XCTAssertEqual(statusDocument.progress?.unit, .steps)
  }

  func test_should_decode_complete_mtp_depth_status_without_inventing_application() throws {
    let statusDocument = try JSONDecoder().decode(
      SupervisorStatusDocument.self,
      from: Data(#"{"status":"ready","activity":"idle","mtp_enabled":true,"mtp_configured_draft_depth":3,"mtp_artifact_maximum_draft_depth":3,"mtp_artifact_default_draft_depth":2,"mtp_resolved_requested_draft_depth":3,"mtp_effective_execution_draft_depth":1}"#.utf8)
    )

    XCTAssertEqual(statusDocument.mtpConfiguredDraftDepth, 3)
    XCTAssertEqual(statusDocument.mtpArtifactMaximumDraftDepth, 3)
    XCTAssertEqual(statusDocument.mtpArtifactDefaultDraftDepth, 2)
    XCTAssertEqual(statusDocument.mtpResolvedRequestedDraftDepth, 3)
    XCTAssertEqual(statusDocument.mtpEffectiveExecutionDraftDepth, 1)
  }
  func test_should_decode_a_legacy_status_document_without_new_telemetry_fields() throws {
    let statusDocument = try JSONDecoder().decode(
      SupervisorStatusDocument.self,
      from: Data(
        """
        {"status":"ready","activity":"idle","ready_model_id":null,"progress":null,"expert_memory_mode":null}
        """.utf8)
    )

    XCTAssertNil(statusDocument.mlxMemorySnapshot)
    XCTAssertNil(statusDocument.application)
    XCTAssertEqual(statusDocument.mlxMemoryCeilingBytes, 0)
    XCTAssertEqual(statusDocument.mlxMemoryBreakdown.speculativePrefillDraftMemoryByteCount, 0)
    XCTAssertEqual(statusDocument.servingSession.completedRequestCount, 0)
  }
}
