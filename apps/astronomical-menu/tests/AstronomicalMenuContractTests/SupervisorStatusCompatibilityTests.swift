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
      from: Data(#"{"status":"ready","activity":"idle","mtp_enabled":true,"mtp_configured_draft_depth":3,"mtp_artifact_maximum_draft_depth":1,"mtp_artifact_default_draft_depth":null,"mtp_resolved_requested_draft_depth":3,"mtp_capped_draft_depth":1,"mtp_effective_execution_draft_depth":1,"mtp_depth_resolution_reason":"configured MTP draft depth was clamped to the declared artifact maximum"}"#.utf8)
    )

    XCTAssertEqual(statusDocument.mtpConfiguredDraftDepth, 3)
    XCTAssertEqual(statusDocument.mtpArtifactMaximumDraftDepth, 1)
    XCTAssertNil(statusDocument.mtpArtifactDefaultDraftDepth)
    XCTAssertEqual(statusDocument.mtpResolvedRequestedDraftDepth, 3)
    XCTAssertEqual(statusDocument.mtpCappedDraftDepth, 1)
    XCTAssertEqual(statusDocument.mtpEffectiveExecutionDraftDepth, 1)
    XCTAssertEqual(
      statusDocument.mtpDepthResolutionReason,
      "configured MTP draft depth was clamped to the declared artifact maximum")
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

  func test_should_decode_path_safe_duplicate_model_feedback_for_the_menu() throws {
    let statusDocument = try JSONDecoder().decode(
      SupervisorStatusDocument.self,
      from: Data(
        #"{"status":"ready","activity":"idle","configuration":{"configured_generation":"configured","resolved_generation":"resolved","effective_generation":"effective","is_effective":false,"restart_required":false,"model_discovery_diagnostics":[{"code":"ambiguous_model_identity","model_id":"shared-model","configured_root_numbers":[1,2]}]}}"#.utf8
      )
    )

    let diagnostic = try XCTUnwrap(statusDocument.configuration?.modelDiscoveryDiagnostics?.first)
    XCTAssertEqual(diagnostic.code, "ambiguous_model_identity")
    XCTAssertEqual(diagnostic.message, "Model shared-model appears in model_directories entries 1, 2. Remove one duplicate root.")
    XCTAssertFalse(diagnostic.message.contains("private-root-marker"))
  }

  func test_should_decode_unavailable_model_directory_feedback_for_the_menu() throws {
    let statusDocument = try JSONDecoder().decode(
      SupervisorStatusDocument.self,
      from: Data(
        #"{"status":"ready","activity":"idle","configuration":{"configured_generation":"configured","resolved_generation":"resolved","effective_generation":"effective","is_effective":true,"restart_required":false,"model_discovery_diagnostics":[{"code":"unavailable_model_directory","model_id":"","configured_root_numbers":[4]}]}}"#.utf8
      )
    )

    let diagnostic = try XCTUnwrap(statusDocument.configuration?.modelDiscoveryDiagnostics?.first)
    XCTAssertEqual(diagnostic.code, "unavailable_model_directory")
    XCTAssertEqual(diagnostic.configuredRootNumbers, [4])
    XCTAssertFalse(diagnostic.message.contains("private-root-marker"))
  }
}
