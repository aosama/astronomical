import Foundation
import XCTest

@testable import AstronomicalMenu

final class SupervisorStatusCompatibilityTests: XCTestCase {
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
