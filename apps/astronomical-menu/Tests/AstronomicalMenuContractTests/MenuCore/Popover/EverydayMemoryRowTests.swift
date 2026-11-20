import XCTest

@testable import AstronomicalMenuCore

final class EverydayMemoryRowTests: XCTestCase {
  private let oneGigabyte: UInt64 = 1_073_741_824

  private func statusDocument(activeBytes: UInt64?, ceilingBytes: UInt64) throws -> DaemonStatusDocument {
    let snapshotField =
      activeBytes.map {
        ",\"mlx_memory_snapshot\":{\"source\":\"generation\",\"active_memory_bytes\":\($0),\"mlx_buffer_cache_bytes\":0,\"peak_memory_bytes\":\($0),\"expert_payload_bytes\":0,\"model_core_payload_bytes\":0,\"context_state_payload_bytes\":0}"
      } ?? ""
    let statusJSON =
      "{\"status\":\"ready\",\"activity\":\"idle\",\"mlx_memory_ceiling_bytes\":\(ceilingBytes)\(snapshotField)}"
    return try JSONDecoder().decode(DaemonStatusDocument.self, from: Data(statusJSON.utf8))
  }

  func test_should_say_how_much_memory_is_used_of_the_limit_in_binary_gigabytes() throws {
    let halfUsed = try statusDocument(activeBytes: 18 * oneGigabyte, ceilingBytes: 36 * oneGigabyte)
    XCTAssertEqual(halfUsed.everydayMemoryTitle, "18.00 GB of 36.00 GB")
    XCTAssertEqual(halfUsed.everydayMemoryFillFraction, 0.5, accuracy: 0.0001)
  }

  func test_should_name_the_limit_when_no_model_holds_memory() throws {
    let nothingLoaded = try statusDocument(activeBytes: nil, ceilingBytes: 36 * oneGigabyte)
    XCTAssertEqual(nothingLoaded.everydayMemoryTitle, "None used of 36.00 GB")
    XCTAssertEqual(nothingLoaded.everydayMemoryFillFraction, 0)
  }

  func test_should_not_claim_a_figure_before_the_engine_reports_a_limit() {
    XCTAssertEqual(DaemonStatusDocument.unavailable.everydayMemoryTitle, "Not measured")
    XCTAssertEqual(DaemonStatusDocument.unavailable.everydayMemoryFillFraction, 0)
  }

  func test_should_keep_the_bar_inside_its_track_when_use_passes_the_limit() throws {
    let overLimit = try statusDocument(activeBytes: 40 * oneGigabyte, ceilingBytes: 36 * oneGigabyte)
    XCTAssertEqual(overLimit.everydayMemoryFillFraction, 1, accuracy: 0.0001)
  }
}
