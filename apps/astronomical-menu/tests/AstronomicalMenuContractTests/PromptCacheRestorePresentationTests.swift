// Journey: a request is already active, but the first prefill progress event has
// not arrived yet. The menu extra and popover must name prompt-cache restore
// instead of looking idle. Prompt and GEN titles after progress exists stay in
// StatusPresentationContractTests.

import XCTest

@testable import AstronomicalMenuCore

final class PromptCacheRestorePresentationTests: XCTestCase {
  func test_should_name_prompt_processing_without_progress_as_prompt_cache_restore() throws {
    let statusDocument = try JSONDecoder().decode(
      SupervisorStatusDocument.self,
      from: Data(#"{"status":"ready","activity":"prompt_processing"}"#.utf8)
    )

    XCTAssertEqual(statusDocument.menuBarTitle, "Restoring…")
    XCTAssertEqual(statusDocument.phaseTitle, "Restoring prompt cache")
    XCTAssertEqual(statusDocument.flightTitle, "Restoring prompt cache")
    XCTAssertEqual(statusDocument.progressTitle, "Restoring prompt cache")
    XCTAssertFalse(
      statusDocument.hasDeterminateProgress,
      "restore has no token totals yet, so the popover must not draw a 0% bar"
    )
    XCTAssertEqual(statusDocument.elapsedTimeTitle, "Calculating")
  }
}
