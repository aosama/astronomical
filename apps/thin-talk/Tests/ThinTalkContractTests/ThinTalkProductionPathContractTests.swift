import Foundation
import XCTest

// Issue #713: the ThinTalk production path must never depend on the debug-only
// UX preview scaffold. Two regressions are guarded here:
//
// 1. The shipped app entry must always run the real, client-backed RootView.
//    It must not route to PreviewRootView, must not read the
//    THINTALK_RENDER_PNG environment flag, and must never exit the process
//    from inside a Scene task.
// 2. The production chat views must not build interface from PreviewData or
//    PreviewSidebar; those types exist only for the debug preview scaffold.
final class ThinTalkProductionPathContractTests: XCTestCase {
  private let packageDirectoryURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()

  func test_should_run_only_the_production_root_from_the_app_entry() throws {
    let source = try String(
      contentsOf: packageDirectoryURL
        .appendingPathComponent("Sources/ThinTalk/ThinTalkApp.swift"),
      encoding: .utf8
    )

    for forbidden in ["PreviewRootView", "THINTALK_RENDER_PNG", "exit(0)"] {
      XCTAssertFalse(
        source.contains(forbidden),
        "ThinTalkApp.swift must not reference \(forbidden). The shipped entry renders RootView only; preview rendering and process exit belong to the debug preview scaffold, never to the app entry (issue #713)."
      )
    }
    XCTAssertTrue(
      source.contains("RootView()"),
      "ThinTalkApp.swift must render RootView in the shipped WindowGroup (issue #713)."
    )
  }

  func test_should_build_production_views_without_preview_mock_data() throws {
    let source = try String(
      contentsOf: packageDirectoryURL
        .appendingPathComponent("Sources/ThinTalk/Views.swift"),
      encoding: .utf8
    )

    for forbidden in ["PreviewData", "PreviewSidebar"] {
      XCTAssertFalse(
        source.contains(forbidden),
        "Views.swift must not reference \(forbidden). Production views never render mock conversation data; until a real session store exists the chat surface stays single-pane (issue #713)."
      )
    }
  }
}
