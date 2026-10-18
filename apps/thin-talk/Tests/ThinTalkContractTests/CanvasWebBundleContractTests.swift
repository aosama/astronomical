import Foundation
import XCTest

// The conversation canvas boots from bundle.js/bundle.css, which
// apps/thin-talk/web/build-web.mjs generates and the root .gitignore excludes.
// SwiftPM copies Resources/web verbatim, so on a fresh clone every target
// still builds and the app opens a silently blank conversation. This contract
// turns that silent failure into a failing test with the recovery command.
final class CanvasWebBundleContractTests: XCTestCase {
  private let packageDirectoryURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()

  func test_generated_web_client_bundle_exists_before_the_canvas_ships() throws {
    let webResourcesURL = packageDirectoryURL
      .appendingPathComponent("Sources/ThinTalkCanvas/Resources/web")

    for generatedClientAsset in ["bundle.js", "bundle.css"] {
      XCTAssertTrue(
        FileManager.default.fileExists(
          atPath: webResourcesURL.appendingPathComponent(generatedClientAsset).path),
        "\(generatedClientAsset) is missing: the web client is generated and never committed."
          + " Build it once with: (cd apps/thin-talk/web && npm install && npm run build)"
      )
    }
  }
}
