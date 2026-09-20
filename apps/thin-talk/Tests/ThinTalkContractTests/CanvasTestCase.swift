import Foundation
import ThinTalkCanvas
import XCTest

/// Shared harness for the suites that drive the real conversation canvas.
///
/// The canvas suites are split by concern — answers and interactions in
/// CanvasRenderingTests, maths and diagrams in CanvasMathAndDiagramTests — and they
/// share this setup so one shell change cannot fix one suite while silently
/// breaking the other. Every wait is bounded, so no case can hang if the page
/// misbehaves.
@MainActor
class CanvasTestCase: XCTestCase {
  var harness: CanvasHarness?

  override func setUp() async throws {
    try await super.setUp()
    let webDirectory = try XCTUnwrap(
      CanvasWebResources.bundledDirectory(),
      "the canvas shell must ship inside the target's resource bundle"
    )
    let harness = CanvasHarness(webDirectory: webDirectory)
    try await harness.start()
    self.harness = harness
  }

  override func tearDown() async throws {
    harness = nil
    try await super.tearDown()
  }
}