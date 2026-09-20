import Foundation
import ThinTalkCanvas
import ThinTalkCore
import XCTest

/// Maths and diagrams in the real conversation canvas.
///
/// These two modalities take a different road from markdown — their output is
/// generated per span and grafted after sanitisation — so they are asserted on
/// what a reader receives: real typeset output, node labels that stay readable, and
/// no raw TeX, placeholder tokens, or executing markup anywhere in the result.
@MainActor
final class CanvasMathAndDiagramTests: CanvasTestCase {
  func test_should_load_katex_and_mermaid_with_the_shell() async throws {
    let harness = try XCTUnwrap(harness)
    let diagnostics = try await harness.pageDiagnostics()
    XCTAssertTrue(diagnostics.contains("\"katex\":\"object\""), diagnostics)
    // mermaid exposes itself on globalThis, which surfaces as "object".
    XCTAssertTrue(diagnostics.contains("\"mermaid\":\"object\""), diagnostics)
  }

  func test_should_render_inline_and_display_math_through_katex() async throws {
    let harness = try XCTUnwrap(harness)
    let message = TranscriptMessage(
      id: UUID(),
      role: .assistant,
      markdown: """
        Attention weighted sum: $\\sum_i v_i \\text{softmax}(q \\cdot k_i)$, and

        $$\\frac{\\Phi}{\\Theta} = \\int_0^1 x^2 \\,\\mathrm{d}x$$
        """,
      state: .complete)
    try await harness.send(
      .snapshot(TranscriptSnapshot(messages: [message], channel: "Development")))
    try await harness.waitForElement("article .katex-display .katex")

    let inlineMath = try await harness.count(of: "article .markdown-body .katex")
    XCTAssertGreaterThanOrEqual(inlineMath, 2, "both an inline and a display span must render")
    // KaTeX keeps the TeX source in a clipped MathML <annotation> for assistive
    // technology, so the reader-facing text is what must be free of maths syntax
    // while the token pipeline must not leave a placeholder behind.
    let answerText = try await harness.text(of: "article .markdown-body")
    XCTAssertFalse(answerText.contains("%%MATH"), "a placeholder token must never be visible")
    XCTAssertFalse(answerText.contains("$$"), "display delimiters must be consumed, not printed")
    let mathmlAnnotations = try await harness.count(of: "article .katex annotation")
    XCTAssertGreaterThanOrEqual(
      mathmlAnnotations, 1, "the rendered maths must carry real MathML semantics")
  }

  func test_should_not_render_currency_or_inline_code_as_math() async throws {
    let harness = try XCTUnwrap(harness)
    let message = TranscriptMessage(
      id: UUID(),
      role: .assistant,
      markdown: "Priced at $5-$10 per seat, configured by `let budget = $money`.",
      state: .complete)
    try await harness.send(
      .snapshot(TranscriptSnapshot(messages: [message], channel: "Development")))
    try await harness.waitForParagraph(
      "Priced at $5-$10 per seat, configured by let budget = $money.")

    let mathNodes = try await harness.count(of: "article .katex")
    XCTAssertEqual(
      mathNodes, 0, "money and code that merely contain dollar signs must stay text")
  }

  func test_should_render_a_mermaid_fence_into_a_sanitised_figure_when_complete() async throws {
    let harness = try XCTUnwrap(harness)
    let message = TranscriptMessage(
      id: UUID(),
      role: .assistant,
      markdown: """
        ```mermaid
        flowchart LR
          prefill["Prefill"] --> decode["Decode"]
        ```
        """,
      state: .complete)
    try await harness.send(
      .snapshot(TranscriptSnapshot(messages: [message], channel: "Development")))
    try await harness.waitForElement("article figure.diagram svg")

    let figureCount = try await harness.count(of: "article .code-block--diagram")
    XCTAssertEqual(figureCount, 1, "the fenced block becomes exactly one figure")
    let leftoverCode = try await harness.count(of: "article .code-block--diagram pre")
    XCTAssertEqual(leftoverCode, 0, "the source block is replaced by the figure")
    // Mermaid ships its theme as an SVG <style> block; without it the labels are
    // colourless and invisible over the node fills, so its survival is the check
    // that the diagram is readable rather than merely present.
    let themeStyle = try await harness.count(of: "article figure.diagram svg style")
    XCTAssertGreaterThanOrEqual(themeStyle, 1, "the diagram theme CSS must survive sanitisation")
    let diagramText = try await harness.text(of: "article figure.diagram")
    XCTAssertTrue(
      diagramText.contains("Prefill") || diagramText.contains("prefill"),
      "node labels must be present as text, found: \(diagramText)")
    let foreignObjects = try await harness.count(of: "article figure.diagram foreignObject")
    XCTAssertEqual(
      foreignObjects, 0,
      "labels must be native SVG text, not HTML smuggled through foreignObject")
    let executed = try await harness.bool(of: "window.__thintalkPwned !== undefined")
    XCTAssertFalse(executed)
  }

  func test_should_skip_diagrams_while_streaming_then_render_once_finished() async throws {
    let harness = try XCTUnwrap(harness)
    let messageID = UUID()
    let fencedDiagram = "```mermaid\nflowchart LR\n  a --> b\n```"
    let streaming = TranscriptMessage(
      id: messageID, role: .assistant, markdown: fencedDiagram, state: .streaming)
    try await harness.send(
      .snapshot(TranscriptSnapshot(messages: [streaming], channel: "Development")))
    try await harness.waitForElement("article pre code.language-mermaid")

    for _ in 0..<3 {
      try await Task.sleep(nanoseconds: 200_000_000)
      let streamingFigures = try await harness.count(of: "article figure.diagram svg")
      XCTAssertEqual(
        streamingFigures, 0, "rendering mermaid while streaming would thrash the CPU")
    }

    let finished = TranscriptMessage(
      id: messageID, role: .assistant, markdown: fencedDiagram, state: .complete)
    try await harness.send(.message(finished))
    try await harness.waitForElement("article figure.diagram svg")
  }
}