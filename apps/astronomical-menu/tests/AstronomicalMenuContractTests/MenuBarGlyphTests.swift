import AppKit
import XCTest

@testable import AstronomicalMenuCore

@MainActor
final class MenuBarGlyphTests: XCTestCase {
  private func statusDocument(
    status: String, activity: String, readyModel: String?, expertMemoryMode: String?
  ) throws -> DaemonStatusDocument {
    var fields = ["\"status\":\"\(status)\"", "\"activity\":\"\(activity)\""]
    if let readyModel { fields.append("\"ready_model_id\":\"\(readyModel)\"") }
    if let expertMemoryMode { fields.append("\"expert_memory_mode\":\"\(expertMemoryMode)\"") }
    fields.append(
      "\"serving_session\":{\"completed_request_count\":0,\"total_prompt_token_count\":0,\"total_reused_prompt_token_count\":0,\"target_prompt_work_token_count\":0,\"target_reused_prompt_work_token_count\":0,\"average_prefill_tok_per_second\":0,\"average_generation_tok_per_second\":0}"
    )
    return try JSONDecoder().decode(
      DaemonStatusDocument.self, from: Data("{\(fields.joined(separator: ","))}".utf8))
  }

  private func alphaSamples(of glyphImage: NSImage) throws -> [UInt8] {
    let bitmap = try XCTUnwrap(
      NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: 36, pixelsHigh: 36, bitsPerSample: 8,
        samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
        bytesPerRow: 36 * 4, bitsPerPixel: 32))
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
    glyphImage.draw(in: NSRect(x: 0, y: 0, width: 36, height: 36))
    NSGraphicsContext.restoreGraphicsState()
    let pixelBytes = try XCTUnwrap(bitmap.bitmapData)
    return (0..<(36 * 36)).map { pixelIndex in pixelBytes[pixelIndex * 4 + 3] }
  }

  func test_should_show_resident_when_a_dense_or_resident_model_is_ready() throws {
    XCTAssertEqual(
      try statusDocument(status: "ready", activity: "idle", readyModel: "dense", expertMemoryMode: nil)
        .menuBarGlyphState, .resident)
    XCTAssertEqual(
      try statusDocument(
        status: "ready", activity: "idle", readyModel: "moe", expertMemoryMode: "resident"
      ).menuBarGlyphState, .resident)
  }

  func test_should_show_streaming_when_experts_page_from_the_ssd() throws {
    for streamingMode in ["paged", "hybrid"] {
      XCTAssertEqual(
        try statusDocument(
          status: "ready", activity: "idle", readyModel: "moe", expertMemoryMode: streamingMode
        ).menuBarGlyphState, .streaming, streamingMode)
    }
  }

  private func denseStatusDocument(keptLayerCount: Int, streamedLayerPayloadBytes: Int) throws
    -> DaemonStatusDocument
  {
    let denseLayers =
      "{\"decoder_layer_count\":32,\"kept_layer_count\":\(keptLayerCount),\"streamed_layer_payload_bytes\":\(streamedLayerPayloadBytes)}"
    let expertResidency =
      "{\"total_layer_count\":0,\"resident_expert_count\":0,\"resident_expert_payload_bytes\":0,\"dense_layers\":\(denseLayers)}"
    let servingSession =
      "{\"completed_request_count\":0,\"total_prompt_token_count\":0,\"total_reused_prompt_token_count\":0,\"target_prompt_work_token_count\":0,\"target_reused_prompt_work_token_count\":0,\"average_prefill_tok_per_second\":0,\"average_generation_tok_per_second\":0}"
    let statusJSON =
      "{\"status\":\"ready\",\"activity\":\"idle\",\"ready_model_id\":\"dense\",\"expert_memory_mode\":\"resident\",\"expert_residency\":\(expertResidency),\"serving_session\":\(servingSession)}"
    return try JSONDecoder().decode(DaemonStatusDocument.self, from: Data(statusJSON.utf8))
  }

  func test_should_show_streaming_when_a_dense_model_reads_layers_from_the_ssd() throws {
    XCTAssertEqual(
      try denseStatusDocument(keptLayerCount: 7, streamedLayerPayloadBytes: 4_437_000_000)
        .menuBarGlyphState, .streaming)
    XCTAssertEqual(
      try denseStatusDocument(keptLayerCount: 0, streamedLayerPayloadBytes: 5_680_000_000)
        .menuBarGlyphState, .streaming)
  }

  func test_should_show_resident_when_a_dense_model_keeps_every_layer() throws {
    XCTAssertEqual(
      try denseStatusDocument(keptLayerCount: 32, streamedLayerPayloadBytes: 0).menuBarGlyphState,
      .resident)
  }

  func test_should_show_generating_over_residency_while_a_request_is_live() throws {
    XCTAssertEqual(
      try statusDocument(
        status: "ready", activity: "generating", readyModel: "moe", expertMemoryMode: "paged"
      ).menuBarGlyphState, .generating)
    XCTAssertEqual(
      try statusDocument(
        status: "ready", activity: "prompt_processing", readyModel: "dense", expertMemoryMode: nil
      ).menuBarGlyphState, .generating)
  }

  func test_should_show_idle_when_no_model_is_ready() throws {
    XCTAssertEqual(
      try statusDocument(status: "loading", activity: "idle", readyModel: "moe", expertMemoryMode: "paged")
        .menuBarGlyphState, .idle)
    XCTAssertEqual(
      try statusDocument(status: "ready", activity: "generating", readyModel: nil, expertMemoryMode: nil)
        .menuBarGlyphState, .idle)
    XCTAssertEqual(DaemonStatusDocument.unavailable.menuBarGlyphState, .idle)
  }

  // A model swap reports "ready" with no resident model until the load ends, so the glyph has to
  // read the load itself rather than the status beside it.
  func test_should_show_loading_over_every_status_while_a_load_is_in_flight() throws {
    let modelLoadInFlight = DaemonStatusDocument.ModelLoadInFlight(modelIdentifier: "second-model")
    for status in ["ready", "loading"] {
      let statusDocument = DaemonStatusDocument(
        status: status, activity: "idle", readyModelIdentifier: nil, progress: nil,
        modelLoadInFlight: modelLoadInFlight, expertMemoryMode: nil,
        mlxMemoryCeilingBytes: 0, servingSession: .empty)
      XCTAssertEqual(statusDocument.menuBarGlyphState, .loadingModel, status)
    }
  }

  func test_should_describe_the_loading_glyph_for_accessibility() {
    XCTAssertEqual(
      makeMenuBarGlyphImage(state: .loadingModel).accessibilityDescription,
      "Astronomical, loading a model")
  }

  func test_should_render_five_distinct_template_glyphs() throws {
    let glyphStates: [MenuBarGlyphState] = [
      .idle, .resident, .streaming, .generating, .loadingModel,
    ]
    var renderedAlphaSamples: [[UInt8]] = []
    for glyphState in glyphStates {
      let glyphImage = makeMenuBarGlyphImage(state: glyphState)
      XCTAssertTrue(glyphImage.isTemplate, "\(glyphState) must adapt to the menu bar appearance")
      XCTAssertEqual(glyphImage.size, NSSize(width: 18, height: 18))
      renderedAlphaSamples.append(try alphaSamples(of: glyphImage))
    }
    for (firstIndex, firstSamples) in renderedAlphaSamples.enumerated() {
      for secondIndex in (firstIndex + 1)..<renderedAlphaSamples.count {
        XCTAssertNotEqual(
          firstSamples, renderedAlphaSamples[secondIndex],
          "\(glyphStates[firstIndex]) and \(glyphStates[secondIndex]) render identically")
      }
    }
    let inkCoverage = renderedAlphaSamples.map { samples in samples.filter { $0 > 127 }.count }
    // Solid disc carries more ink than hatching, and hatching more than the empty plate.
    XCTAssertGreaterThan(inkCoverage[1], inkCoverage[2])
    XCTAssertGreaterThan(inkCoverage[2], inkCoverage[0])
  }
}
