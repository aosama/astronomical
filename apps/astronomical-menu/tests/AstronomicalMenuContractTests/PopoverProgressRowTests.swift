// Contract for `PopoverProgressRow`: whether or not the bar is drawn, the row
// is exactly one metric line tall, so the popover shows no empty band between
// the Flight and Progress rows and nothing below them moves between phases.

import AppKit
import SwiftUI
import XCTest

@testable import AstronomicalMenuCore

@MainActor
final class PopoverProgressRowTests: XCTestCase {
  // The popover is 390pt wide with 16pt padding on each side.
  private let popoverContentWidth: CGFloat = 358

  private func statusDocument(_ statusJson: String) throws -> DaemonStatusDocument {
    try JSONDecoder().decode(DaemonStatusDocument.self, from: Data(statusJson.utf8))
  }

  private func progressStatus(phase: String, processedTokens: Int, totalTokens: Int) throws
    -> DaemonStatusDocument
  {
    try statusDocument(
      """
      {"status":"ready","activity":"generating","ready_model_id":"Ornith-9B","progress":{"phase":"\(phase)","processed_tokens":\(processedTokens),"total_tokens":\(totalTokens),"elapsed_ms":1000}}
      """)
  }

  private func fittedHeight<Row: View>(_ row: Row) -> CGFloat {
    NSHostingView(rootView: row.frame(width: popoverContentWidth)).fittingSize.height
  }

  /// The height of a plain metric line, built the way the popover's other
  /// rows are, which the Progress row must match.
  private func metricLineHeight() -> CGFloat {
    fittedHeight(
      HStack {
        Text("Flight").foregroundStyle(.secondary)
        Spacer()
        Text("Generating · 9.2 tok/s").font(PopoverTypography.monospacedBody)
      })
  }

  func test_should_keep_one_line_height_while_drawing_the_bar_for_a_capped_generation() throws {
    let cappedGeneration = try progressStatus(phase: "generation", processedTokens: 77, totalTokens: 512)
    XCTAssertTrue(cappedGeneration.showsProgressBar)

    XCTAssertEqual(fittedHeight(PopoverProgressRow(statusDocument: cappedGeneration)), metricLineHeight())
  }

  func test_should_keep_one_line_height_without_a_bar_for_an_uncapped_generation() throws {
    let uncappedGeneration = try progressStatus(
      phase: "generation", processedTokens: 77, totalTokens: 65_535)
    XCTAssertFalse(uncappedGeneration.showsProgressBar)

    XCTAssertEqual(fittedHeight(PopoverProgressRow(statusDocument: uncappedGeneration)), metricLineHeight())
  }

  func test_should_keep_the_longest_prefill_title_on_one_line_beside_the_bar() throws {
    let longPrefill = try progressStatus(phase: "prefill", processedTokens: 54_321, totalTokens: 98_765)
    XCTAssertTrue(longPrefill.showsProgressBar)
    XCTAssertEqual(longPrefill.progressTitle, "55% · 54321 / 98765 tokens")

    XCTAssertEqual(fittedHeight(PopoverProgressRow(statusDocument: longPrefill)), metricLineHeight())
  }

  func test_should_keep_one_line_height_while_idle_with_an_empty_bar() throws {
    let idle = try statusDocument(#"{"status":"ready","activity":"idle","ready_model_id":null,"progress":null}"#)
    XCTAssertTrue(idle.showsProgressBar)

    XCTAssertEqual(fittedHeight(PopoverProgressRow(statusDocument: idle)), metricLineHeight())
  }
}
