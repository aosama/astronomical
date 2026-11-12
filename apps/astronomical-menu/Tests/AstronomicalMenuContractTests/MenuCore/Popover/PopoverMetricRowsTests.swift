// Contract for `PopoverProgressRow`: in every phase, with or without its bar,
// the Progress row is exactly as tall as a `PopoverMetricRow`, so the popover
// shows no empty band between the Flight and Progress rows and nothing below
// them moves between phases.

import AppKit
import SwiftUI
import XCTest

@testable import AstronomicalMenuCore

@MainActor
final class PopoverMetricRowsTests: XCTestCase {
  private func statusDocument(_ statusJson: String) throws -> DaemonStatusDocument {
    try JSONDecoder().decode(DaemonStatusDocument.self, from: Data(statusJson.utf8))
  }

  private func tokenProgressStatus(phase: String, processedTokens: UInt32, totalTokens: UInt32) throws
    -> DaemonStatusDocument
  {
    try statusDocument(
      """
      {"status":"ready","activity":"generating","ready_model_id":"Ornith-9B","progress":{"phase":"\(phase)","processed_tokens":\(processedTokens),"total_tokens":\(totalTokens),"elapsed_ms":1000}}
      """)
  }

  private func imageStatus(phase: String) throws -> DaemonStatusDocument {
    try statusDocument(
      """
      {"status":"ready","activity":"image_generation","ready_model_id":"fictional/flux-model","progress":{"phase":"\(phase)","completed_steps":2,"total_steps":4,"elapsed_ms":1000}}
      """)
  }

  /// Measures a row the way the popover lays it out: its content width and
  /// its body font, which follows the system text size.
  private func fittedHeight<Row: View>(_ row: Row) -> CGFloat {
    let laidOutRow = row.font(PopoverTypography.body).frame(width: PopoverLayout.contentWidth)
    return NSHostingView(rootView: laidOutRow).fittingSize.height
  }

  private func assertOneMetricLineTall(
    _ document: DaemonStatusDocument, file: StaticString = #filePath, line: UInt = #line
  ) {
    let metricLineHeight = fittedHeight(
      PopoverMetricRow(metricLabel: "Flight", metricText: "Generating · 9.2 tok/s"))
    XCTAssertEqual(
      fittedHeight(PopoverProgressRow(statusDocument: document)), metricLineHeight, file: file, line: line)
  }

  func test_should_keep_one_line_height_while_drawing_the_bar_for_a_capped_generation() throws {
    let cappedGeneration = try tokenProgressStatus(phase: "generation", processedTokens: 77, totalTokens: 512)
    XCTAssertTrue(cappedGeneration.showsProgressBar)
    assertOneMetricLineTall(cappedGeneration)
  }

  func test_should_keep_one_line_height_without_a_bar_for_an_uncapped_generation() throws {
    let uncappedGeneration = try tokenProgressStatus(
      phase: "generation", processedTokens: 77, totalTokens: 65_535)
    XCTAssertFalse(uncappedGeneration.showsProgressBar)
    assertOneMetricLineTall(uncappedGeneration)
  }

  func test_should_keep_a_long_prefill_title_on_one_line_beside_the_bar() throws {
    let longPrefill = try tokenProgressStatus(phase: "prefill", processedTokens: 54_321, totalTokens: 98_765)
    XCTAssertTrue(longPrefill.showsProgressBar)
    XCTAssertEqual(longPrefill.progressTitle, "55% · 54321 / 98765 tokens")
    assertOneMetricLineTall(longPrefill)
  }

  func test_should_truncate_rather_than_wrap_a_title_wider_than_the_row() throws {
    let widestPrefill = try tokenProgressStatus(
      phase: "prefill", processedTokens: UInt32.max - 1, totalTokens: UInt32.max)
    XCTAssertEqual(widestPrefill.progressTitle, "99% · 4294967294 / 4294967295 tokens")
    assertOneMetricLineTall(widestPrefill)
  }

  func test_should_keep_one_line_height_while_idle_with_an_empty_bar() throws {
    let idleStatus = try statusDocument(
      #"{"status":"ready","activity":"idle","ready_model_id":null,"progress":null}"#)
    XCTAssertTrue(idleStatus.showsProgressBar)
    assertOneMetricLineTall(idleStatus)
  }

  func test_should_keep_one_line_height_without_a_bar_during_a_model_load() throws {
    let modelLoad = try statusDocument(
      #"{"status":"ready","activity":"idle","model_load_in_flight":{"model_id":"Ornith"}}"#)
    XCTAssertFalse(modelLoad.showsProgressBar)
    assertOneMetricLineTall(modelLoad)
  }

  func test_should_keep_one_line_height_without_a_bar_during_a_prompt_cache_restore() throws {
    let promptCacheRestore = try statusDocument(#"{"status":"ready","activity":"prompt_processing"}"#)
    XCTAssertFalse(promptCacheRestore.showsProgressBar)
    assertOneMetricLineTall(promptCacheRestore)
  }

  func test_should_keep_one_line_height_while_drawing_the_bar_for_image_denoising() throws {
    let denoising = try imageStatus(phase: "denoising")
    XCTAssertTrue(denoising.showsProgressBar)
    assertOneMetricLineTall(denoising)
  }

  func test_should_keep_one_line_height_without_a_bar_while_decoding_an_image() throws {
    let decoding = try imageStatus(phase: "decoding")
    XCTAssertFalse(decoding.showsProgressBar)
    assertOneMetricLineTall(decoding)
  }
}

/// The line under the build identity names the configuration file and the port.
final class ConfigurationLocationTitleTests: XCTestCase {
  func test_should_show_a_four_digit_port_without_a_thousands_separator() {
    let title = configurationLocationTitle(testApplicationIdentity(channel: .stable))

    XCTAssertTrue(title.hasSuffix("Port 6732"), title)
    XCTAssertFalse(title.contains(","), title)
  }

  func test_should_show_the_largest_port_without_a_thousands_separator() {
    let highPortIdentity = ApplicationIdentity(
      channel: .development, daemonPort: 65_535, stateDirectoryName: ".astronomical-dev",
      version: "1.0.0", buildNumber: "1", commit: "abcdef0", isDirty: false)

    XCTAssertTrue(configurationLocationTitle(highPortIdentity).hasSuffix("Port 65535"))
  }

  func test_should_name_each_channels_own_configuration_file() {
    let stableTitle = configurationLocationTitle(testApplicationIdentity(channel: .stable))
    let developmentTitle = configurationLocationTitle(testApplicationIdentity(channel: .development))

    XCTAssertNotEqual(stableTitle, developmentTitle)
    XCTAssertTrue(developmentTitle.contains("astronomical-dev"), developmentTitle)
  }
}
