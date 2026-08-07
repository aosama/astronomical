import AppKit
import SwiftUI
import XCTest

@testable import AstronomicalMenu

final class StatusPresentationContractTests: XCTestCase {
  func test_should_show_effective_mtp_runtime_state_for_every_status() throws {
    let runtimeStateCases: [(runtimeState: String, readyModelJSON: String, expectedTitle: String)] = [
      ("active", #", "ready_model_id":"Ornith""#, "Active"),
      ("target_only", #", "ready_model_id":"Ornith""#, "Standard generation"),
      ("unavailable", #", "ready_model_id":"Ornith""#, "Unavailable"),
      ("disabled", #", "ready_model_id":"Ornith""#, "Disabled"),
      ("disabled", "", "Not loaded"),
    ]

    for runtimeStateCase in runtimeStateCases {
      let statusDocument = try JSONDecoder().decode(
        SupervisorStatusDocument.self,
        from: Data(
          """
          {"status":"ready","activity":"idle","mtp_enabled":true,"mtp_runtime_state":"\(runtimeStateCase.runtimeState)"\(runtimeStateCase.readyModelJSON)}
          """.utf8)
      )

      XCTAssertEqual(
        statusDocument.mtpRuntimeStateTitle,
        runtimeStateCase.expectedTitle,
        "unexpected title for \(runtimeStateCase.runtimeState)"
      )
    }
  }

  func test_should_decode_the_bounded_mtp_unavailable_reason() throws {
    let statusDocument = try JSONDecoder().decode(
      SupervisorStatusDocument.self,
      from: Data(
        #"{"status":"ready","activity":"idle","ready_model_id":"Ornith","mtp_runtime_state":"unavailable","mtp_unavailable_reason":"MTP layer count does not match the model configuration"}"#.utf8)
    )

    XCTAssertEqual(
      statusDocument.mtpUnavailableReason,
      "MTP layer count does not match the model configuration"
    )
  }

  func test_should_use_the_colorblind_safe_mlx_memory_palette() throws {
    let expectedColorComponents: [(Color, CGFloat, CGFloat, CGFloat)] = [
      (MlxMemoryPalette.experts, 10, 132, 255),
      (MlxMemoryPalette.modelCore, 86, 180, 233),
      (MlxMemoryPalette.contextState, 240, 228, 66),
      (MlxMemoryPalette.runtimeWork, 167, 139, 250),
    ]

    for (paletteColor, expectedRed, expectedGreen, expectedBlue) in expectedColorComponents {
      let sRGBColor = try XCTUnwrap(NSColor(paletteColor).usingColorSpace(.sRGB))
      XCTAssertEqual(sRGBColor.redComponent * 255, expectedRed, accuracy: 0.5)
      XCTAssertEqual(sRGBColor.greenComponent * 255, expectedGreen, accuracy: 0.5)
      XCTAssertEqual(sRGBColor.blueComponent * 255, expectedBlue, accuracy: 0.5)
    }
  }

  func test_should_describe_each_mlx_memory_category_in_its_click_explanation() {
    XCTAssertEqual(
      MlxMemoryLegendItem.experts.explanationText,
      "Sparse MoE weights currently resident in MLX, including loaded expert pages."
    )
    XCTAssertEqual(
      MlxMemoryLegendItem.modelCore.explanationText,
      "Always-resident non-expert weights, including embeddings, attention, and vision weights."
    )
    XCTAssertEqual(
      MlxMemoryLegendItem.contextState.explanationText,
      "Decoder state for the active request, including conversation key-value state. It is released after completion and is separate from the client conversation window."
    )
    XCTAssertEqual(
      MlxMemoryLegendItem.runtimeWork.explanationText,
      "Temporary computation work and other active MLX memory not attributed above."
    )
    XCTAssertEqual(
      MlxMemoryLegendItem.available.explanationText,
      "Calculated capacity below this Mac's MLX ceiling. It is not free RAM; macOS memory pressure and temporary work can reduce what is safely usable."
    )
  }

  func test_should_name_each_mlx_memory_explanation_control() {
    XCTAssertEqual(MlxMemoryLegendItem.experts.infoButtonAccessibilityLabel, "Explain Experts")
    XCTAssertEqual(MlxMemoryLegendItem.modelCore.infoButtonAccessibilityLabel, "Explain Model core")
    XCTAssertEqual(MlxMemoryLegendItem.contextState.infoButtonAccessibilityLabel, "Explain Live context state")
    XCTAssertEqual(MlxMemoryLegendItem.runtimeWork.infoButtonAccessibilityLabel, "Explain Runtime work")
    XCTAssertEqual(
      MlxMemoryLegendItem.available.infoButtonAccessibilityLabel,
      "Explain Nominal MLX headroom"
    )
  }

  func test_should_match_the_available_memory_color_to_the_unused_utilization_track() throws {
    // Nominal headroom reuses the same translucent secondary track as the unused
    // portion of the GPU utilization and prompt-reuse bars.
    let availableAppearanceColor = try XCTUnwrap(NSColor(MlxMemoryPalette.available).usingColorSpace(.sRGB))
    let trackAppearanceColor = try XCTUnwrap(NSColor(Color.secondary.opacity(0.18)).usingColorSpace(.sRGB))
    XCTAssertEqual(availableAppearanceColor.redComponent, trackAppearanceColor.redComponent, accuracy: 0.001)
    XCTAssertEqual(availableAppearanceColor.greenComponent, trackAppearanceColor.greenComponent, accuracy: 0.001)
    XCTAssertEqual(availableAppearanceColor.blueComponent, trackAppearanceColor.blueComponent, accuracy: 0.001)
    XCTAssertEqual(availableAppearanceColor.alphaComponent, trackAppearanceColor.alphaComponent, accuracy: 0.001)
  }

  func test_should_match_the_visually_rendered_dark_background() throws {
    var resolvedActualBackgroundColor: NSColor?
    var resolvedExpectedBackgroundColor: NSColor?
    try XCTUnwrap(NSAppearance(named: .darkAqua)).performAsCurrentDrawingAppearance {
      resolvedActualBackgroundColor = NSColor(
        orbitalTelemetryBackgroundColor(colorScheme: ColorScheme.dark)
      ).usingColorSpace(.sRGB)
      resolvedExpectedBackgroundColor = NSColor(
        srgbRed: 24 / 255,
        green: 25 / 255,
        blue: 25 / 255,
        alpha: 1
      )
    }

    let actualBackgroundColor = try XCTUnwrap(resolvedActualBackgroundColor)
    let expectedBackgroundColor = try XCTUnwrap(resolvedExpectedBackgroundColor)
    XCTAssertEqual(actualBackgroundColor.redComponent, expectedBackgroundColor.redComponent, accuracy: 0.001)
    XCTAssertEqual(actualBackgroundColor.greenComponent, expectedBackgroundColor.greenComponent, accuracy: 0.001)
    XCTAssertEqual(actualBackgroundColor.blueComponent, expectedBackgroundColor.blueComponent, accuracy: 0.001)
    XCTAssertEqual(actualBackgroundColor.alphaComponent, expectedBackgroundColor.alphaComponent, accuracy: 0.001)
  }

  func test_should_freeze_the_status_item_width_only_while_the_popover_is_open() {
    XCTAssertEqual(
      menuBarStatusItemLength(popoverIsShown: true, currentButtonWidth: 180),
      180
    )
    XCTAssertEqual(
      menuBarStatusItemLength(popoverIsShown: false, currentButtonWidth: 180),
      NSStatusItem.variableLength
    )
  }

  func test_should_keep_the_status_item_title_stable_while_the_popover_is_open() {
    XCTAssertEqual(
      menuBarTitleToDisplay(
        currentTitle: "", latestTitle: "GEN 24.7 tok/s", popoverIsShown: true),
      ""
    )
    XCTAssertEqual(
      menuBarTitleToDisplay(
        currentTitle: "", latestTitle: "GEN 24.7 tok/s", popoverIsShown: false),
      "GEN 24.7 tok/s"
    )
  }

  func test_should_anchor_the_popover_to_the_status_item_icon() {
    let statusButtonBounds = CGRect(x: 0, y: 0, width: 180, height: 24)
    let statusItemImageRect = CGRect(x: 3, y: 3, width: 18, height: 18)

    XCTAssertEqual(
      popoverAnchorRect(
        statusButtonBounds: statusButtonBounds,
        statusItemImageRect: statusItemImageRect
      ),
      statusItemImageRect
    )
    XCTAssertEqual(
      popoverAnchorRect(statusButtonBounds: statusButtonBounds, statusItemImageRect: nil),
      statusButtonBounds
    )
  }

  func test_should_render_memory_using_decimal_gigabytes() {
    XCTAssertEqual(decimalGigabyteText(byteCount: 0), "Not measured")
    XCTAssertEqual(decimalGigabyteText(byteCount: 1_000_000_000), "1.00 GB")
    XCTAssertEqual(decimalGigabyteText(byteCount: 1_073_741_824), "1.07 GB")
    XCTAssertEqual(decimalGigabyteText(byteCount: 20_710_000_000), "20.71 GB")
    // The real `iogpu.wired_limit_mb=38912` value is 40_802_189_312 bytes.
    XCTAssertEqual(decimalGigabyteText(byteCount: 40_802_189_312), "40.80 GB")
    XCTAssertEqual(decimalGigabyteValueText(byteCount: 0), "0.00 GB")
  }

  func test_should_render_generation_rate_for_an_active_generation() throws {
    let statusDocument = try JSONDecoder().decode(
      SupervisorStatusDocument.self,
      from: Data(
        """
        {"status":"ready","activity":"generating","ready_model_id":"Ornith","ready_model_size_bytes":18420000000,"progress":{"phase":"generation","processed_tokens":27,"total_tokens":512,"elapsed_ms":1000},"expert_memory_mode":"paged","mlx_memory_snapshot":{"source":"decode_submitted","active_memory_bytes":12000000000,"allocator_cache_memory_bytes":2000000000,"peak_memory_bytes":14000000000,"expert_payload_bytes":5000000000,"model_core_payload_bytes":4000000000,"context_state_payload_bytes":1000000000},"mlx_memory_ceiling_bytes":40000000000,"serving_session":{"completed_request_count":4,"total_prompt_token_count":4096,"total_reused_prompt_token_count":2048,"average_prefill_tok_per_second":1000,"average_generation_tok_per_second":27.4}}
        """.utf8)
    )

    XCTAssertEqual(statusDocument.menuBarTitle, "GEN 27.0 tok/s")
    XCTAssertEqual(statusDocument.phaseTitle, "Generating")
    XCTAssertEqual(statusDocument.modelFootprintTitle, "RAM + SSD streaming")
    XCTAssertEqual(statusDocument.progressTitle, "27 / 512 tokens")
    XCTAssertEqual(statusDocument.elapsedTimeMetricTitle, "Elapsed")
    XCTAssertEqual(statusDocument.elapsedTimeTitle, "1.0 s")
    XCTAssertEqual(statusDocument.progressProcessedTokenCount, 27)
    XCTAssertEqual(statusDocument.progressTotalTokenCount, 512)
    XCTAssertEqual(statusDocument.mlxMemoryLimitTitle, "40.00 GB")
    XCTAssertEqual(statusDocument.sessionTitle, "4 requests · 27 tok/s avg")
    XCTAssertEqual(statusDocument.modelDiskSizeTitle, "18.42 GB")
    XCTAssertEqual(statusDocument.mlxMemoryBreakdown.expertPayloadByteCount, 5_000_000_000)
    XCTAssertEqual(statusDocument.mlxMemoryBreakdown.runtimeWorkByteCount, 2_000_000_000)
    XCTAssertEqual(statusDocument.mlxMemoryBreakdown.availableByteCount, 28_000_000_000)
    XCTAssertEqual(
      statusDocument.mlxMemoryBreakdown.expertPayloadByteCount
        + statusDocument.mlxMemoryBreakdown.modelCorePayloadByteCount
        + statusDocument.mlxMemoryBreakdown.contextStatePayloadByteCount
        + statusDocument.mlxMemoryBreakdown.runtimeWorkByteCount,
      statusDocument.mlxMemoryActiveBytes
    )
  }

  func test_should_label_each_worker_owned_mlx_memory_snapshot_source() throws {
    let expectedSourceTitles = [
      "model_loaded": "Model loaded",
      "prefill": "Prompt snapshot",
      "decode_submitted": "Live decode",
      "finalized": "After cleanup",
      "idle_poll": "Idle sample",
      "memory_limit_adjusted": "Memory limit adjusted",
    ]

    for (source, expectedSourceTitle) in expectedSourceTitles {
      let statusDocument = try JSONDecoder().decode(
        SupervisorStatusDocument.self,
        from: Data(
          """
          {"status":"ready","activity":"idle","mlx_memory_snapshot":{"source":"\(source)","active_memory_bytes":12000000000,"allocator_cache_memory_bytes":0,"peak_memory_bytes":12000000000,"expert_payload_bytes":5000000000,"model_core_payload_bytes":4000000000,"context_state_payload_bytes":1000000000},"mlx_memory_ceiling_bytes":40000000000}
          """.utf8)
      )

      XCTAssertEqual(statusDocument.mlxMemorySourceTitle, expectedSourceTitle)
      XCTAssertEqual(statusDocument.mlxMemoryActiveBytes, 12_000_000_000)
      XCTAssertEqual(statusDocument.mlxMemoryBreakdown.availableByteCount, 28_000_000_000)
    }
  }

  func test_should_keep_idle_progress_and_model_footprint_placeholders_visible() throws {
    let statusDocument = try JSONDecoder().decode(
      SupervisorStatusDocument.self,
      from: Data(
        """
        {"status":"ready","activity":"idle","ready_model_id":null,"progress":null,"expert_memory_mode":null,"mlx_memory_snapshot":null,"mlx_memory_ceiling_bytes":40000000000,"serving_session":{"completed_request_count":0,"total_prompt_token_count":0,"total_reused_prompt_token_count":0,"average_prefill_tok_per_second":0,"average_generation_tok_per_second":0}}
        """.utf8)
    )

    XCTAssertEqual(statusDocument.flightTitle, "Standing by")
    XCTAssertEqual(statusDocument.progressTitle, "Standing by")
    XCTAssertEqual(statusDocument.elapsedTimeTitle, "Not active")
    XCTAssertEqual(statusDocument.progressProcessedTokenCount, 0)
    XCTAssertEqual(statusDocument.progressTotalTokenCount, 1)
    XCTAssertEqual(statusDocument.modelFootprintTitle, "Not loaded")
    XCTAssertEqual(statusDocument.mlxMemoryLimitTitle, "40.00 GB")
  }

  func test_should_calculate_eta_after_the_first_progress_token() throws {
    let statusDocument = try JSONDecoder().decode(
      SupervisorStatusDocument.self,
      from: Data(
        #"{"status":"ready","activity":"prompt_processing","progress":{"phase":"prefill","processed_tokens":0,"total_tokens":14412,"elapsed_ms":250}}"#.utf8
      )
    )

    XCTAssertEqual(
      statusDocument.elapsedTimeTitle,
      "0.2 s / Calculating"
    )
    XCTAssertEqual(statusDocument.flightTitle, "Prompt processing")
  }

  func test_should_use_plain_language_for_prompt_processing_rate() throws {
    let statusDocument = try JSONDecoder().decode(
      SupervisorStatusDocument.self,
      from: Data(
        """
        {"status":"ready","activity":"prompt_processing","ready_model_id":"Ornith","progress":{"phase":"prefill","processed_tokens":1076,"total_tokens":14412,"elapsed_ms":1000},"expert_memory_mode":"resident","mlx_memory_snapshot":{"source":"prefill","active_memory_bytes":12000000000,"allocator_cache_memory_bytes":2000000000,"peak_memory_bytes":14000000000,"expert_payload_bytes":0,"model_core_payload_bytes":0,"context_state_payload_bytes":0},"mlx_memory_ceiling_bytes":40000000000,"serving_session":{"completed_request_count":0,"total_prompt_token_count":0,"total_reused_prompt_token_count":0,"average_prefill_tok_per_second":0,"average_generation_tok_per_second":0}}
        """.utf8)
    )

    XCTAssertEqual(statusDocument.menuBarTitle, "PP 1076 tok/s")
    XCTAssertEqual(statusDocument.flightTitle, "Prompt processing · 1076 tok/s")
    XCTAssertEqual(statusDocument.elapsedTimeMetricTitle, "Elapsed / ETA")
    XCTAssertEqual(statusDocument.elapsedTimeTitle, "1.0 s / 12.4 s")
  }

  func test_should_use_singular_request_grammar_for_one_completed_request() throws {
    let statusDocument = try JSONDecoder().decode(
      SupervisorStatusDocument.self,
      from: Data(
        """
        {"status":"ready","activity":"idle","ready_model_id":"Ornith","progress":null,"expert_memory_mode":"resident","mlx_memory_snapshot":{"source":"finalized","active_memory_bytes":12000000000,"allocator_cache_memory_bytes":2000000000,"peak_memory_bytes":14000000000,"expert_payload_bytes":0,"model_core_payload_bytes":0,"context_state_payload_bytes":0},"mlx_memory_ceiling_bytes":40000000000,"serving_session":{"completed_request_count":1,"total_prompt_token_count":4096,"total_reused_prompt_token_count":2048,"average_prefill_tok_per_second":1000,"average_generation_tok_per_second":70.4}}
        """.utf8)
    )

    XCTAssertEqual(statusDocument.sessionTitle, "1 request · 70 tok/s avg")
  }

  func test_should_explain_how_much_of_the_session_prompts_were_reused_and_new() throws {
    let statusDocument = try JSONDecoder().decode(
      SupervisorStatusDocument.self,
      from: Data(
        """
        {"status":"ready","activity":"idle","serving_session":{"completed_request_count":3,"total_prompt_token_count":20637,"total_reused_prompt_token_count":16384,"average_prefill_tok_per_second":1000,"average_generation_tok_per_second":70.4}}
        """.utf8)
    )

    XCTAssertEqual(
      statusDocument.sessionPromptReusePercentageTitle,
      "79.3%"
    )
    XCTAssertEqual(
      statusDocument.sessionPromptReuseBreakdownTitle,
      "16,384 reused · 4,253 new"
    )
    XCTAssertEqual(statusDocument.sessionPromptReuseFraction, 16_384.0 / 20_637.0)
  }

  func test_should_distinguish_loading_from_unavailable() throws {
    let statusDocument = try JSONDecoder().decode(
      SupervisorStatusDocument.self,
      from: Data(#"{"status":"loading","activity":"idle"}"#.utf8)
    )

    XCTAssertEqual(statusDocument.phaseTitle, "Loading")
    XCTAssertEqual(statusDocument.menuBarTitle, " Loading")
  }

  func test_should_not_round_partial_prompt_reuse_up_to_one_hundred_percent() {
    XCTAssertEqual(
      promptReusePercentageText(
        reusedPromptTokenCount: 73_728,
        totalPromptTokenCount: 73_945
      ),
      "99.7%"
    )
    XCTAssertEqual(
      promptReusePercentageText(
        reusedPromptTokenCount: 4_000,
        totalPromptTokenCount: 4_000
      ),
      "100%"
    )
    XCTAssertEqual(
      promptReusePercentageText(
        reusedPromptTokenCount: UInt64.max - 1,
        totalPromptTokenCount: UInt64.max
      ),
      "99.9%"
    )
  }
}
