import Foundation
import ThinTalkCanvas
import ThinTalkCore
import XCTest

/// The menu bar and toolbar reach the page only through `CanvasPageCommand`
/// names the page's `HostCommand` enum understands. A name that exists on one
/// side only is a menu item that silently does nothing, so both sides are read
/// here and must list exactly the same commands.
final class CanvasPageCommandContractTests: XCTestCase {
  private let packageDirectoryURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()

  private func pageCommandNames() throws -> Set<String> {
    let hostCommandSource = try String(
      contentsOf: packageDirectoryURL.appendingPathComponent("web/src/HostCommand.ts"), encoding: .utf8)
    let enumMemberPattern = try NSRegularExpression(pattern: #"^\s+[A-Z_]+ = "([A-Za-z]+)",$"#, options: [.anchorsMatchLines])
    let sourceRange = NSRange(hostCommandSource.startIndex..., in: hostCommandSource)
    var commandNames = Set<String>()
    for enumMemberMatch in enumMemberPattern.matches(in: hostCommandSource, range: sourceRange) {
      if let nameRange = Range(enumMemberMatch.range(at: 1), in: hostCommandSource) {
        commandNames.insert(String(hostCommandSource[nameRange]))
      }
    }
    return commandNames
  }

  func test_should_send_only_commands_the_page_understands_and_cover_every_one() throws {
    let nativeCommandNames = Set(CanvasPageCommand.allCases.map(\.rawValue))
      .union([CanvasPageCommand.titlebarInsetCommandName])
    let pageCommandNames = try pageCommandNames()
    XCTAssertFalse(pageCommandNames.isEmpty, "HostCommand.ts must still declare its commands")
    XCTAssertEqual(nativeCommandNames, pageCommandNames)
  }

  func test_should_deliver_each_command_through_the_page_router_without_failing_before_boot() {
    for pageCommand in CanvasPageCommand.allCases {
      XCTAssertEqual(
        pageCommand.scriptSource,
        "window.__thintalkCommands?.run({\"name\": \"\(pageCommand.rawValue)\"});")
    }
  }

  func test_should_restate_the_titlebar_inset_in_css_pixels_at_the_current_zoom() {
    let expectedPixelsByZoom: [(CGFloat, String)] = [(1.0, "52.0"), (0.5, "104.0"), (3.0, "17.33")]
    for (pageZoom, expectedPixels) in expectedPixelsByZoom {
      XCTAssertEqual(
        CanvasPageCommand.titlebarInsetScript(insetPoints: 52, pageZoom: pageZoom),
        "window.__thintalkCommands?.run({\"name\": \"setTitlebarInset\", \"points\": \(expectedPixels)});",
        "zoom \(pageZoom)")
    }
  }

  func test_should_treat_a_zoom_that_is_not_positive_as_actual_size() {
    for invalidPageZoom: CGFloat in [0, -1] {
      XCTAssertTrue(
        CanvasPageCommand.titlebarInsetScript(insetPoints: 52, pageZoom: invalidPageZoom)
          .contains("\"points\": 52.0"))
    }
  }

  func test_should_send_the_titlebar_inset_only_when_the_host_has_one() throws {
    let identity = ThinTalkApplicationIdentity(channel: .stable, daemonPort: 6732)
    let insetScript = try CanvasRuntimeConfig(applicationIdentity: identity, titlebarInset: 52)
      .userScriptSource()
    XCTAssertTrue(insetScript.contains("\"titlebarInset\":52"), insetScript)
    let plainScript = try CanvasRuntimeConfig(applicationIdentity: identity).userScriptSource()
    XCTAssertFalse(plainScript.contains("titlebarInset"), plainScript)
  }

  func test_should_tell_the_page_whether_the_host_can_restart_the_engine() throws {
    let identity = ThinTalkApplicationIdentity(channel: .development, daemonPort: 6733)
    let restartingScript = try CanvasRuntimeConfig(applicationIdentity: identity, canRestartEngine: true)
      .userScriptSource()
    XCTAssertTrue(restartingScript.contains("\"canRestartEngine\":true"), restartingScript)
    let plainScript = try CanvasRuntimeConfig(applicationIdentity: identity).userScriptSource()
    XCTAssertTrue(
      plainScript.contains("\"canRestartEngine\":false"), "a host without a restart says so: \(plainScript)")
  }
}
