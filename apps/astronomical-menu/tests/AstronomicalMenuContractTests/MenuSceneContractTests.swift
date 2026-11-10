import Foundation
import XCTest

@testable import AstronomicalMenuCore

// Issue #710: an orphaned `Settings { EmptyView() }` scene opened a blank
// app-titled window on every launch. Every executable owns its interface
// through its NSApplicationDelegate (the agent's status-bar popover, the main
// app's chat window); no SwiftUI scene belongs in any of them.
final class MenuSceneContractTests: XCTestCase {
  private let executableSourceDirectories = [
    "Sources/AstronomicalMenu",
    "Sources/AstronomicalApp",
    "Sources/AstronomicalAppAppStore",
  ]

  private let forbiddenSceneDeclarations = [
    "Settings {",
    "WindowGroup",
    "MenuBarExtra",
  ]

  func test_should_declare_no_swiftui_scene_in_any_executable() throws {
    let packageDirectoryURL = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()

    for executableSourceDirectory in executableSourceDirectories {
      let directoryURL = packageDirectoryURL.appendingPathComponent(executableSourceDirectory)
      let sourceFileURLs = try FileManager.default
        .contentsOfDirectory(at: directoryURL, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "swift" }

      XCTAssertGreaterThan(
        sourceFileURLs.count, 0, "\(executableSourceDirectory) must still contain its entry point")

      for sourceFileURL in sourceFileURLs {
        let source = try String(contentsOf: sourceFileURL, encoding: .utf8)
        for sceneDeclaration in forbiddenSceneDeclarations {
          XCTAssertFalse(
            source.contains(sceneDeclaration),
            "\(sourceFileURL.lastPathComponent) must not declare SwiftUI scene \(sceneDeclaration). Interface belongs to the NSApplicationDelegate; a scene renders a window that auto-opens on launch (issue #710)."
          )
        }
      }
    }
  }
}
