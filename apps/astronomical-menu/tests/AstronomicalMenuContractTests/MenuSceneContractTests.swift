import Foundation
import XCTest

@testable import AstronomicalMenuCore

// Issue #710: an orphaned `Settings { EmptyView() }` scene opened a blank
// app-titled window on every launch. The menu applications own all interface
// through their NSApplicationDelegate (status-bar popover plus first-run
// welcome window); no SwiftUI scene belongs in either executable.
final class MenuSceneContractTests: XCTestCase {
  private let executableSourceDirectories = [
    "Sources/AstronomicalMenu",
    "Sources/AstronomicalMenuAppStore",
  ]

  private let forbiddenSceneDeclarations = [
    "Settings {",
    "WindowGroup",
    "MenuBarExtra",
  ]

  func test_should_declare_no_swiftui_scene_in_either_menu_executable() throws {
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
        sourceFileURLs.count, 0, "\(executableSourceDirectory) must still contain menu sources")

      for sourceFileURL in sourceFileURLs {
        let source = try String(contentsOf: sourceFileURL, encoding: .utf8)
        for sceneDeclaration in forbiddenSceneDeclarations {
          XCTAssertFalse(
            source.contains(sceneDeclaration),
            "\(sourceFileURL.lastPathComponent) must not declare SwiftUI scene \(sceneDeclaration). Menu interface belongs to the NSApplicationDelegate; a scene renders a window that auto-opens on launch (issue #710)."
          )
        }
      }
    }
  }
}
