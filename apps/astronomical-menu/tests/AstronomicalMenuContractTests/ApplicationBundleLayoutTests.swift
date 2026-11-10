import Foundation
import XCTest

@testable import AstronomicalShared

/// The agent is nested inside the main bundle, yet the engine and the
/// terminal command stay in the main bundle's Contents/MacOS. These tests pin
/// how each process finds the other's files.
final class ApplicationBundleLayoutTests: XCTestCase {
  private let mainBundlePath = "/Applications/Astronomical.app"
  private var agentBundlePath: String {
    mainBundlePath + "/Contents/Library/LoginItems/Astronomical Agent.app"
  }

  func test_should_find_the_main_bundles_executables_from_the_nested_agent() {
    let agentExecutableURL = URL(fileURLWithPath: agentBundlePath + "/Contents/MacOS/astronomical-menu")

    let executablesDirectoryURL = ApplicationBundleLayout.bundledExecutablesDirectory(
      forExecutableAt: agentExecutableURL)

    XCTAssertEqual(executablesDirectoryURL.path, mainBundlePath + "/Contents/MacOS")
  }

  func test_should_look_beside_an_executable_that_is_not_nested() {
    let mainExecutableURL = URL(fileURLWithPath: mainBundlePath + "/Contents/MacOS/astronomical-app")
    let bareExecutableURL = URL(fileURLWithPath: "/tmp/build/release/astronomical-menu")

    XCTAssertEqual(
      ApplicationBundleLayout.bundledExecutablesDirectory(forExecutableAt: mainExecutableURL).path,
      mainBundlePath + "/Contents/MacOS")
    XCTAssertEqual(
      ApplicationBundleLayout.bundledExecutablesDirectory(forExecutableAt: bareExecutableURL).path,
      "/tmp/build/release")
  }

  /// A bundle that itself sits under a folder named LoginItems must resolve to
  /// the innermost enclosing bundle, not to an outer folder of the same name.
  func test_should_use_the_innermost_login_items_folder() {
    let nestedExecutableURL = URL(
      fileURLWithPath:
        "/Volumes/Contents/Library/LoginItems/Astronomical.app/Contents/Library/LoginItems/Agent.app/Contents/MacOS/astronomical-menu"
    )

    XCTAssertEqual(
      ApplicationBundleLayout.bundledExecutablesDirectory(forExecutableAt: nestedExecutableURL).path,
      "/Volumes/Contents/Library/LoginItems/Astronomical.app/Contents/MacOS")
  }

  func test_should_name_the_main_bundle_that_contains_an_agent() {
    XCTAssertEqual(
      ApplicationBundleLayout.mainApplicationBundleURL(forAgentBundleAt: URL(fileURLWithPath: agentBundlePath))?
        .path,
      mainBundlePath)
  }

  func test_should_name_no_main_bundle_for_an_agent_that_is_not_nested() {
    XCTAssertNil(
      ApplicationBundleLayout.mainApplicationBundleURL(
        forAgentBundleAt: URL(fileURLWithPath: "/tmp/build/Astronomical Agent.app")))
    XCTAssertNil(
      ApplicationBundleLayout.mainApplicationBundleURL(forAgentBundleAt: URL(fileURLWithPath: "/Agent.app")))
  }

  func test_should_find_the_agent_bundle_inside_a_main_bundle_and_none_when_absent() throws {
    let scratchDirectoryURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("astronomical-layout-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: scratchDirectoryURL) }
    let bundleWithAgentURL = scratchDirectoryURL.appendingPathComponent("With.app", isDirectory: true)
    let bundleWithoutAgentURL = scratchDirectoryURL.appendingPathComponent("Without.app", isDirectory: true)
    let nestedAgentURL = bundleWithAgentURL
      .appendingPathComponent("Contents/Library/LoginItems/Astronomical Agent.app", isDirectory: true)
    try FileManager.default.createDirectory(at: nestedAgentURL, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: bundleWithoutAgentURL, withIntermediateDirectories: true)

    XCTAssertEqual(
      ApplicationBundleLayout.agentBundleURL(inApplicationBundleAt: bundleWithAgentURL)?.lastPathComponent,
      "Astronomical Agent.app")
    XCTAssertNil(ApplicationBundleLayout.agentBundleURL(inApplicationBundleAt: bundleWithoutAgentURL))
  }
}
