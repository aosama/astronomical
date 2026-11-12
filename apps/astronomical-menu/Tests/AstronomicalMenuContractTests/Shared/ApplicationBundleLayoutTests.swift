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

  /// launchd starts a login item with its service label as the first launch
  /// argument. Read as a path that label names a file in `/`, which is how
  /// the agent lost the engine after every login.
  func test_should_find_the_engine_directory_when_launchd_passes_a_service_label() throws {
    let agentExecutableURL = try XCTUnwrap(
      ApplicationBundleLayout.runningExecutableURL(
        bundleExecutableURL: URL(fileURLWithPath: agentBundlePath + "/Contents/MacOS/astronomical-menu"),
        launchArguments: ["dev.astronomical.app.agent"]))

    XCTAssertEqual(agentExecutableURL.path, agentBundlePath + "/Contents/MacOS/astronomical-menu")
    XCTAssertEqual(
      ApplicationBundleLayout.bundledExecutablesDirectory(forExecutableAt: agentExecutableURL).path,
      mainBundlePath + "/Contents/MacOS")
  }

  func test_should_prefer_the_bundle_executable_over_an_absolute_launch_argument() {
    XCTAssertEqual(
      ApplicationBundleLayout.runningExecutableURL(
        bundleExecutableURL: URL(fileURLWithPath: "/opt/loaded/astronomical-menu"),
        launchArguments: ["/opt/claimed/astronomical-menu"])?.path,
      "/opt/loaded/astronomical-menu")
  }

  func test_should_use_an_absolute_launch_argument_when_the_bundle_names_no_executable() {
    XCTAssertEqual(
      ApplicationBundleLayout.runningExecutableURL(
        bundleExecutableURL: nil, launchArguments: ["/opt/build/../build/astronomical-menu", "--flag"])?.path,
      "/opt/build/astronomical-menu")
  }

  func test_should_name_no_executable_from_a_label_a_relative_path_or_no_arguments() {
    for unusableLaunchArguments in [["dev.astronomical.app.agent"], ["./astronomical-menu"], [""], []] {
      XCTAssertNil(
        ApplicationBundleLayout.runningExecutableURL(
          bundleExecutableURL: nil, launchArguments: unusableLaunchArguments),
        "\(unusableLaunchArguments) must not be read as the running executable")
    }
  }

  func test_should_resolve_a_symbolic_link_to_the_executable_it_points_at() throws {
    let scratchDirectoryURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("astronomical-layout-\(UUID().uuidString)", isDirectory: true)
      .resolvingSymlinksInPath()
    defer { try? FileManager.default.removeItem(at: scratchDirectoryURL) }
    try FileManager.default.createDirectory(at: scratchDirectoryURL, withIntermediateDirectories: true)
    let executableURL = scratchDirectoryURL.appendingPathComponent("astronomical-menu")
    let linkURL = scratchDirectoryURL.appendingPathComponent("linked-menu")
    XCTAssertTrue(FileManager.default.createFile(atPath: executableURL.path, contents: Data()))
    try FileManager.default.createSymbolicLink(at: linkURL, withDestinationURL: executableURL)

    XCTAssertEqual(
      ApplicationBundleLayout.runningExecutableURL(bundleExecutableURL: linkURL, launchArguments: [])?.path,
      executableURL.path)
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
