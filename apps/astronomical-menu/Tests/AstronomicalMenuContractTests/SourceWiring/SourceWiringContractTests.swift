import Foundation
import XCTest

@testable import AstronomicalAppCore
@testable import AstronomicalMenuCore

/// Every contract that can only be stated about the source text lives here.
///
/// Some wiring has no behaviour a unit test can reach: which delegate method
/// registers an Apple-event handler, whether an executable declares a SwiftUI
/// scene, whether xtask and the app spell one launch argument alike. These
/// tests read the source for exactly those facts. They are kept in one file
/// so the pattern stays rare and visible; a behaviour that can be exercised
/// belongs in an ordinary test beside its subject instead.
final class SourceWiringContractTests: XCTestCase {
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

  /// Issue #710: an orphaned `Settings { EmptyView() }` scene opened a blank
  /// app-titled window on every launch. Every executable owns its interface
  /// through its NSApplicationDelegate (the agent's status-bar popover, the
  /// main app's chat window); no SwiftUI scene belongs in any of them.
  func test_should_declare_no_swiftui_scene_in_any_executable() throws {
    for executableSourceDirectory in executableSourceDirectories {
      let sourceFileURLs = try PackageSources.swiftFileURLs(under: executableSourceDirectory)
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

  func test_should_never_disable_the_restart_server_button() throws {
    let popoverSource = try PackageSources.text(ofSourceFileNamed: "OrbitalTelemetryPopover.swift")
    let restartButtonStart = try XCTUnwrap(
      popoverSource.range(of: "Button(\"Restart server\", action: restartServer)"))
    let trailingActionsStart = try XCTUnwrap(
      popoverSource.range(
        of: "Spacer()", range: restartButtonStart.upperBound..<popoverSource.endIndex))
    let restartButtonSource = popoverSource[
      restartButtonStart.lowerBound..<trailingActionsStart.lowerBound]

    XCTAssertFalse(restartButtonSource.contains(".disabled("))
  }

  /// The main app owns the updater and registers the agent at login, so their
  /// settings stay in its menu. The popover keeps the one-shot update check
  /// people reach for in the menu bar, which it hands to the main app.
  func test_should_keep_the_update_check_in_the_popover_and_leave_settings_to_the_main_app() throws {
    let popoverSource = try PackageSources.text(ofSourceFileNamed: "OrbitalTelemetryPopover.swift")
    let settingsMenuSource = try PackageSources.text(ofSourceFileNamed: "ApplicationSettingsMenu.swift")

    for mainApplicationSetting in ["Launch at Login", "Check automatically", "Update Channel"] {
      XCTAssertFalse(popoverSource.contains("\"\(mainApplicationSetting)"), mainApplicationSetting)
    }
    XCTAssertTrue(popoverSource.contains("\"Check for Updates…\""))
    XCTAssertTrue(settingsMenuSource.contains("\"Launch at Login\""))
    XCTAssertTrue(settingsMenuSource.contains("\"Update Channel\""))
  }

  /// The link matcher is only half the contract: if the main app's delegate
  /// stops registering the GURL Apple-event handler before launch finishes, a
  /// link from the Observatory or from the menu-bar agent silently dead-ends,
  /// and no behavioral test can catch it without a real event loop.
  func test_should_register_the_link_handler_in_the_main_app_before_launch_finishes() throws {
    let applicationSource = try PackageSources.text(ofSourceFileNamed: "AstronomicalApplication.swift")

    let willFinishLaunchingStart = try XCTUnwrap(
      applicationSource.range(of: "func applicationWillFinishLaunching"))
    let handlerRegistrationStart = try XCTUnwrap(
      applicationSource.range(of: "forEventClass: AEEventClass(kInternetEventClass)"))
    let didFinishLaunchingStart = try XCTUnwrap(
      applicationSource.range(of: "func applicationDidFinishLaunching"))
    let matcherUsageStart = try XCTUnwrap(
      applicationSource.range(of: "ApplicationLink.parse(openedURL, channel: applicationIdentity.channel)"))

    XCTAssertLessThan(willFinishLaunchingStart.lowerBound, handlerRegistrationStart.lowerBound)
    XCTAssertLessThan(handlerRegistrationStart.lowerBound, didFinishLaunchingStart.lowerBound)
    XCTAssertLessThan(didFinishLaunchingStart.lowerBound, matcherUsageStart.lowerBound)
  }

  /// The agent has no window: it must ask the main app through a link, never
  /// build a conversation window of its own.
  func test_should_open_chat_from_the_agent_through_a_link_and_link_no_chat_window() throws {
    let agentSource = try PackageSources.text(ofSourceFileNamed: "AstronomicalMenuApplication.swift")

    XCTAssertTrue(agentSource.contains("openInMainApplication(.chat)"))
    XCTAssertFalse(agentSource.contains("import ThinTalk"))
    XCTAssertFalse(agentSource.contains("setActivationPolicy(.regular)"), "the agent never becomes a regular app")
  }

  /// An update that installs at quit has no relaunch to hook, so the main app
  /// must stop its agent as it terminates. No behavioral test reaches that
  /// wiring without a real update.
  func test_should_stop_the_agent_at_quit_when_an_update_waits_for_quit() throws {
    let applicationSource = try PackageSources.text(ofSourceFileNamed: "AstronomicalApplication.swift")
    let willTerminateStart = try XCTUnwrap(applicationSource.range(of: "func applicationWillTerminate"))
    let stopOnQuit = try XCTUnwrap(
      applicationSource.range(
        of: "if applicationUpdateController.installsUpdateOnQuit { stopAgentBeforeUpdate() }"))

    XCTAssertLessThan(willTerminateStart.lowerBound, stopOnQuit.lowerBound)
  }

  /// xtask passes the argument and the app reads it; a rename on one side
  /// alone would silently bring every background rebuild to the front again.
  func test_should_read_the_same_launch_argument_xtask_passes() throws {
    let xtaskSource = try String(
      contentsOf: PackageSources.repositoryRootURL
        .appendingPathComponent("xtask/src/app_bundle/running_instance.rs"),
      encoding: .utf8)

    XCTAssertTrue(
      xtaskSource.contains(
        "LAUNCH_IN_BACKGROUND_ARGUMENT: &str = \"\(ChatLaunchPlacement.launchInBackgroundArgument)\";"))
  }
}
