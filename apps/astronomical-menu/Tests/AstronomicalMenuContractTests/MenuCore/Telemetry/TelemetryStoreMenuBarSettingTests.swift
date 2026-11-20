import XCTest

@testable import AstronomicalMenuCore

/// An engine that stores the menu bar choice it is sent and reports it back, or refuses it.
private final class MenuBarSettingDaemonClient: DaemonClient, @unchecked Sendable {
  private let stateLock = NSLock()
  private var storedSwitches: [MenuBarSwitch: Bool] = [:]
  private let refusesUpdates: Bool

  init(refusesUpdates: Bool) { self.refusesUpdates = refusesUpdates }

  func fetchStatus() async throws -> DaemonStatusDocument {
    let storedSwitches = stateLock.withLock { self.storedSwitches }
    return DaemonStatusDocument(
      status: "ready", activity: "idle", readyModelIdentifier: nil, progress: nil,
      expertMemoryMode: nil, mlxMemoryCeilingBytes: 0,
      menuBarShowsProgressAndSpeed: storedSwitches[.progressAndSpeed] ?? false,
      menuBarShowsTechnicalDetails: storedSwitches[.technicalDetails] ?? false,
      servingSession: .empty)
  }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    ConfigurationReloadResult(message: "Config reloaded")
  }

  func requestShutdown() async throws {}

  func updateMenuBarSetting(_ menuBarSwitch: MenuBarSwitch, isOn: Bool) async throws {
    if refusesUpdates { throw DaemonClientError.serverRejected("config.json could not be read") }
    stateLock.withLock { storedSwitches[menuBarSwitch] = isOn }
  }

  func healthIsAvailable() async -> Bool { true }
}

@MainActor
final class TelemetryStoreMenuBarSettingTests: XCTestCase {
  func test_should_show_the_saved_choice_in_the_menu_bar_after_the_engine_stores_it() async {
    let telemetryStore = TelemetryStore(daemonClient: MenuBarSettingDaemonClient(refusesUpdates: false))
    var latestMenuBarPresentation: MenuBarPresentation?
    telemetryStore.onMenuBarPresentationChanged = { latestMenuBarPresentation = $0 }

    await telemetryStore.performMenuBarSettingUpdate(.progressAndSpeed, isOn: true)

    XCTAssertTrue(telemetryStore.statusDocument.menuBarShowsProgressAndSpeed)
    XCTAssertFalse(telemetryStore.statusDocument.menuBarShowsTechnicalDetails)
    XCTAssertEqual(latestMenuBarPresentation?.showsProgressAndSpeed, true)
    XCTAssertNil(telemetryStore.controlActionFeedback)

    await telemetryStore.performMenuBarSettingUpdate(.progressAndSpeed, isOn: false)

    XCTAssertEqual(latestMenuBarPresentation?.showsProgressAndSpeed, false)
  }

  func test_should_keep_the_stored_choice_and_say_why_when_the_engine_refuses() async {
    let telemetryStore = TelemetryStore(daemonClient: MenuBarSettingDaemonClient(refusesUpdates: true))

    await telemetryStore.performMenuBarSettingUpdate(.technicalDetails, isOn: true)

    XCTAssertFalse(telemetryStore.statusDocument.menuBarShowsTechnicalDetails)
    XCTAssertEqual(telemetryStore.controlActionFeedback?.isFailure, true)
  }

  func test_should_read_an_engine_that_does_not_report_the_choice_as_icon_only() throws {
    let statusDocument = try JSONDecoder().decode(
      DaemonStatusDocument.self, from: Data(#"{"status":"ready","activity":"idle"}"#.utf8))
    XCTAssertFalse(statusDocument.menuBarShowsProgressAndSpeed)
    XCTAssertFalse(statusDocument.menuBarShowsTechnicalDetails)
  }

  func test_should_change_only_the_switch_it_was_asked_to_change() async {
    let telemetryStore = TelemetryStore(daemonClient: MenuBarSettingDaemonClient(refusesUpdates: false))

    await telemetryStore.performMenuBarSettingUpdate(.technicalDetails, isOn: true)

    XCTAssertTrue(telemetryStore.statusDocument.menuBarShowsTechnicalDetails)
    XCTAssertFalse(telemetryStore.statusDocument.menuBarShowsProgressAndSpeed)
  }

  func test_should_name_each_switch_as_config_json_does() {
    XCTAssertEqual(MenuBarSwitch.progressAndSpeed.rawValue, "show_progress_and_speed")
    XCTAssertEqual(MenuBarSwitch.technicalDetails.rawValue, "show_technical_details")
  }
}
