import Foundation
import XCTest

@testable import AstronomicalMenu

final class DaemonControlTests: XCTestCase {
  @MainActor
  func test_should_report_when_an_unowned_server_does_not_stop() async {
    let daemonLifecycleController = DaemonLifecycleController(
      supervisorClient: StuckExternalSupervisorClient()
    )

    do {
      _ = try await daemonLifecycleController.restartDaemon()
      XCTFail("A still-running unowned server must not be reported as restarted")
    } catch {
      XCTAssertEqual(error.localizedDescription, "The existing server did not stop; quit it and retry")
    }
  }
}

private struct StuckExternalSupervisorClient: SupervisorClient {
  func fetchStatus() async throws -> SupervisorStatusDocument { .unavailable }

  func reloadConfiguration() async throws -> ConfigurationReloadResult {
    ConfigurationReloadResult(message: "Config reloaded")
  }

  func requestShutdown() async throws {}

  func healthIsAvailable() async -> Bool { true }
}
