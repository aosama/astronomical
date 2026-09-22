import Foundation
import Observation
import ThinTalkCore
import XCTest

@testable import ThinTalkUI

/// Regression contract for the live font-zoom control (issue #766).
///
/// The composer's font modifier reads `viewModel.fontZoom`, so a Cmd+/Cmd- zoom
/// only re-renders when `fontZoom` is tracked by `@Observable`. It shipped marked
/// `@ObservationIgnored`: the value changed and persisted to UserDefaults, but no
/// view was ever invalidated, so the composer stayed at its init size while zoom
/// keystrokes appeared dead. This test fails if `fontZoom` leaves observation again.
@MainActor
final class ChatViewModelFontZoomTests: XCTestCase {
  private var client: ThinTalkClient!

  override func setUp() async throws {
    try await super.setUp()
    client = ThinTalkClient(
      applicationIdentity: ThinTalkApplicationIdentity(
        channel: .development, supervisorPort: 6733),
      urlSession: URLSession(configuration: StubSupervisorURLProtocol.urlSessionConfiguration()),
      stallTimeout: 60
    )
  }

  override func tearDown() async throws {
    client = nil
    try await super.tearDown()
  }

  func test_zooming_fontZoom_invalidates_observers_of_the_composer_font() async {
    let viewModel = ChatViewModel(client: client)
    let observation = ObservationProbe()
    withObservationTracking {
      _ = viewModel.fontZoom
    } onChange: {
      observation.markInvalidated()
    }

    viewModel.fontZoom = viewModel.fontZoom + ComposerFontSize.step

    for _ in 0 ..< 200 {
      if observation.isInvalidated { break }
      try? await Task.sleep(nanoseconds: 10_000_000)
    }

    XCTAssertTrue(
      observation.isInvalidated,
      "fontZoom must stay @Observable-tracked so Cmd+/Cmd- re-renders the composer; "
        + "@ObservationIgnored froze the live zoom while still persisting the value")
  }

  /// `withObservationTracking`'s onChange closure is `@Sendable` and runs off the
  /// test actor, so the signal crosses threads through a locked box instead of a
  /// captured local that Swift 6 would reject.
  private final class ObservationProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var invalidated = false

    var isInvalidated: Bool {
      lock.lock()
      defer { lock.unlock() }
      return invalidated
    }

    func markInvalidated() {
      lock.lock()
      invalidated = true
      lock.unlock()
    }
  }
}
