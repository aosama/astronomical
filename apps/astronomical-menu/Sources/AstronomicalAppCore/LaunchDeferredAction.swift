import Foundation

/// Runs an action asked for by a link, holding it back while the app is still
/// launching.
///
/// The popover's "Check for Updates…" reaches the main app as a link, and
/// that link can be what launches the app. macOS delivers it before launching
/// has finished, when the updater has not started and refuses a check, so the
/// request would be lost without a word. A request made that early is kept
/// and runs once, as soon as the app says it is ready.
@MainActor
final class LaunchDeferredAction {
  private let action: @MainActor () -> Void
  private var isReady = false
  private var isRequestedBeforeReady = false

  init(action: @escaping @MainActor () -> Void) {
    self.action = action
  }

  func request() {
    guard isReady else {
      isRequestedBeforeReady = true
      return
    }
    action()
  }

  /// Called once what the action needs has started. Runs a request that
  /// arrived earlier, however many times it was asked for, exactly once.
  func becomeReady() {
    isReady = true
    guard isRequestedBeforeReady else { return }
    isRequestedBeforeReady = false
    action()
  }
}
