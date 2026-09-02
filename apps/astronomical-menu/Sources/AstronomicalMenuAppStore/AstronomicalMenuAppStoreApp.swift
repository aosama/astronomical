import AstronomicalMenuCore
import SwiftUI

@main
struct AstronomicalMenuApp: App {
  // Store channel: no installer call by design. The core's default update
  // controller carries App Store semantics, and updates arrive through the
  // store itself rather than any in-app mechanism.
  @NSApplicationDelegateAdaptor(AstronomicalMenuApplication.self) private var applicationDelegate

  var body: some Scene {
    Settings { EmptyView() }
  }
}
