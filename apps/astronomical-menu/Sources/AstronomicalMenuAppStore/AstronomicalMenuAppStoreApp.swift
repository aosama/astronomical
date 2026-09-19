import AstronomicalMenuCore
import SwiftUI

@main
struct AstronomicalMenuApp: App {
  // Store channel: no installer call by design. The core's default update
  // controller carries App Store semantics, and updates arrive through the
  // store itself rather than any in-app mechanism.
  @NSApplicationDelegateAdaptor(AstronomicalMenuApplication.self) private var applicationDelegate

  // No scenes. The menu app manages its own state bar UI entirely through its
  // NSApplicationDelegate, so a SwiftUI scene is unnecessary — and a `Settings`
  // scene is a bug: it renders a titled Settings window that auto-opens on launch.
  var body: some Scene {
  }
}
