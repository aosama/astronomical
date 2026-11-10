import AstronomicalMenuCore
import SwiftUI

/// The menu-bar agent's executable. One agent serves every channel: it links
/// neither Sparkle nor the chat window, both of which belong to the main app.
@main
struct AstronomicalMenuApp: App {
  @NSApplicationDelegateAdaptor(AstronomicalMenuApplication.self) private var applicationDelegate

  // Checked before the delegate exists, so a second agent leaves without
  // ever showing a menu-bar item or reaching for the engine.
  init() {
    SingleAgentGuard.endThisProcessIfAnotherAgentRuns()
  }

  // No scenes. The agent manages its status-bar interface entirely through its
  // NSApplicationDelegate, so a SwiftUI scene is unnecessary — and a `Settings`
  // scene is a bug: it renders a titled Settings window that auto-opens on launch.
  var body: some Scene {
  }
}
