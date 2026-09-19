import AstronomicalMenuCore
import AstronomicalMenuSparkleUpdateController
import SwiftUI

@main
struct AstronomicalMenuApp: App {
  init() {
    // Direct channel: opt into Sparkle-backed updates. The App Store executable
    // is byte-identical except for this line and therefore always falls back to
    // the store-semantics controller in the core.
    ApplicationUpdateControllerInstaller.install { applicationChannel in
      SparkleApplicationUpdateController(applicationChannel: applicationChannel)
    }
  }

  @NSApplicationDelegateAdaptor(AstronomicalMenuApplication.self) private var applicationDelegate

  // No scenes. The menu app manages its own state bar UI entirely through its
  // NSApplicationDelegate, so a SwiftUI scene is unnecessary — and a `Settings`
  // scene is a bug: it renders a titled Settings window that auto-opens on launch.
  var body: some Scene {
  }
}
