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

  var body: some Scene {
    Settings { EmptyView() }
  }
}
