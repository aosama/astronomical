import AstronomicalAppCore
import AstronomicalSparkleUpdateController

@main
enum AstronomicalApp {
  @MainActor
  static func main() {
    // Direct channel: opt into Sparkle-backed updates. The App Store executable
    // is identical except for this call and therefore always falls back to the
    // store-semantics controller in the core.
    ApplicationUpdateControllerInstaller.install { applicationChannel in
      SparkleApplicationUpdateController(applicationChannel: applicationChannel)
    }
    runAstronomicalApplication()
  }
}
