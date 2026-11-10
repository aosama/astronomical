import AstronomicalAppCore

@main
enum AstronomicalAppStoreApp {
  // Store channel: no update controller is installed, by design. The core's
  // default controller carries App Store semantics, and updates arrive
  // through the store itself rather than any in-app mechanism.
  @MainActor
  static func main() {
    runAstronomicalApplication()
  }
}
