import SwiftUI

@main
struct AstronomicalMenuApp: App {
  @NSApplicationDelegateAdaptor(AstronomicalMenuApplication.self) private var applicationDelegate

  var body: some Scene {
    Settings { EmptyView() }
  }
}
