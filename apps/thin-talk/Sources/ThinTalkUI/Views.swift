import SwiftUI
import ThinTalkCanvas
import ThinTalkCore

/// Root of the chat window: wires the dumb-host canvas to the channel's
/// identity and state directory. When the bundled shell is missing the window
/// says so instead of showing a silent blank surface, because a blank
/// conversation is indistinguishable from a broken one.
public struct RootView: View {
  private let applicationIdentity: ThinTalkApplicationIdentity

  public init(applicationIdentity: ThinTalkApplicationIdentity) {
    self.applicationIdentity = applicationIdentity
  }

  public var body: some View {
    if let webDirectory = CanvasWebResources.bundledDirectory() {
      CanvasWebView(webDirectory: webDirectory, applicationIdentity: applicationIdentity)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    } else {
      CanvasUnavailableView()
    }
  }
}

/// Bridges the canvas NSView into SwiftUI. The canvas installs its web view
/// when it lands in a window, so the representable only has to create it once.
private struct CanvasWebView: NSViewRepresentable {
  let webDirectory: URL
  let applicationIdentity: ThinTalkApplicationIdentity

  func makeNSView(context: Context) -> ConversationCanvasView {
    ConversationCanvasView(
      webDirectory: webDirectory,
      assetRegistry: TranscriptAssetRegistry(),
      sessionFileStore: SessionFileStore(directory: Self.stateDirectory(for: applicationIdentity)),
      runtimeConfig: CanvasRuntimeConfig(applicationIdentity: applicationIdentity)
    )
  }

  func updateNSView(_ nsView: ConversationCanvasView, context: Context) {}

  /// The channel's thin-talk state directory, derived from the identity's state
  /// directory name so Stable and Development never share session files.
  private static func stateDirectory(for identity: ThinTalkApplicationIdentity) -> URL {
    FileManager.default.homeDirectoryForCurrentUser
      .appendingPathComponent(identity.stateDirectoryName, isDirectory: true)
      .appendingPathComponent("thin-talk", isDirectory: true)
  }
}

/// Visible failure for a canvas that cannot load its own shell.
struct CanvasUnavailableView: View {
  var body: some View {
    VStack(spacing: 8) {
      Image(systemName: "rectangle.on.rectangle.slash")
        .font(.title2)
        .accessibilityHidden(true)
      Text("The conversation canvas is unavailable.")
        .font(.headline)
      Text("Its bundled rendering assets are missing from this build.")
        .font(.caption)
        .foregroundColor(.secondary)
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity)
  }
}
