import SwiftUI
import ThinTalkCanvas
import ThinTalkCore

/// Root of the chat application. Owns the view model for the lifetime of the
/// window and loads the models on first appearance. Production renders the real,
/// client-backed conversation; it never depends on the preview mock data.
public struct RootView: View {
  @State private var viewModel: ChatViewModel
  @State private var loadStarted = false

  public init(viewModel: ChatViewModel = .default()) {
    _viewModel = State(initialValue: viewModel)
  }

  public var body: some View {
    ChatPane(viewModel: viewModel)
      .background(PreviewTheme.windowBackground)
      .onAppear {
        guard !loadStarted else { return }
        loadStarted = true
        Task { await viewModel.load() }
      }
  }
}

/// The single-pane conversation surface: one canvas that carries the transcript,
/// the composer, and the failure banner. Native code keeps every behaviour; the
/// surface is a pure function of the view model's state.
struct ChatPane: View {
  @Bindable var viewModel: ChatViewModel

  var body: some View {
    ConversationCanvas(
      snapshot: viewModel.canvasSnapshot,
      pageZoom: viewModel.fontZoom / 14,
      composerState: viewModel.composerState,
      assetRegistry: viewModel.assetRegistry,
      onAction: viewModel.handle
    )
    .frame(minWidth: 1100, minHeight: 700)
    .background(PreviewTheme.windowBackground)
  }
}

// MARK: - Canvas

/// Hosts the conversation canvas. The canvas renders the transcript, the
/// composer, and the failure banner, so this view only supplies the state, the
/// appearance, and the assets the page may load.
///
/// When the bundled shell is missing the pane says so instead of showing an empty
/// surface, because a silent blank conversation is indistinguishable from a broken
/// model.
struct ConversationCanvas: View {
  @Environment(\.colorScheme) private var colorScheme

  let snapshot: TranscriptSnapshot
  let pageZoom: CGFloat
  let composerState: CanvasComposerState
  let assetRegistry: TranscriptAssetRegistry
  let onAction: (CanvasAction) -> Void

  var body: some View {
    if let webDirectory = CanvasWebResources.bundledDirectory() {
      ConversationCanvasView(
        snapshot: snapshot,
        isDarkAppearance: colorScheme == .dark,
        composerState: composerState,
        webDirectory: webDirectory,
        assetRegistry: assetRegistry,
        pageZoom: pageZoom,
        onAction: onAction
      )
      .frame(maxWidth: .infinity, maxHeight: .infinity)
      .background(PreviewTheme.windowBackground)
    } else {
      CanvasUnavailableView()
    }
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
    .background(PreviewTheme.windowBackground)
  }
}