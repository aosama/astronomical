import SwiftUI
import ThinTalkCanvas
import ThinTalkCore

/// Root of the chat application. Owns the view model for the lifetime of the
/// window and loads the models on first appearance. Production renders the real,
/// client-backed conversation; it never depends on the preview mock data.
struct RootView: View {
  @State private var viewModel: ChatViewModel
  @State private var loadStarted = false

  init(viewModel: ChatViewModel = .default()) {
    _viewModel = State(initialValue: viewModel)
  }

  var body: some View {
    ChatPane(viewModel: viewModel)
      .background(PreviewTheme.windowBackground)
      .onAppear {
        guard !loadStarted else { return }
        loadStarted = true
        Task { await viewModel.load() }
      }
  }
}

/// The single-pane conversation surface: canvas, composer, and any failure or
/// state banner. Each section is its own `View` type so a change in one does not
/// re-evaluate the others.
struct ChatPane: View {
  @Bindable var viewModel: ChatViewModel

  var body: some View {
    VStack(spacing: 0) {
      ConversationCanvas(
        snapshot: viewModel.canvasSnapshot,
        assetRegistry: viewModel.assetRegistry,
        onAction: viewModel.handle
      )
      Divider()
        .background(PreviewTheme.border)
        .padding(.horizontal, 24)
      ComposeBar(viewModel: viewModel)
      if let failure = viewModel.currentFailure {
        ChatFailureBanner(failure: failure, onRetry: viewModel.retry)
      }
    }
    .frame(minWidth: 1100, minHeight: 700)
  }
}

// MARK: - Canvas

/// Hosts the conversation canvas. The canvas renders the transcript, so this view
/// only supplies the snapshot, the appearance, and the assets the page may load.
///
/// When the bundled shell is missing the pane says so instead of showing an empty
/// surface, because a silent blank conversation is indistinguishable from a broken
/// model.
struct ConversationCanvas: View {
  @Environment(\.colorScheme) private var colorScheme

  let snapshot: TranscriptSnapshot
  let assetRegistry: TranscriptAssetRegistry
  let onAction: (CanvasAction) -> Void

  var body: some View {
    if let webDirectory = CanvasWebResources.bundledDirectory() {
      ConversationCanvasView(
        snapshot: snapshot,
        isDarkAppearance: colorScheme == .dark,
        webDirectory: webDirectory,
        assetRegistry: assetRegistry,
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

// MARK: - Compose

/// One continuous rounded bar: the message field on top, the model selector
/// centered below it, and the action button at the trailing edge, so the
/// composer reads as a single surface instead of stacked fields.
struct ComposeBar: View {
  @Bindable var viewModel: ChatViewModel

  var body: some View {
    VStack(spacing: 6) {
      TextField("Message Thin Talk", text: $viewModel.draft, axis: .vertical)
        .textFieldStyle(.plain)
        .lineLimit(1...6)
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .disabled(viewModel.state != .ready && viewModel.state != .empty)
        .onKeyPress(.return, phases: .down) { _ in
          sendDraft()
          return .handled
        }
      HStack(spacing: 8) {
        Spacer(minLength: 8)
        ModelPicker(
          models: viewModel.availableModels,
          selection: $viewModel.selectedModelID
        )
        ThinkingEffortPicker(effort: $viewModel.thinkingEffort)
        Spacer(minLength: 8)
        ActionButton(viewModel: viewModel)
      }
      .padding(.horizontal, 10)
      .padding(.bottom, 8)
    }
    .background(PreviewTheme.panel)
    .clipShape(RoundedRectangle(cornerRadius: 14))
    .overlay(RoundedRectangle(cornerRadius: 14).stroke(PreviewTheme.border, lineWidth: 1))
    .padding(.horizontal, 24)
    .padding(.top, 4)
    .padding(.bottom, 14)
    .background(PreviewTheme.windowBackground)
  }

  private func sendDraft() {
    guard !viewModel.isStreaming else { return }
    viewModel.sendMessage()
  }
}

/// Model selector shown when one or more chat-capable models are available.
struct ModelPicker: View {
  let models: [ThinTalkModel]
  @Binding var selection: String?

  var body: some View {
    if !models.isEmpty {
      Picker("Model", selection: $selection) {
        ForEach(models) { model in
          Text(model.id).tag(model.id)
        }
      }
      .pickerStyle(.menu)
      .frame(maxWidth: 260)
      .labelStyle(.titleAndIcon)
    }
  }
}

/// Thinking-budget selector. Each choice states its own token cost, so the reader
/// sees what depth costs before spending it rather than after.
struct ThinkingEffortPicker: View {
  @Binding var effort: ThinkingEffort

  var body: some View {
    Picker("Thinking", selection: $effort) {
      ForEach(ThinkingEffort.allCases) { level in
        Text(level.budgetSummary).tag(level)
      }
    }
    .pickerStyle(.menu)
    .frame(maxWidth: 200)
    .labelStyle(.titleAndIcon)
    .help("How many tokens the model may spend thinking before it answers.")
  }
}

/// Send / Stop action button, driven purely by the streaming state.
struct ActionButton: View {
  @Bindable var viewModel: ChatViewModel

  var body: some View {
    if viewModel.isStreaming {
      Button("Stop") {
        viewModel.stopStreaming()
      }
      .buttonStyle(.bordered)
    } else {
      Button("Send") { viewModel.sendMessage() }
        .buttonStyle(.borderedProminent)
        .disabled(viewModel.draft.isEmpty || viewModel.state != .ready)
        .keyboardShortcut(.defaultAction)
    }
  }
}

// MARK: - Banners

/// A classified failure with its specific next action and a retry affordance.
struct ChatFailureBanner: View {
  let failure: ChatFailure
  let onRetry: () -> Void

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      if let reason = failure.message { Text(reason).foregroundColor(.red) }
      Text(failure.nextAction).font(.caption)
      Button("Retry", action: onRetry)
        .buttonStyle(.borderedProminent)
    }
    .padding()
    .background(Color.red.opacity(0.08))
  }
}