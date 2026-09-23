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

/// The single-pane conversation surface: canvas, composer, and any failure or
/// state banner. Each section is its own `View` type so a change in one does not
/// re-evaluate the others.
struct ChatPane: View {
  @Bindable var viewModel: ChatViewModel

  var body: some View {
    VStack(spacing: 0) {
      ConversationCanvas(
        snapshot: viewModel.canvasSnapshot,
        pageZoom: viewModel.fontZoom / 14,
        assetRegistry: viewModel.assetRegistry,
        onAction: viewModel.handle
      )
      ComposeBar(viewModel: viewModel)
      if let failure = viewModel.currentFailure {
        ChatFailureBanner(failure: failure, textScale: viewModel.fontZoom / 14, onRetry: viewModel.retry)
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
  let pageZoom: CGFloat
  let assetRegistry: TranscriptAssetRegistry
  let onAction: (CanvasAction) -> Void

  var body: some View {
    if let webDirectory = CanvasWebResources.bundledDirectory() {
      ConversationCanvasView(
        snapshot: snapshot,
        isDarkAppearance: colorScheme == .dark,
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

// MARK: - Compose

/// One continuous rounded bar: the message field on top, the action pill bar
/// below it. Layout matches Copilot's composer: left-side pill buttons for
/// attachments, thinking effort, and features; right-side icon buttons for
/// visual mode and voice input. The send/stop action lives on the far right.
struct ComposeBar: View {
  @Bindable var viewModel: ChatViewModel

  var body: some View {
    VStack(spacing: 0) {
      TextField("Message Thin Talk", text: $viewModel.draft, axis: .vertical)
        .textFieldStyle(.plain)
        .font(.system(size: viewModel.fontZoom))
        .lineLimit(1...6)
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        // Composer stays editable in all states except loading, so the user can
        // type their ask immediately rather than hearing alert beeps while waiting
        // for the supervisor to connect or recover from a failure.
        .disabled(viewModel.state == .loading)
        .onKeyPress(.return, phases: .down) { _ in
          sendDraft()
          return .handled
        }
      .submitLabel(.send)

      HStack(spacing: 8) {
        // Left-side pill buttons
        AttachmentPillButton()
        SmartThinkingPill(effort: $viewModel.thinkingEffort, textScale: viewModel.fontZoom / 14)
        FeaturePillButton()

        Spacer(minLength: 16)

        // Right-side icon buttons
        GlassesIconPillButton()
        VoiceIconPillButton()

        if viewModel.isStreaming {
          StopActionIconButton()
        } else {
          SendActionIconButton(isEnabled: !viewModel.draft.isEmpty && viewModel.state == .ready)
            .onTapGesture { viewModel.sendMessage() }
            .keyboardShortcut(.defaultAction)
        }
      }
      .padding(.horizontal, 12)
      .padding(.vertical, 8)
    }
    .background(PreviewTheme.panel)
    .clipShape(RoundedRectangle(cornerRadius: 22))
    .overlay(
      RoundedRectangle(cornerRadius: 22)
        .stroke(
          PreviewTheme.accent.opacity(0.45),
          lineWidth: 1.2
        )
    )
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

// MARK: - Compose Pill Buttons

/// Leftmost pill: attachment/add button placeholder.
struct AttachmentPillButton: View {
  var body: some View {
    Button {
      // Placeholder for attachment picker
    } label: {
      Image(systemName: "plus")
        .font(.body)
        .foregroundColor(.white)
        .frame(width: 32, height: 32)
        .background(Circle().fill(PreviewTheme.panelAlt))
    }
    .buttonStyle(.plain)
    .accessibilityLabel("Attach file")
  }
}

/// "Smart" pill: Thinking-effort selector styled as a rounded pill with dropdown arrow.
struct SmartThinkingPill: View {
  @Binding var effort: ThinkingEffort
  /// GUI-wide zoom factor (1.0 at the 14pt base) so pill and menu text follow
  /// the same scale as the rest of the native surface.
  let textScale: CGFloat

  var body: some View {
    Menu {
      ForEach(ThinkingEffort.allCases) { level in
        Button {
          effort = level
        } label: {
          Text(level.budgetSummary).font(.system(size: 14 * textScale))
        }
      }
    } label: {
      HStack(spacing: 4) {
        Text("Smart")
          .font(.system(size: 14 * textScale))
          .foregroundColor(.white)
        Image(systemName: "chevron.down")
          .font(.caption)
          .foregroundColor(.secondary)
      }
      .padding(.horizontal, 10)
      .padding(.vertical, 4)
      .background(Capsule().fill(PreviewTheme.panelAlt))
    }
    .buttonStyle(.plain)
    .accessibilityLabel("Thinking level")
    .help("How many tokens the model may spend thinking before it answers.")
  }
}

/// Feature pill: placeholder for additional features dropdown.
struct FeaturePillButton: View {
  var body: some View {
    Menu {
      // Placeholder menu items
      Text("Coming soon")
    } label: {
      Image(systemName: "sparkles")
        .font(.body)
        .foregroundColor(.white)
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
        .background(Capsule().fill(PreviewTheme.panelAlt))
    }
    .buttonStyle(.plain)
    .accessibilityLabel("Features")
  }
}

/// Right-side icon pill: glasses/visual mode placeholder.
struct GlassesIconPillButton: View {
  var body: some View {
    Button {
      // Placeholder for visual mode toggle
    } label: {
      Image(systemName: "goggles")
        .font(.body)
        .foregroundColor(.secondary)
        .frame(width: 32, height: 32)
    }
    .buttonStyle(.plain)
    .accessibilityLabel("Toggle visual mode")
  }
}

/// Right-side icon pill: voice input placeholder.
struct VoiceIconPillButton: View {
  var body: some View {
    Button {
      // Placeholder for voice input
    } label: {
      Image(systemName: "waveform")
        .font(.body)
        .foregroundColor(.secondary)
        .frame(width: 32, height: 32)
    }
    .buttonStyle(.plain)
    .accessibilityLabel("Voice input")
  }
}

/// Send icon button (arrow up) for the rightmost action.
struct SendActionIconButton: View {
  let isEnabled: Bool

  var body: some View {
    Image(systemName: "arrowshape.turn.up.right")
      .font(.body)
      .foregroundColor(isEnabled ? .white : PreviewTheme.textSecondary.opacity(0.5))
      .frame(width: 32, height: 32)
      .background(isEnabled ? Circle().fill(PreviewTheme.accent) : Circle().fill(PreviewTheme.panelAlt))
  }
}

/// Stop icon button (square) for the rightmost action during streaming.
struct StopActionIconButton: View {
  var body: some View {
    Image(systemName: "stop.fill")
      .font(.body)
      .foregroundColor(.white)
      .frame(width: 32, height: 32)
      .background(Circle().fill(Color.red.opacity(0.8)))
  }
}

// MARK: - Banners

/// A classified failure with its specific next action and a retry affordance.
struct ChatFailureBanner: View {
  let failure: ChatFailure
  /// GUI-wide zoom factor (1.0 at the 14pt base) so the banner follows the
  /// same scale as the rest of the native surface.
  let textScale: CGFloat
  let onRetry: () -> Void

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      if let reason = failure.message {
        Text(reason).foregroundColor(.red).font(.system(size: 14 * textScale))
      }
      Text(failure.nextAction).font(.system(size: 12 * textScale))
      Button("Retry", action: onRetry)
        .controlSize(textScale > 1 ? .large : .regular)
        .buttonStyle(.borderedProminent)
    }
    .padding()
    .background(Color.red.opacity(0.08))
  }
}