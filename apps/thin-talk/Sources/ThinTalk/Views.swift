import SwiftUI
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

/// The single-pane conversation surface: message list, composer, and any failure
/// or state banner. This is one View type so a change in one section does not
/// re-run the others.
struct ChatPane: View {
  @Bindable var viewModel: ChatViewModel

  var body: some View {
    VStack(spacing: 0) {
      MessageList(viewModel: viewModel)
      Divider()
        .background(PreviewTheme.border)
        .padding(.horizontal, 24)
      ComposeBar(viewModel: viewModel)
      if let failure = viewModel.currentFailure {
        ChatFailureBanner(failure: failure, onRetry: viewModel.retry)
      }
      if let stateError = stateErrorText {
        StateErrorBanner(message: stateError)
      }
    }
    .frame(minWidth: 1100, minHeight: 700)
  }

  private var stateErrorText: String? {
    switch viewModel.state {
    case .failed(let message): return message
    default: return nil
    }
  }
}

// MARK: - Messages

/// Scrollable conversation with an empty-state overlay that centers vertically
/// across the whole pane, like the product's home screen.
struct MessageList: View {
  @Bindable var viewModel: ChatViewModel

  var body: some View {
    ScrollView {
      LazyVStack(alignment: .leading, spacing: 12) {
        ForEach(viewModel.messages) { message in
          MessageRow(
            message: message,
            content: viewModel.attributedText(message.content),
            reasoning: viewModel.attributedText(message.reasoning)
          )
        }
      }
      .padding()
      .frame(maxWidth: .infinity)
    }
    .overlay { EmptyStateView(viewModel: viewModel) }
  }
}

/// One conversation message: role label, optional reasoning block, and the
/// rendered (already-attributed) answer. Attributed strings are pre-computed so
/// this row stays cheap and does no parsing in body.
struct MessageRow: View {
  let message: ChatMessage
  let content: AttributedString
  let reasoning: AttributedString

  var body: some View {
    let isUser = message.role == .user
    return VStack(alignment: .leading, spacing: 6) {
      Text(message.role == .user ? "You" : "Assistant")
        .font(.caption)
        .foregroundColor(.secondary)
      if !message.reasoning.isEmpty {
        Text(reasoning)
          .font(.callout)
          .foregroundColor(.secondary)
          .italic()
          .frame(maxWidth: .infinity, alignment: .leading)
          .padding(8)
          .background(Color.secondary.opacity(0.08))
          .clipShape(RoundedRectangle(cornerRadius: 8))
      }
      Text(content)
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
    .padding(10)
    .background((isUser ? Color.blue : Color.gray.opacity(0.15)).opacity(0.12))
    .clipShape(RoundedRectangle(cornerRadius: 10))
    .padding(.horizontal)
  }
}

/// Empty conversation state, including the "no model available" guidance.
struct EmptyStateView: View {
  @Bindable var viewModel: ChatViewModel

  var body: some View {
    if viewModel.messages.isEmpty {
      VStack(spacing: 10) {
        HStack(spacing: 10) {
          Image(systemName: "brain.head.profile")
            .font(.title2)
            .foregroundStyle(
              LinearGradient(
                colors: [Color.cyan, Color.purple],
                startPoint: .topLeading, endPoint: .bottomTrailing))
            .accessibilityHidden(true)
          Text("Thin Talk")
            .font(.title)
        }
        Text("Ask your local model anything.")
          .font(.headline)
        if viewModel.state == .empty {
          Text("No chat model is available yet. Add one from the Library.")
            .font(.caption)
        }
      }
      .foregroundColor(.primary)
    }
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
        .disabled(viewModel.state != .ready)
        .onKeyPress(.return, phases: .down) {
          _ in
          sendDraft()
          return .handled
        }
      HStack(spacing: 8) {
        Spacer(minLength: 8)
        ModelPicker(
          models: viewModel.availableModels,
          selection: $viewModel.selectedModelID
        )
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

/// A general state error (for example, a startup handshake failure).
struct StateErrorBanner: View {
  let message: String

  var body: some View {
    Text(message)
      .font(.caption)
      .foregroundColor(.secondary)
      .padding()
      .frame(maxWidth: .infinity)
      .background(Color.orange.opacity(0.1))
  }
}
