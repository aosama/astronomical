import SwiftUI
import ThinTalkCore

/// Root of the chat application. Owns the view model for the lifetime of the
/// window and loads the models on first appearance.
struct RootView: View {
  @StateObject private var viewModel: ChatViewModel
  @State private var loadStarted = false

  init(viewModel: ChatViewModel = .default()) {
    _viewModel = StateObject(wrappedValue: viewModel)
  }

  var body: some View {
    HStack(spacing: 0) {
      // Sidebar skeleton: the session rail is lifted from the UX preview so the
      // agreed layout can be styled here in the real app. Styling lives in the
      // preview scaffold; this wires it beside the backend-backed chat below.
      PreviewSidebar(
        viewModel: PreviewSidebarViewModel(
          conversations: PreviewData.conversations,
          identity: PreviewData.identity)
      )
      // A hairline seam instead of a system divider, which renders as a thick
      // grabbable wedge that reads as a resize handle.
      Rectangle()
        .fill(PreviewTheme.border)
        .frame(width: 1)
      mainPane
    }
    .background(PreviewTheme.windowBackground)
    .onAppear {
      guard !loadStarted else { return }
      loadStarted = true
      Task { await viewModel.load() }
    }
  }

  private var mainPane: some View {
    VStack(spacing: 0) {
      messageList
      Divider()
        .background(PreviewTheme.border)
        .padding(.horizontal, 24)
      composeBar
      if let failure = viewModel.currentFailure { failureBanner(failure) }
      if let stateError = stateErrorText { banner(stateError) }
    }
    .frame(minWidth: 1100, minHeight: 700)
  }

  private var stateErrorText: String? {
    switch viewModel.state {
    case .failed(let message): return message
    default: return nil
    }
  }

  // MARK: - Messages

  private var messageList: some View {
    // The empty state overlays the scroll view instead of living inside it so
    // it can center vertically across the whole pane, like the product's home
    // screen, instead of sitting at the top of an empty list.
    ScrollView {
      LazyVStack(alignment: .leading, spacing: 12) {
        ForEach(viewModel.messages) { message in messageRow(message) }
      }
      .padding()
      .frame(maxWidth: .infinity)
    }
    .overlay { emptyState }
  }

  @ViewBuilder
  private var emptyState: some View {
    if viewModel.messages.isEmpty {
      VStack(spacing: 10) {
        HStack(spacing: 10) {
          Image(systemName: "brain.head.profile")
            .font(.system(size: 28, weight: .medium))
            .foregroundStyle(
              LinearGradient(
                colors: [Color.cyan, Color.purple],
                startPoint: .topLeading, endPoint: .bottomTrailing))
          Text("Thin Talk")
            .font(.system(size: 30, weight: .bold))
        }
        Text("Ask your local model anything.")
          .font(.system(size: 15, weight: .medium))
        if viewModel.state == .empty {
          Text("No chat model is available yet. Add one from the Library.")
            .font(.caption)
        }
      }
      .foregroundColor(.primary)
    }
  }

  private func messageRow(_ message: ChatMessage) -> some View {
    let isUser = message.role == .user
    return VStack(alignment: .leading, spacing: 6) {
      Text(message.role == .user ? "You" : "Assistant").font(.caption).foregroundColor(.secondary)
      if !message.reasoning.isEmpty {
        Text(markdownText(message.reasoning))
          .font(.callout)
          .foregroundColor(.secondary)
          .italic()
          .frame(maxWidth: .infinity, alignment: .leading)
          .padding(8)
          .background(Color.secondary.opacity(0.08))
          .clipShape(RoundedRectangle(cornerRadius: 8))
      }
      Text(markdownText(message.content))
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
    .padding(10)
    .background((isUser ? Color.blue : Color.gray.opacity(0.15)).opacity(0.12))
    .clipShape(RoundedRectangle(cornerRadius: 10))
    .padding(.horizontal)
  }

  /// Renders assistant markdown inline. Falls back to the raw string when the
  /// text is not valid markdown, because a partially streamed reply is often
  /// mid-syntax and must never crash the conversation.
  private func markdownText(_ raw: String) -> AttributedString {
    if let attributed = try? AttributedString(
      markdown: raw,
      options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)
    ) {
      return attributed
    }
    return AttributedString(raw)
  }

  // MARK: - Compose

  private var composeBar: some View {
    // One continuous rounded bar: the message field on top, the model
    // selector centered below it, and the action button at the trailing edge,
    // so the composer reads as a single surface instead of stacked fields.
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
        modelPicker
        Spacer(minLength: 8)
        actionButton
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

  @ViewBuilder
  private var modelPicker: some View {
    if !viewModel.availableModels.isEmpty {
      Picker("Model", selection: $viewModel.selectedModelID) {
        ForEach(viewModel.availableModels) { model in
          Text(model.id).tag(model.id)
        }
      }
      .pickerStyle(.menu)
      .frame(maxWidth: 260)
      .labelStyle(.titleAndIcon)
    }
  }

  @ViewBuilder
  private var actionButton: some View {
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

  private func sendDraft() {
    guard !viewModel.isStreaming else { return }
    viewModel.sendMessage()
  }

  // MARK: - Banners

  private func failureBanner(_ failure: ChatFailure) -> some View {
    VStack(alignment: .leading, spacing: 6) {
      if let reason = failure.message { Text(reason).foregroundColor(.red) }
      Text(failure.nextAction).font(.caption)
      Button("Retry") { viewModel.retry() }
        .buttonStyle(.borderedProminent)
    }
    .padding()
    .background(Color.red.opacity(0.08))
  }

  private func banner(_ message: String) -> some View {
    Text(message)
      .font(.caption)
      .foregroundColor(.secondary)
      .padding()
      .frame(maxWidth: .infinity)
      .background(Color.orange.opacity(0.1))
  }
}
