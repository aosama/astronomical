import AppKit
import ThinTalkCore
import UniformTypeIdentifiers

/// Answers the page's host-tool requests. One request is one whole use of a
/// tool: the approval sheet and, for the file tools, the open panel, so the
/// page can never reach a panel without the sheet.
///
/// Uses run one at a time, in arrival order, and the approval policy is
/// checked again when a use's turn comes, so "Allow for This Conversation" on
/// one sheet spares the next. The page can cancel a use by call id when its
/// turn stops; a waiting use is dropped and a showing sheet or panel closes.
/// Nothing is ever asked on a hidden window or as an app-wide alert: with no
/// visible conversation window the use is unavailable.
@MainActor
final class ThinTalkToolHost {
  private var approvalPolicy = ToolApprovalPolicy()
  private let windowProvider: () -> NSWindow?
  /// The use that started last; the next one waits for it.
  private var lastQueuedUse: Task<Void, Never>?
  private var cancelledCallIDs = Set<String>()
  /// The call whose sheet or panel is on screen, and how to close it.
  private var presentedCallID: String?
  private var dismissPresentedChoice: (() -> Void)?

  init(windowProvider: @escaping () -> NSWindow?) {
    self.windowProvider = windowProvider
  }

  /// Runs one use of a host tool after any use already waiting.
  func use(toolName: String, conversationID: String, callID: String) async -> HostToolOutcome {
    let previousUse = lastQueuedUse
    let queuedUse = Task { @MainActor [weak self] () -> HostToolOutcome in
      await previousUse?.value
      guard let self else { return .cancelled }
      return await self.perform(toolName: toolName, conversationID: conversationID, callID: callID)
    }
    lastQueuedUse = Task { _ = await queuedUse.value }
    return await queuedUse.value
  }

  /// Cancels a use: it is skipped if still waiting, and its sheet or panel is
  /// closed if showing.
  func cancel(callID: String) {
    cancelledCallIDs.insert(callID)
    if presentedCallID == callID {
      dismissPresentedChoice?()
    }
  }

  private func perform(toolName: String, conversationID: String, callID: String) async -> HostToolOutcome {
    defer { cancelledCallIDs.remove(callID) }
    guard !cancelledCallIDs.contains(callID) else { return .cancelled }
    let actionDescription: String
    switch approvalPolicy.decision(toolName: toolName, conversationID: conversationID) {
    case .refuse:
      return .unavailable(reason: "this tool is not available in this conversation")
    case .alreadyAllowed:
      actionDescription = ""
    case .ask(let askedActionDescription):
      actionDescription = askedActionDescription
    }
    guard let window = windowProvider(), window.isVisible else {
      return .unavailable(reason: "the conversation window is closed")
    }
    if !actionDescription.isEmpty {
      let sheetOutcome = await askForApproval(
        toolName: toolName, conversationID: conversationID, actionDescription: actionDescription, window: window, callID: callID)
      guard sheetOutcome == .approved else { return sheetOutcome }
    }
    return await chooseFile(for: toolName, window: window, callID: callID)
  }

  private func askForApproval(
    toolName: String,
    conversationID: String,
    actionDescription: String,
    window: NSWindow,
    callID: String
  ) async -> HostToolOutcome {
    let approvalAlert = NSAlert()
    // The sheet names the app as the menu bar and the Dock do. "Thin Talk" is
    // the chat surface's internal name; a person who installed Astronomical
    // has never been told it, and a sheet asking permission for an unknown
    // name reads as something else asking.
    approvalAlert.messageText = "Allow \(ToolApprovalPolicy.applicationName) to \(actionDescription)?"
    approvalAlert.informativeText = ToolApprovalPolicy.consequencesByToolName[toolName] ?? ""
    approvalAlert.addButton(withTitle: "Allow")
    approvalAlert.addButton(withTitle: "Allow for This Conversation")
    // Escape answers no, as it does on other Mac sheets.
    approvalAlert.addButton(withTitle: "Don’t Allow").keyEquivalent = "\u{1b}"
    presentedCallID = callID
    dismissPresentedChoice = { [weak window] in window?.endSheet(approvalAlert.window, returnCode: .abort) }
    let chosenButton = await approvalAlert.beginSheetModal(for: window)
    presentedCallID = nil
    dismissPresentedChoice = nil
    switch chosenButton {
    case .alertFirstButtonReturn:
      return .approved
    case .alertSecondButtonReturn:
      approvalPolicy.allowForConversation(toolName: toolName, conversationID: conversationID)
      return .approved
    case .abort:
      return .cancelled
    default:
      return .refused
    }
  }

  /// The file tools go on to an open panel; conversation search needs nothing
  /// more than the approval.
  private func chooseFile(for toolName: String, window: NSWindow, callID: String) async -> HostToolOutcome {
    guard !cancelledCallIDs.contains(callID) else { return .cancelled }
    guard toolName == "read_file" || toolName == "describe_image" else { return .approved }
    let isImageTool = toolName == "describe_image"
    guard let fileURL = await presentOpenPanel(isImage: isImageTool, window: window, callID: callID) else {
      return cancelledCallIDs.contains(callID) ? .cancelled : .nothingChosen
    }
    do {
      return isImageTool
        ? .chosenImageFile(try ChosenFileReader.imageFile(at: fileURL))
        : .chosenTextFile(try ChosenFileReader.textFile(at: fileURL))
    } catch {
      return .unavailable(reason: "the chosen file could not be used: \(error)")
    }
  }

  /// The text panel accepts any file because ChosenFileReader decides what
  /// is readable text; restricting by type would grey out plain text files
  /// without a known extension (Makefile, .env).
  private func presentOpenPanel(isImage: Bool, window: NSWindow, callID: String) async -> URL? {
    let openPanel = NSOpenPanel()
    openPanel.allowedContentTypes = isImage ? [.png, .jpeg] : [.data]
    openPanel.allowsMultipleSelection = false
    openPanel.canChooseDirectories = false
    openPanel.message =
      isImage
      ? "Choose an image for \(ToolApprovalPolicy.applicationName) to look at."
      : "Choose a text file for \(ToolApprovalPolicy.applicationName) to read."
    presentedCallID = callID
    dismissPresentedChoice = { [weak openPanel] in openPanel?.cancel(nil) }
    let response = await openPanel.beginSheetModal(for: window)
    presentedCallID = nil
    dismissPresentedChoice = nil
    return response == .OK ? openPanel.url : nil
  }
}
