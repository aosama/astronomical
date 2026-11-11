import AppKit

/// How a save ended, as the page receives it. The page learns the file's
/// name, to say it was saved, and never its folder.
public enum ExportedFileSaveOutcome: Equatable, Sendable {
  case saved(fileName: String)
  case cancelled

  var bridgePayload: [String: String] {
    switch self {
    case .saved(let fileName): return ["outcome": "saved", "fileName": fileName]
    case .cancelled: return ["outcome": "cancelled"]
    }
  }
}

/// Asks where to save an exported file and writes it there.
///
/// The save panel is the only way a file leaves: it works the same inside the
/// App Store sandbox, where writing anywhere else would need an entitlement,
/// and the person chooses the folder and the name.
@MainActor
enum ExportedFileSaver {
  static func save(_ request: ExportedFileRequest, from window: NSWindow?) async throws -> ExportedFileSaveOutcome {
    let savePanel = NSSavePanel()
    savePanel.nameFieldStringValue = request.suggestedFileName
    savePanel.allowedContentTypes = [request.format.contentType]
    savePanel.canCreateDirectories = true
    savePanel.isExtensionHidden = false
    let panelAnswer: NSApplication.ModalResponse
    if let window {
      panelAnswer = await savePanel.beginSheetModal(for: window)
    } else {
      panelAnswer = savePanel.runModal()
    }
    guard panelAnswer == .OK, let destinationURL = savePanel.url else { return .cancelled }
    if request.format == .pdf {
      try await ExportPDFPrinter().printPDF(html: request.html ?? "", to: destinationURL)
    } else {
      try ExportedFileWriter.write(request, to: destinationURL)
    }
    return .saved(fileName: destinationURL.lastPathComponent)
  }
}
