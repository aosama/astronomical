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
/// The save panel is how a file leaves for everyone who uses the app: it
/// works the same inside the App Store sandbox, where writing anywhere else
/// would need an entitlement, and the person chooses the folder and the name.
/// The one other way, `save(_:into:)`, is reached only through the
/// Development channel's export-folder link.
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
    return try await write(request, to: destinationURL)
  }

  /// Writes the file into `folder` under its suggested name with no panel,
  /// replacing a file of that name. The folder must already exist: a path
  /// that is not there is a mistake in the command, not a folder to make.
  static func save(_ request: ExportedFileRequest, into folder: URL) async throws -> ExportedFileSaveOutcome {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: folder.path, isDirectory: &isDirectory), isDirectory.boolValue else {
      throw ExportFolderError.folderMissing
    }
    return try await write(request, to: folder.appendingPathComponent(request.suggestedFileName))
  }

  enum ExportFolderError: Error, Equatable {
    case folderMissing
  }

  private static func write(_ request: ExportedFileRequest, to destinationURL: URL) async throws -> ExportedFileSaveOutcome {
    if request.format == .pdf {
      try await ExportPDFPrinter().printPDF(html: request.html ?? "", to: destinationURL)
    } else {
      try ExportedFileWriter.write(request, to: destinationURL)
    }
    return .saved(fileName: destinationURL.lastPathComponent)
  }
}
