import Foundation
import UniformTypeIdentifiers

/// The kinds of file the page exports. The raw values are the page's
/// `ExportFormat` values (web/src/export/ExportFormat.ts), which are also the
/// file extensions.
public enum ExportedFileFormat: String, CaseIterable, Sendable {
  case pdf
  case docx
  case html

  public var contentType: UTType {
    switch self {
    case .pdf: return .pdf
    case .docx: return UTType(filenameExtension: "docx") ?? .data
    case .html: return .html
    }
  }
}

/// One exported file as the page hands it over: a PDF or a web page as the
/// page's HTML, a Word document as its bytes.
///
/// The page is a rendering tenant for model text, so nothing in the request
/// is trusted to choose a path: the name is only a suggestion for the save
/// panel, cleaned here again, and the person picks where the file goes.
public struct ExportedFileRequest: Sendable, Equatable {
  /// The longest name offered, before the extension.
  static let maximumSuggestedNameCharacters = 80
  static let untitledName = "Conversation"

  public let format: ExportedFileFormat
  public let suggestedName: String
  /// The page to save (a web page) or print (a PDF).
  public let html: String?
  /// The file's bytes (a Word document).
  public let documentData: Data?

  /// Reads the page's `saveExport` payload, or nil when it does not carry
  /// what its format needs.
  public init?(payload: [String: Any]) {
    guard let formatValue = payload["format"] as? String, let format = ExportedFileFormat(rawValue: formatValue) else {
      return nil
    }
    let rawSuggestedName = payload["suggestedName"] as? String ?? ""
    switch format {
    case .pdf, .html:
      guard let html = payload["html"] as? String, !html.isEmpty else { return nil }
      self.init(format: format, suggestedName: rawSuggestedName, html: html, documentData: nil)
    case .docx:
      guard let base64 = payload["base64"] as? String, let documentData = Data(base64Encoded: base64), !documentData.isEmpty
      else { return nil }
      self.init(format: format, suggestedName: rawSuggestedName, html: nil, documentData: documentData)
    }
  }

  init(format: ExportedFileFormat, suggestedName: String, html: String?, documentData: Data?) {
    self.format = format
    self.suggestedName = Self.cleanedName(suggestedName)
    self.html = html
    self.documentData = documentData
  }

  /// The name the save panel offers, with its extension.
  public var suggestedFileName: String {
    "\(suggestedName).\(format.rawValue)"
  }

  /// A name that is only a name: no folder separators, no colon (Finder's
  /// separator), no leading dot that would hide the file, no control
  /// characters, and not empty.
  static func cleanedName(_ rawName: String) -> String {
    let unsafeCharacters = CharacterSet(charactersIn: "/\\:").union(.controlCharacters).union(.newlines)
    let separatedWords = rawName.components(separatedBy: unsafeCharacters).joined(separator: " ")
    let collapsedName = separatedWords.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    let visibleName = String(collapsedName.drop(while: { $0 == "." || $0 == " " }))
    let shortenedName = String(visibleName.prefix(maximumSuggestedNameCharacters)).trimmingCharacters(in: .whitespaces)
    return shortenedName.isEmpty ? untitledName : shortenedName
  }
}

/// Writes the formats that need no rendering: a web page and a Word file.
public enum ExportedFileWriter {
  public enum WriteError: Error, Equatable {
    /// A PDF is printed by `ExportPDFPrinter`, not written from data.
    case needsPrinting
    case missingContent
  }

  /// Writes the file atomically, so a failed write never leaves half a file
  /// where the person chose to save, nor removes one they agreed to replace.
  public static func write(_ request: ExportedFileRequest, to destinationURL: URL) throws {
    switch request.format {
    case .pdf:
      throw WriteError.needsPrinting
    case .html:
      guard let html = request.html else { throw WriteError.missingContent }
      try Data(html.utf8).write(to: destinationURL, options: .atomic)
    case .docx:
      guard let documentData = request.documentData else { throw WriteError.missingContent }
      try documentData.write(to: destinationURL, options: .atomic)
    }
  }
}
