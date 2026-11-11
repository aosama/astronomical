import AppKit
import PDFKit
import XCTest

@testable import ThinTalkCanvas

/// Saving an exported conversation: what the page may hand over, the name the
/// save panel offers, the web page and Word file written as given, and a PDF
/// printed page by page. The save panel itself needs a person, so it is
/// exercised live.
final class ExportedFileTests: XCTestCase {
  private var scratchDirectory: URL!

  override func setUpWithError() throws {
    try super.setUpWithError()
    scratchDirectory = FileManager.default.temporaryDirectory
      .appendingPathComponent("thintalk-export-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: scratchDirectory, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    if let scratchDirectory {
      try? FileManager.default.removeItem(at: scratchDirectory)
    }
    scratchDirectory = nil
    try super.tearDownWithError()
  }

  func test_should_read_each_format_with_the_content_it_needs() throws {
    let webPage = try XCTUnwrap(
      ExportedFileRequest(payload: ["format": "html", "suggestedName": "Mitosis", "html": "<p>Hi</p>"]))
    let wordFile = try XCTUnwrap(
      ExportedFileRequest(payload: ["format": "docx", "suggestedName": "Mitosis", "base64": Data("PK".utf8).base64EncodedString()]))

    XCTAssertEqual(webPage.format, .html)
    XCTAssertEqual(webPage.suggestedFileName, "Mitosis.html")
    XCTAssertEqual(wordFile.documentData, Data("PK".utf8))
    XCTAssertEqual(wordFile.suggestedFileName, "Mitosis.docx")
  }

  func test_should_refuse_a_request_missing_its_format_or_content() {
    let refusedPayloads: [[String: Any]] = [
      [:],
      ["format": "exe", "html": "<p>Hi</p>"],
      ["format": "pdf"],
      ["format": "html", "html": ""],
      ["format": "docx", "base64": "not base64!"],
      ["format": "docx", "base64": ""],
      ["format": "docx", "html": "<p>Word needs bytes</p>"],
    ]
    for refusedPayload in refusedPayloads {
      XCTAssertNil(ExportedFileRequest(payload: refusedPayload), "\(refusedPayload)")
    }
  }

  /// The name is the page's suggestion and nothing more: it can never name a
  /// folder, hide the file, or be empty.
  func test_should_offer_only_a_plain_visible_name() {
    XCTAssertEqual(ExportedFileRequest.cleanedName("../../Library/LaunchAgents/x"), "Library LaunchAgents x")
    XCTAssertEqual(ExportedFileRequest.cleanedName(".hidden"), "hidden")
    XCTAssertEqual(ExportedFileRequest.cleanedName("Mitosis: the phases"), "Mitosis the phases")
    XCTAssertEqual(ExportedFileRequest.cleanedName("line\nbreak\u{0007}"), "line break")
    XCTAssertEqual(ExportedFileRequest.cleanedName("   "), "Conversation")
    XCTAssertEqual(ExportedFileRequest.cleanedName(String(repeating: "a", count: 200)).count, 80)
    XCTAssertEqual(ExportedFileRequest.cleanedName("Café 日本語"), "Café 日本語")
  }

  func test_should_give_each_format_the_content_type_the_save_panel_filters_on() {
    XCTAssertEqual(ExportedFileFormat.pdf.contentType, .pdf)
    XCTAssertEqual(ExportedFileFormat.html.contentType, .html)
    XCTAssertEqual(ExportedFileFormat.docx.contentType.preferredFilenameExtension, "docx")
  }

  func test_should_write_a_web_page_and_a_word_file_as_given_replacing_an_older_file() throws {
    let pageURL = scratchDirectory.appendingPathComponent("Mitosis.html")
    try Data("an older file".utf8).write(to: pageURL)
    let webPage = ExportedFileRequest(format: .html, suggestedName: "Mitosis", html: "<p>Café</p>", documentData: nil)
    let wordFile = ExportedFileRequest(format: .docx, suggestedName: "Mitosis", html: nil, documentData: Data([0x50, 0x4B, 0x03, 0x04]))
    let wordURL = scratchDirectory.appendingPathComponent("Mitosis.docx")

    try ExportedFileWriter.write(webPage, to: pageURL)
    try ExportedFileWriter.write(wordFile, to: wordURL)

    XCTAssertEqual(try String(contentsOf: pageURL, encoding: .utf8), "<p>Café</p>")
    XCTAssertEqual(try Data(contentsOf: wordURL), Data([0x50, 0x4B, 0x03, 0x04]))
  }

  func test_should_refuse_to_write_a_pdf_without_printing_it() {
    let pdfRequest = ExportedFileRequest(format: .pdf, suggestedName: "Mitosis", html: "<p>Hi</p>", documentData: nil)
    XCTAssertThrowsError(try ExportedFileWriter.write(pdfRequest, to: scratchDirectory.appendingPathComponent("x.pdf"))) {
      XCTAssertEqual($0 as? ExportedFileWriter.WriteError, .needsPrinting)
    }
  }

  func test_should_report_a_saved_file_by_its_name_alone() {
    XCTAssertEqual(ExportedFileSaveOutcome.saved(fileName: "Mitosis.pdf").bridgePayload, ["outcome": "saved", "fileName": "Mitosis.pdf"])
    XCTAssertEqual(ExportedFileSaveOutcome.cancelled.bridgePayload, ["outcome": "cancelled"])
  }

  /// A long answer must come out as several pages of the Mac's paper, with
  /// its text still text: WebKit's own PDF call would draw one tall page.
  @MainActor
  func test_should_print_a_long_page_as_several_paper_pages_of_selectable_text() async throws {
    let paragraph = "<p>Mitosis divides one cell into two identical cells, phase by phase, so the page keeps filling up.</p>"
    let html = "<!DOCTYPE html><html><body><h1>Mitosis explained</h1>\(String(repeating: paragraph, count: 120))</body></html>"
    let pdfURL = scratchDirectory.appendingPathComponent("Mitosis.pdf")

    try await ExportPDFPrinter().printPDF(html: html, to: pdfURL)

    let printedDocument = try XCTUnwrap(PDFDocument(url: pdfURL))
    XCTAssertGreaterThan(printedDocument.pageCount, 1)
    let firstPage = try XCTUnwrap(printedDocument.page(at: 0))
    let paperSize = NSPrintInfo.shared.paperSize
    XCTAssertEqual(firstPage.bounds(for: .mediaBox).size.width, paperSize.width, accuracy: 1)
    XCTAssertEqual(firstPage.bounds(for: .mediaBox).size.height, paperSize.height, accuracy: 1)
    XCTAssertTrue(firstPage.string?.contains("Mitosis explained") == true, "the text stays selectable text")
  }

  @MainActor
  func test_should_print_a_short_page_as_one_page() async throws {
    let pdfURL = scratchDirectory.appendingPathComponent("Short.pdf")

    try await ExportPDFPrinter().printPDF(html: "<p>One line.</p>", to: pdfURL)

    XCTAssertEqual(PDFDocument(url: pdfURL)?.pageCount, 1)
  }

  /// WebKit lays a printed page out 1.25 times wider than the paper's text
  /// column, so a CSS pixel prints as 0.8 of a point, not 0.75, and every
  /// size comes out sixteen fifteenths too large: a 22-point title was 23.47
  /// points on paper. The export stylesheet zooms its printed body by fifteen
  /// sixteenths to undo that; this pins the engine behaviour it relies on.
  /// Scaling the print job instead was tried and cut tables across pages.
  @MainActor
  func test_should_print_text_at_its_css_size_when_the_page_zooms_by_fifteen_sixteenths() async throws {
    let trueSizeURL = scratchDirectory.appendingPathComponent("TrueSize.pdf")
    let enlargedURL = scratchDirectory.appendingPathComponent("Enlarged.pdf")
    let paragraph = "<p style=\"font: 20pt Helvetica; margin: 0\">Sized</p>"

    try await ExportPDFPrinter().printPDF(
      html: "<!DOCTYPE html><html><head><style>@media print { body { zoom: 0.9375; } }</style></head><body>\(paragraph)</body></html>",
      to: trueSizeURL)
    try await ExportPDFPrinter().printPDF(html: "<!DOCTYPE html><html><body>\(paragraph)</body></html>", to: enlargedURL)

    XCTAssertEqual(try printedFontSize(at: trueSizeURL), 20, accuracy: 0.05)
    XCTAssertEqual(try printedFontSize(at: enlargedURL), 20 * 16 / 15, accuracy: 0.05)
  }

  private func printedFontSize(at pdfURL: URL) throws -> CGFloat {
    let printedPage = try XCTUnwrap(PDFDocument(url: pdfURL)?.page(at: 0))
    let printedText = try XCTUnwrap(printedPage.attributedString)
    let printedFont = try XCTUnwrap(printedText.attribute(.font, at: 0, effectiveRange: nil) as? NSFont)
    return printedFont.pointSize
  }

  @MainActor
  func test_should_write_into_a_given_folder_under_the_suggested_name_without_a_panel() async throws {
    let webPageRequest = ExportedFileRequest(format: .html, suggestedName: "Mitosis", html: "<p>Hi</p>", documentData: nil)

    let saveOutcome = try await ExportedFileSaver.save(webPageRequest, into: scratchDirectory)

    XCTAssertEqual(saveOutcome, .saved(fileName: "Mitosis.html"))
    XCTAssertEqual(try String(contentsOf: scratchDirectory.appendingPathComponent("Mitosis.html"), encoding: .utf8), "<p>Hi</p>")
  }

  @MainActor
  func test_should_print_a_pdf_into_a_given_folder_without_a_panel() async throws {
    let pdfRequest = ExportedFileRequest(format: .pdf, suggestedName: "Short", html: "<p>One line.</p>", documentData: nil)

    let saveOutcome = try await ExportedFileSaver.save(pdfRequest, into: scratchDirectory)

    XCTAssertEqual(saveOutcome, .saved(fileName: "Short.pdf"))
    XCTAssertEqual(PDFDocument(url: scratchDirectory.appendingPathComponent("Short.pdf"))?.pageCount, 1)
  }

  @MainActor
  func test_should_refuse_a_folder_that_is_missing_or_is_a_file() async throws {
    let webPageRequest = ExportedFileRequest(format: .html, suggestedName: "Mitosis", html: "<p>Hi</p>", documentData: nil)
    let plainFile = scratchDirectory.appendingPathComponent("plain.txt")
    try Data("x".utf8).write(to: plainFile)

    for unusableFolder in [scratchDirectory.appendingPathComponent("absent"), plainFile] {
      do {
        _ = try await ExportedFileSaver.save(webPageRequest, into: unusableFolder)
        XCTFail("saved into \(unusableFolder.lastPathComponent)")
      } catch {
        XCTAssertEqual(error as? ExportedFileSaver.ExportFolderError, .folderMissing)
      }
    }
  }
}
