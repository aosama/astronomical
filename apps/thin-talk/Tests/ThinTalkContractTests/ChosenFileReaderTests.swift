import Foundation
import ThinTalkCore
import XCTest

/// The limits and type checks applied to a file the user chose for a tool.
/// Only the file's name ever leaves the reader, never its path.
final class ChosenFileReaderTests: XCTestCase {
  private var chosenFilesDirectory: URL!

  override func setUpWithError() throws {
    try super.setUpWithError()
    chosenFilesDirectory = FileManager.default.temporaryDirectory
      .appendingPathComponent("thintalk-chosen-files-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: chosenFilesDirectory, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    if let chosenFilesDirectory {
      try? FileManager.default.removeItem(at: chosenFilesDirectory)
    }
    chosenFilesDirectory = nil
    try super.tearDownWithError()
  }

  private func chosenFile(named fileName: String, contents: Data) throws -> URL {
    let fileURL = chosenFilesDirectory.appendingPathComponent(fileName)
    try contents.write(to: fileURL)
    return fileURL
  }

  func test_should_read_a_text_file_by_name_without_its_path() throws {
    let textFile = try ChosenFileReader.textFile(at: chosenFile(named: "sonnet.txt", contents: Data("Shall I compare thee".utf8)))
    XCTAssertEqual(textFile, .init(fileName: "sonnet.txt", text: "Shall I compare thee"))
  }

  func test_should_accept_text_at_the_limit_and_refuse_one_byte_over() throws {
    let textAtLimit = Data(repeating: 0x61, count: ChosenFileReader.maximumTextFileByteCount)
    XCTAssertEqual(try ChosenFileReader.textFile(at: chosenFile(named: "at-limit.txt", contents: textAtLimit)).text.count, textAtLimit.count)
    let textOverLimit = Data(repeating: 0x61, count: ChosenFileReader.maximumTextFileByteCount + 1)
    XCTAssertThrowsError(try ChosenFileReader.textFile(at: chosenFile(named: "over.txt", contents: textOverLimit))) { error in
      XCTAssertEqual(
        error as? ChosenFileReaderError,
        .fileTooLarge(byteCount: textOverLimit.count, maximumByteCount: ChosenFileReader.maximumTextFileByteCount))
    }
  }

  func test_should_refuse_a_file_that_is_not_text() throws {
    for binaryContents in [Data([0xFF, 0xFE, 0xFD]), Data("text\u{0}with a nul".utf8)] {
      XCTAssertThrowsError(try ChosenFileReader.textFile(at: chosenFile(named: "binary.dat", contents: binaryContents))) { error in
        XCTAssertEqual(error as? ChosenFileReaderError, .unsupportedContent)
      }
    }
  }

  func test_should_read_png_and_jpeg_images_with_their_media_type() throws {
    let pngBytes = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00])
    let pngImage = try ChosenFileReader.imageFile(at: chosenFile(named: "rose.png", contents: pngBytes))
    XCTAssertEqual(pngImage, .init(fileName: "rose.png", base64: pngBytes.base64EncodedString(), mimeType: "image/png"))
    let jpegImage = try ChosenFileReader.imageFile(at: chosenFile(named: "rose.jpg", contents: Data([0xFF, 0xD8, 0xFF, 0xE0])))
    XCTAssertEqual(jpegImage.mimeType, "image/jpeg")
  }

  func test_should_refuse_an_image_that_is_not_png_or_jpeg() throws {
    XCTAssertThrowsError(try ChosenFileReader.imageFile(at: chosenFile(named: "rose.gif", contents: Data("GIF89a".utf8)))) { error in
      XCTAssertEqual(error as? ChosenFileReaderError, .unsupportedContent)
    }
  }

  func test_should_accept_an_image_at_the_limit_and_refuse_one_byte_over() throws {
    var imageAtLimit = Data([0xFF, 0xD8, 0xFF, 0xE0])
    imageAtLimit.append(Data(count: ChosenFileReader.maximumImageFileByteCount - imageAtLimit.count))
    XCTAssertEqual(try ChosenFileReader.imageFile(at: chosenFile(named: "at-limit.jpg", contents: imageAtLimit)).mimeType, "image/jpeg")
    imageAtLimit.append(0)
    XCTAssertThrowsError(try ChosenFileReader.imageFile(at: chosenFile(named: "over.jpg", contents: imageAtLimit))) { error in
      guard case .fileTooLarge = error as? ChosenFileReaderError else { return XCTFail("\(error)") }
    }
  }

  func test_should_read_an_empty_text_file_and_fail_for_a_missing_one() throws {
    XCTAssertEqual(try ChosenFileReader.textFile(at: chosenFile(named: "empty.txt", contents: Data())).text, "")
    XCTAssertThrowsError(try ChosenFileReader.textFile(at: chosenFilesDirectory.appendingPathComponent("missing.txt")))
  }
}
