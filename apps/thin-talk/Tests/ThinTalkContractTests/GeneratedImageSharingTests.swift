import AppKit
import ThinTalkCanvas
import XCTest

/// Saving and copying a generated picture: the name the save panel offers, the
/// copy written to the chosen place, and the image put on the pasteboard.
final class GeneratedImageSharingTests: XCTestCase {
  private var scratchDirectory: URL!

  override func setUpWithError() throws {
    try super.setUpWithError()
    scratchDirectory = FileManager.default.temporaryDirectory
      .appendingPathComponent("thintalk-image-sharing-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: scratchDirectory, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    if let scratchDirectory {
      try? FileManager.default.removeItem(at: scratchDirectory)
    }
    scratchDirectory = nil
    try super.tearDownWithError()
  }

  /// A real one-pixel PNG, so AppKit can read it back as an image.
  private func writeOnePixelPNG(named fileName: String) throws -> URL {
    let bitmap = try XCTUnwrap(
      NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: 1, pixelsHigh: 1, bitsPerSample: 8, samplesPerPixel: 4,
        hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0))
    let pngData = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
    let imageURL = scratchDirectory.appendingPathComponent(fileName)
    try pngData.write(to: imageURL)
    return imageURL
  }

  func test_should_offer_a_dated_file_name_that_keeps_the_image_type() throws {
    let savedAt = Date(timeIntervalSince1970: 1_794_355_503)
    let greenwich = try XCTUnwrap(TimeZone(identifier: "UTC"))

    XCTAssertEqual(
      GeneratedImageSharing.suggestedFileName(
        forStoredImageNamed: "0b1d7f0e-3a52-4c8e-9d11-2f6a4c1e7b90.png", savedAt: savedAt, timeZone: greenwich),
      "Picture 2026-11-11 at 00.05.03.png")
    XCTAssertEqual(
      GeneratedImageSharing.suggestedFileName(
        forStoredImageNamed: "0b1d7f0e-3a52-4c8e-9d11-2f6a4c1e7b90.jpg", savedAt: savedAt, timeZone: greenwich),
      "Picture 2026-11-11 at 00.05.03.jpg")
  }

  /// Finder cannot show a colon in a file name, and the stored identifier
  /// means nothing to the person, so neither may reach the offered name.
  func test_should_offer_a_name_without_colons_or_the_stored_identifier() {
    let offeredName = GeneratedImageSharing.suggestedFileName(
      forStoredImageNamed: "0b1d7f0e-3a52-4c8e-9d11-2f6a4c1e7b90.png", savedAt: Date())

    XCTAssertFalse(offeredName.contains(":"))
    XCTAssertFalse(offeredName.contains("0b1d7f0e"))
    XCTAssertFalse(offeredName.contains("/"))
  }

  func test_should_write_a_copy_and_leave_the_stored_image_in_place() throws {
    let storedImageURL = try writeOnePixelPNG(named: "stored.png")
    let destinationURL = scratchDirectory.appendingPathComponent("Picture saved.png")

    try GeneratedImageSharing.writeCopy(of: storedImageURL, to: destinationURL)

    XCTAssertEqual(try Data(contentsOf: destinationURL), try Data(contentsOf: storedImageURL))
    XCTAssertTrue(FileManager.default.fileExists(atPath: storedImageURL.path))
  }

  /// The save panel has already asked whether to replace an existing file.
  func test_should_replace_a_file_already_at_the_chosen_place() throws {
    let storedImageURL = try writeOnePixelPNG(named: "stored.png")
    let destinationURL = scratchDirectory.appendingPathComponent("Picture saved.png")
    try Data("an older file".utf8).write(to: destinationURL)

    try GeneratedImageSharing.writeCopy(of: storedImageURL, to: destinationURL)

    XCTAssertEqual(try Data(contentsOf: destinationURL), try Data(contentsOf: storedImageURL))
  }

  /// Agreeing to replace a file is not agreeing to lose it: when the new copy
  /// cannot be made, the file that was there must still be there.
  func test_should_keep_the_older_file_when_its_replacement_cannot_be_copied() throws {
    let missingImageURL = scratchDirectory.appendingPathComponent("deleted.png")
    let destinationURL = scratchDirectory.appendingPathComponent("Picture saved.png")
    try Data("an older file".utf8).write(to: destinationURL)

    XCTAssertThrowsError(try GeneratedImageSharing.writeCopy(of: missingImageURL, to: destinationURL))

    XCTAssertEqual(try Data(contentsOf: destinationURL), Data("an older file".utf8))
  }

  func test_should_fail_to_write_a_copy_of_an_image_that_is_gone() {
    let missingImageURL = scratchDirectory.appendingPathComponent("deleted.png")
    let destinationURL = scratchDirectory.appendingPathComponent("Picture saved.png")

    XCTAssertThrowsError(try GeneratedImageSharing.writeCopy(of: missingImageURL, to: destinationURL))
    XCTAssertFalse(FileManager.default.fileExists(atPath: destinationURL.path))
  }

  @MainActor
  func test_should_put_the_image_on_the_pasteboard() throws {
    let storedImageURL = try writeOnePixelPNG(named: "stored.png")
    let scratchPasteboard = NSPasteboard(name: NSPasteboard.Name("thintalk-image-sharing-\(UUID().uuidString)"))
    defer { scratchPasteboard.releaseGlobally() }

    XCTAssertTrue(GeneratedImageSharing.copyToPasteboard(storedImageURL, pasteboard: scratchPasteboard))

    XCTAssertTrue(scratchPasteboard.canReadObject(forClasses: [NSImage.self], options: nil))
  }

  @MainActor
  func test_should_refuse_to_copy_a_file_that_is_not_an_image_and_keep_the_pasteboard() throws {
    let notAnImageURL = scratchDirectory.appendingPathComponent("stored.png")
    try Data("not image bytes".utf8).write(to: notAnImageURL)
    let scratchPasteboard = NSPasteboard(name: NSPasteboard.Name("thintalk-image-sharing-\(UUID().uuidString)"))
    defer { scratchPasteboard.releaseGlobally() }
    scratchPasteboard.clearContents()
    scratchPasteboard.setString("what was copied before", forType: .string)

    XCTAssertFalse(GeneratedImageSharing.copyToPasteboard(notAnImageURL, pasteboard: scratchPasteboard))

    XCTAssertEqual(scratchPasteboard.string(forType: .string), "what was copied before")
  }
}
