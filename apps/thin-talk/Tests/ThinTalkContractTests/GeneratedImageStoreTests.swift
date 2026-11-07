import Foundation
import ThinTalkCanvas
import ThinTalkCore
import XCTest

/// The generated-image store reads files by a name the page sends back, so
/// its contract is exercised directly: only PNG and JPEG bytes are kept, only
/// names the store itself produced are honoured, and a crafted name never
/// reaches the disk.
final class GeneratedImageStoreTests: XCTestCase {
  private var imageDirectory: URL!

  override func setUpWithError() throws {
    try super.setUpWithError()
    imageDirectory = FileManager.default.temporaryDirectory
      .appendingPathComponent("thintalk-image-store-tests-\(UUID().uuidString)", isDirectory: true)
  }

  override func tearDownWithError() throws {
    if let imageDirectory {
      try? FileManager.default.removeItem(at: imageDirectory)
    }
    imageDirectory = nil
    try super.tearDownWithError()
  }

  private let pngBytes = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x01])
  private let jpegBytes = Data([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10])

  func test_should_save_png_and_jpeg_images_and_find_them_by_name() throws {
    let store = GeneratedImageStore(directory: imageDirectory)
    let pngName = try store.save(imageData: pngBytes)
    let jpegName = try store.save(imageData: jpegBytes)
    XCTAssertTrue(pngName.hasSuffix(".png"))
    XCTAssertTrue(jpegName.hasSuffix(".jpg"))
    XCTAssertEqual(try Data(contentsOf: try XCTUnwrap(store.fileURL(forImageName: pngName))), pngBytes)
    XCTAssertNotNil(store.fileURL(forImageName: jpegName))
  }

  func test_should_refuse_bytes_that_are_not_an_image() {
    let store = GeneratedImageStore(directory: imageDirectory)
    for notAnImage in [Data(), Data("<svg/>".utf8), Data([0x89, 0x50, 0x4E])] {
      XCTAssertThrowsError(try store.save(imageData: notAnImage)) { error in
        XCTAssertEqual(error as? GeneratedImageStoreError, .unsupportedImage)
      }
    }
  }

  func test_should_refuse_an_image_one_byte_over_the_limit() {
    let store = GeneratedImageStore(directory: imageDirectory)
    var oversizedImage = pngBytes
    oversizedImage.append(Data(count: GeneratedImageStore.maximumImageByteCount - pngBytes.count + 1))
    XCTAssertThrowsError(try store.save(imageData: oversizedImage)) { error in
      XCTAssertEqual(error as? GeneratedImageStoreError, .imageTooLarge(byteCount: GeneratedImageStore.maximumImageByteCount + 1))
    }
  }

  func test_should_honour_only_names_the_store_produces() {
    let storedName = "\(UUID().uuidString.lowercased()).png"
    XCTAssertTrue(GeneratedImageStore.isStoredImageName(storedName))
    let craftedNames = [
      "../sessions/x.json", "\(UUID().uuidString).png", "\(UUID().uuidString.lowercased()).gif",
      "\(UUID().uuidString.lowercased()).png.png", "a.png", "", ".png",
    ]
    for craftedName in craftedNames {
      XCTAssertFalse(GeneratedImageStore.isStoredImageName(craftedName), craftedName)
    }
  }

  func test_should_find_nothing_for_a_valid_name_that_was_never_saved() {
    let store = GeneratedImageStore(directory: imageDirectory)
    XCTAssertNil(store.fileURL(forImageName: "\(UUID().uuidString.lowercased()).png"))
  }

  func test_should_accept_an_image_of_exactly_the_limit() throws {
    let store = GeneratedImageStore(directory: imageDirectory)
    var imageAtLimit = pngBytes
    imageAtLimit.append(Data(count: GeneratedImageStore.maximumImageByteCount - pngBytes.count))
    XCTAssertNotNil(store.fileURL(forImageName: try store.save(imageData: imageAtLimit)))
  }

  func test_should_save_base64_and_refuse_text_that_is_not_base64_or_too_long() throws {
    let store = GeneratedImageStore(directory: imageDirectory)
    XCTAssertNotNil(store.fileURL(forImageName: try store.save(base64Image: pngBytes.base64EncodedString())))
    XCTAssertThrowsError(try store.save(base64Image: "not base64!")) { error in
      XCTAssertEqual(error as? GeneratedImageStoreError, .unsupportedImage)
    }
    let overlongBase64 = String(repeating: "A", count: GeneratedImageStore.maximumBase64ImageLength + 4)
    XCTAssertThrowsError(try store.save(base64Image: overlongBase64)) { error in
      guard case .imageTooLarge = error as? GeneratedImageStoreError else { return XCTFail("\(error)") }
    }
  }

  func test_should_delete_named_images_and_ignore_crafted_or_missing_names() throws {
    let store = GeneratedImageStore(directory: imageDirectory)
    let keptName = try store.save(imageData: pngBytes)
    let deletedName = try store.save(imageData: jpegBytes)
    let outsideFile = imageDirectory.deletingLastPathComponent().appendingPathComponent("outside-\(UUID().uuidString).json")
    try Data("{}".utf8).write(to: outsideFile)
    defer { try? FileManager.default.removeItem(at: outsideFile) }
    try store.delete(imageNames: [deletedName, "../\(outsideFile.lastPathComponent)", "\(UUID().uuidString.lowercased()).png"])
    XCTAssertNil(store.fileURL(forImageName: deletedName))
    XCTAssertNotNil(store.fileURL(forImageName: keptName))
    XCTAssertTrue(FileManager.default.fileExists(atPath: outsideFile.path))
    try store.delete(imageNames: [deletedName])
  }

}
