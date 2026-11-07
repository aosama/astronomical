import Foundation

/// Errors the generated-image store reports across the bridge.
public enum GeneratedImageStoreError: Error, Equatable, Sendable {
  /// The bytes are neither a PNG nor a JPEG image.
  case unsupportedImage
  /// The image is larger than the store accepts.
  case imageTooLarge(byteCount: Int)
}

/// Keeps the images Thin Talk's image tool generates, so a conversation that
/// shows one still shows it after a relaunch.
///
/// Images are deleted with their conversation: the page, which owns the
/// conversation format, sends the names it recorded when a conversation is
/// deleted. An image whose generation was stopped before its conversation was
/// saved has no owner and stays; the store cannot tell it apart without
/// reading conversations, which only the page does.
///
/// Each image gets a fresh random name; the page stores only that name and the
/// canvas loads the image through the private scheme's `generated` host. The
/// page therefore never learns a file path, and a name it sends back is checked
/// against a strict pattern before it touches the disk, so a crafted name
/// cannot read anything outside this directory.
public final class GeneratedImageStore: @unchecked Sendable {
  /// The largest image the store accepts. A 1024 by 1024 PNG from the image
  /// model is a few megabytes; the cap only stops a runaway payload.
  public static let maximumImageByteCount = 20 * 1024 * 1024

  private static let pngSignature: [UInt8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
  private static let jpegSignature: [UInt8] = [0xFF, 0xD8, 0xFF]

  private let directory: URL
  private let fileManager: FileManager
  private let queue = DispatchQueue(label: "astronomical.thintalk.generatedimagestore", qos: .userInitiated)

  public init(directory: URL, fileManager: FileManager = .default) {
    self.directory = directory
    self.fileManager = fileManager
  }

  /// The longest base64 text that can decode to an accepted image: four
  /// characters per three bytes, plus padding.
  public static let maximumBase64ImageLength = (maximumImageByteCount + 2) / 3 * 4

  /// Saves an image the page sends as base64. The length is checked before
  /// decoding so an oversized payload is refused without being copied again.
  public func save(base64Image: String) throws -> String {
    guard base64Image.utf8.count <= Self.maximumBase64ImageLength else {
      throw GeneratedImageStoreError.imageTooLarge(byteCount: base64Image.utf8.count / 4 * 3)
    }
    guard let imageData = Data(base64Encoded: base64Image) else {
      throw GeneratedImageStoreError.unsupportedImage
    }
    return try save(imageData: imageData)
  }

  /// Deletes the named images. Names the store could not have produced are
  /// ignored, and an image that is already gone is not an error, so deleting
  /// a conversation twice, or one whose images were never saved, succeeds.
  public func delete(imageNames: [String]) throws {
    try queue.sync {
      for imageName in imageNames where Self.isStoredImageName(imageName) {
        let imageURL = directory.appendingPathComponent(imageName)
        guard fileManager.fileExists(atPath: imageURL.path) else { continue }
        try fileManager.removeItem(at: imageURL)
      }
    }
  }

  /// Saves one image and returns the name the page stores and later loads it by.
  public func save(imageData: Data) throws -> String {
    guard imageData.count <= Self.maximumImageByteCount else {
      throw GeneratedImageStoreError.imageTooLarge(byteCount: imageData.count)
    }
    guard let fileExtension = Self.fileExtension(for: imageData) else {
      throw GeneratedImageStoreError.unsupportedImage
    }
    let imageName = "\(UUID().uuidString.lowercased()).\(fileExtension)"
    try queue.sync {
      try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
      try imageData.write(to: directory.appendingPathComponent(imageName), options: [.atomic])
    }
    return imageName
  }

  /// A stored image as base64 with its media type, for showing it to a model
  /// that can see images on a later turn. Nil for a name this store could not
  /// have produced, or an image that is gone (its conversation was deleted).
  public func base64Image(named imageName: String) throws -> (base64: String, mimeType: String)? {
    guard let imageURL = fileURL(forImageName: imageName) else { return nil }
    let imageData = try queue.sync { try Data(contentsOf: imageURL) }
    let mimeType = imageName.hasSuffix(".jpg") ? "image/jpeg" : "image/png"
    return (imageData.base64EncodedString(), mimeType)
  }

  /// The file behind a stored image name, or nil when the name is not one this
  /// store could have produced or no such image exists.
  public func fileURL(forImageName imageName: String) -> URL? {
    guard Self.isStoredImageName(imageName) else { return nil }
    let candidate = directory.appendingPathComponent(imageName)
    return fileManager.fileExists(atPath: candidate.path) ? candidate : nil
  }

  /// A lowercase UUID followed by `.png` or `.jpg`, exactly as `save` names
  /// images. Anything else (separators, `..`, other extensions) is refused.
  public static func isStoredImageName(_ imageName: String) -> Bool {
    let nameParts = imageName.split(separator: ".", omittingEmptySubsequences: false)
    guard nameParts.count == 2, ["png", "jpg"].contains(String(nameParts[1])) else { return false }
    let identifier = String(nameParts[0])
    guard identifier == identifier.lowercased(), let parsedUUID = UUID(uuidString: identifier) else { return false }
    return parsedUUID.uuidString.lowercased() == identifier
  }

  private static func fileExtension(for imageData: Data) -> String? {
    let leadingBytes = [UInt8](imageData.prefix(pngSignature.count))
    if leadingBytes.starts(with: pngSignature) && leadingBytes.count == pngSignature.count { return "png" }
    if leadingBytes.starts(with: jpegSignature) { return "jpg" }
    return nil
  }
}
