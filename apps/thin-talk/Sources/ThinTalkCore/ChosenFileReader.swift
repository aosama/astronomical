import Foundation

/// Errors reading a file the user chose for a tool.
public enum ChosenFileReaderError: Error, Equatable, Sendable {
  /// The file is larger than the tool accepts.
  case fileTooLarge(byteCount: Int, maximumByteCount: Int)
  /// The file is not UTF-8 text (for the text tool) or not a PNG or JPEG image.
  case unsupportedContent
}

/// Reads a file the user picked in an open panel, under the limits each tool
/// sets. Kept apart from the panel so the limits and type checks are tested
/// without any user interface.
public enum ChosenFileReader {
  /// The most text the read-file tool loads. The model sees at most 16 KB of
  /// it (the page's result cap), so a larger file would only cost memory.
  public static let maximumTextFileByteCount = 256 * 1024
  /// The largest image the describe-image tool sends to a vision model. The
  /// daemon refuses an image over 7 MiB decoded (its per-image request
  /// budget), so a larger one is refused here with a clearer reason.
  public static let maximumImageFileByteCount = 7 * 1024 * 1024

  /// One chosen text file: its name (never its path) and its contents.
  public struct TextFile: Equatable, Sendable {
    public let fileName: String
    public let text: String

    public init(fileName: String, text: String) {
      self.fileName = fileName
      self.text = text
    }
  }

  /// One chosen image: its name, its bytes as base64, and its media type.
  public struct ImageFile: Equatable, Sendable {
    public let fileName: String
    public let base64: String
    public let mimeType: String

    public init(fileName: String, base64: String, mimeType: String) {
      self.fileName = fileName
      self.base64 = base64
      self.mimeType = mimeType
    }
  }

  public static func textFile(at fileURL: URL) throws -> TextFile {
    let textFileBytes = try boundedContents(of: fileURL, maximumByteCount: maximumTextFileByteCount)
    guard let text = String(data: textFileBytes, encoding: .utf8), !text.contains("\u{0}") else {
      throw ChosenFileReaderError.unsupportedContent
    }
    return TextFile(fileName: fileURL.lastPathComponent, text: text)
  }

  public static func imageFile(at fileURL: URL) throws -> ImageFile {
    let imageData = try boundedContents(of: fileURL, maximumByteCount: maximumImageFileByteCount)
    guard let mimeType = imageMimeType(of: imageData) else {
      throw ChosenFileReaderError.unsupportedContent
    }
    return ImageFile(fileName: fileURL.lastPathComponent, base64: imageData.base64EncodedString(), mimeType: mimeType)
  }

  /// Reads at most one byte past the limit, so a huge file (or one whose size
  /// is unknown, or that grows while being read) is refused without loading
  /// it all.
  private static func boundedContents(of fileURL: URL, maximumByteCount: Int) throws -> Data {
    let fileHandle = try FileHandle(forReadingFrom: fileURL)
    defer { try? fileHandle.close() }
    let fileContents = try fileHandle.read(upToCount: maximumByteCount + 1) ?? Data()
    guard fileContents.count <= maximumByteCount else {
      let resourceValues = try? fileURL.resourceValues(forKeys: [.fileSizeKey])
      let reportedSize = resourceValues?.fileSize ?? fileContents.count
      throw ChosenFileReaderError.fileTooLarge(byteCount: reportedSize, maximumByteCount: maximumByteCount)
    }
    return fileContents
  }

  private static func imageMimeType(of imageData: Data) -> String? {
    let leadingBytes = [UInt8](imageData.prefix(8))
    if leadingBytes == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] { return "image/png" }
    if leadingBytes.starts(with: [0xFF, 0xD8, 0xFF]) { return "image/jpeg" }
    return nil
  }
}
