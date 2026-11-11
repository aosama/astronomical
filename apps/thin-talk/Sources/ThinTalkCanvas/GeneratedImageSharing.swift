import AppKit
import UniformTypeIdentifiers

/// Getting a generated picture out of a conversation: saved where the person
/// chooses, or put on the pasteboard to paste elsewhere.
///
/// The page never learns a file path, so it names a stored image and the host
/// does the rest. Saving goes through a save panel rather than straight into
/// Downloads: the panel works the same inside the App Store sandbox, where a
/// direct write would need an entitlement, and it lets the person pick the
/// folder and the name.
@MainActor
public enum GeneratedImageSharing {
  /// The name offered in the save panel: "Picture 2026-11-10 at 19.55.03.png".
  /// A stored image is named by a random identifier, which says nothing to the
  /// person and sorts nowhere; the moment it was saved does both. The time
  /// uses dots because a colon cannot be part of a file name in Finder.
  nonisolated public static func suggestedFileName(
    forStoredImageNamed storedImageName: String, savedAt: Date, timeZone: TimeZone = .current
  ) -> String {
    let timestampFormatter = DateFormatter()
    timestampFormatter.locale = Locale(identifier: "en_US_POSIX")
    timestampFormatter.timeZone = timeZone
    timestampFormatter.dateFormat = "yyyy-MM-dd 'at' HH.mm.ss"
    let fileExtension = storedImageName.hasSuffix(".jpg") ? "jpg" : "png"
    return "Picture \(timestampFormatter.string(from: savedAt)).\(fileExtension)"
  }

  /// Asks where to save a copy of the image, as a sheet on `window`, and
  /// copies it there. A cancelled panel does nothing.
  public static func saveCopy(of imageURL: URL, from window: NSWindow?) {
    let savePanel = NSSavePanel()
    savePanel.nameFieldStringValue = suggestedFileName(forStoredImageNamed: imageURL.lastPathComponent, savedAt: Date())
    savePanel.directoryURL = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first
    savePanel.allowedContentTypes = [imageURL.pathExtension == "jpg" ? UTType.jpeg : UTType.png]
    savePanel.canCreateDirectories = true
    let copyToChosenPlace: (NSApplication.ModalResponse) -> Void = { savePanelAnswer in
      guard savePanelAnswer == .OK, let destinationURL = savePanel.url else { return }
      do {
        try writeCopy(of: imageURL, to: destinationURL)
      } catch {
        NSLog("ThinTalk canvas: could not save the picture: \(error.localizedDescription)")
        NSSound.beep()
      }
    }
    if let window {
      savePanel.beginSheetModal(for: window, completionHandler: copyToChosenPlace)
    } else {
      copyToChosenPlace(savePanel.runModal())
    }
  }

  /// Copies the image file, replacing a file already at the destination: the
  /// save panel has by then asked the person whether to replace it.
  ///
  /// The old file is swapped out only once the new copy exists. Removing it
  /// first would leave the person with neither picture if the copy then
  /// failed, on a full disk for instance. The copy is staged in the folder
  /// the system provides for replacing that destination, which is on the same
  /// volume and reachable inside the sandbox, where the destination's own
  /// folder is not.
  nonisolated public static func writeCopy(
    of imageURL: URL, to destinationURL: URL, fileManager: FileManager = .default
  ) throws {
    guard fileManager.fileExists(atPath: destinationURL.path) else {
      try fileManager.copyItem(at: imageURL, to: destinationURL)
      return
    }
    let stagingDirectoryURL = try fileManager.url(
      for: .itemReplacementDirectory, in: .userDomainMask, appropriateFor: destinationURL, create: true)
    defer { try? fileManager.removeItem(at: stagingDirectoryURL) }
    let stagedCopyURL = stagingDirectoryURL.appendingPathComponent(destinationURL.lastPathComponent)
    try fileManager.copyItem(at: imageURL, to: stagedCopyURL)
    _ = try fileManager.replaceItemAt(destinationURL, withItemAt: stagedCopyURL)
  }

  /// Puts the image on the pasteboard so it pastes into a document, a message
  /// or a slide. False when the file is not an image AppKit can read.
  @discardableResult
  public static func copyToPasteboard(_ imageURL: URL, pasteboard: NSPasteboard = .general) -> Bool {
    guard let image = NSImage(contentsOf: imageURL), image.isValid else { return false }
    pasteboard.clearContents()
    return pasteboard.writeObjects([image])
  }
}
