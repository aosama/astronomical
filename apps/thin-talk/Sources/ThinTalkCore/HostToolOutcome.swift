import Foundation

/// How one use of a host tool ended, as the page receives it.
///
/// "Refused" means the user saw the sheet and said no. "Unavailable" means the
/// user was never asked (a tool outside the allowlist, no conversation, or no
/// visible window to ask on), so the page can tell the model the difference
/// instead of claiming a refusal nobody made.
public enum HostToolOutcome: Equatable, Sendable {
  /// The user allowed a tool that needs nothing more (conversation search).
  case approved
  /// The user allowed reading a file and chose this one.
  case chosenTextFile(ChosenFileReader.TextFile)
  /// The user allowed looking at an image and chose this one.
  case chosenImageFile(ChosenFileReader.ImageFile)
  /// The user said no on the sheet.
  case refused
  /// The user allowed it, then cancelled the open panel.
  case nothingChosen
  /// The page stopped the turn while the sheet or panel was up.
  case cancelled
  /// Nobody was asked; the reason says why.
  case unavailable(reason: String)

  /// The JSON object the bridge sends back to the page.
  public var bridgePayload: [String: String] {
    switch self {
    case .approved:
      return ["outcome": "approved"]
    case .chosenTextFile(let textFile):
      return ["outcome": "chosen", "fileName": textFile.fileName, "text": textFile.text]
    case .chosenImageFile(let imageFile):
      return ["outcome": "chosen", "fileName": imageFile.fileName, "base64": imageFile.base64, "mimeType": imageFile.mimeType]
    case .refused:
      return ["outcome": "refused"]
    case .nothingChosen:
      return ["outcome": "nothingChosen"]
    case .cancelled:
      return ["outcome": "cancelled"]
    case .unavailable(let reason):
      return ["outcome": "unavailable", "reason": reason]
    }
  }
}
