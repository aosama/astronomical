// A link that opens the Models page on one model, so a web page or a model
// card can say "Open in Astronomical":
// `astronomical://models/huggingface/OWNER/NAME`.
//
// Any page on the web can open such a link, so it is kept harmless three
// ways. It only opens the Observatory's Models page, where the person still
// has to press Get before anything downloads. The model name is limited to
// the characters Hugging Face allows in an owner and a name, so the link
// cannot carry anything else into the page's address. And the Models page
// offers only the models in Astronomical's own checked list, whatever the
// link names. The contract lives in one pure, testable place.

import Foundation

public enum ModelLink {
  public static let destinationName = "models"
  public static let sourceName = "huggingface"
  public static let modelQueryParameterName = "model"
  /// Hugging Face caps an owner and a name at 96 characters each; a longer
  /// link is not a model name.
  public static let maximumModelIDCharacters = 200

  private static let allowedNameCharacters = CharacterSet(
    charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")

  public static func url(huggingfaceModelID: String, channel: ApplicationChannel) -> URL? {
    URL(string: "\(channel.chatURLScheme)://\(destinationName)/\(sourceName)/\(huggingfaceModelID)")
  }

  /// The `OWNER/NAME` a link names, or nil when the link is not a model link
  /// for this channel or the name is not a plausible Hugging Face model name.
  public static func huggingfaceModelID(from openedURL: URL, channel: ApplicationChannel) -> String? {
    guard openedURL.scheme?.lowercased() == channel.chatURLScheme,
      openedURL.host?.lowercased() == destinationName
    else { return nil }
    let linkParts = openedURL.pathComponents.filter { $0 != "/" }
    guard linkParts.count == 3, linkParts[0].lowercased() == sourceName,
      isPlausibleNamePart(linkParts[1]), isPlausibleNamePart(linkParts[2])
    else { return nil }
    let huggingfaceModelID = "\(linkParts[1])/\(linkParts[2])"
    return huggingfaceModelID.count <= maximumModelIDCharacters ? huggingfaceModelID : nil
  }

  /// The Models page's path with the linked model in its query.
  public static func observatoryPath(huggingfaceModelID: String) -> String {
    var pageComponents = URLComponents()
    pageComponents.path = "/download-models"
    pageComponents.queryItems = [URLQueryItem(name: modelQueryParameterName, value: huggingfaceModelID)]
    return pageComponents.string ?? "/download-models"
  }

  /// A name part must be made of the allowed characters and must not be a
  /// dot-only path step such as "..".
  private static func isPlausibleNamePart(_ namePart: String) -> Bool {
    guard !namePart.isEmpty, namePart.unicodeScalars.allSatisfy(allowedNameCharacters.contains) else {
      return false
    }
    return namePart.contains { $0 != "." }
  }
}
