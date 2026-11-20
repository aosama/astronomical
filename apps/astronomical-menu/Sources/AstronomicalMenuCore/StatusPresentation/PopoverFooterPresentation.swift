// Which exits the popover's footer offers. One quiet row keeps the status above it the loudest
// thing in the popover, and at most one control carries the accent colour.

import Foundation

struct PopoverFooterPresentation: Equatable {
  /// A full-width accent button, shown only on a Mac with no model yet: downloading one is the
  /// only thing that person can usefully do. A stopped engine reports no models either, and
  /// offering a download then would point away from the real problem.
  let offersModelDownload: Bool
  /// Chat carries the accent whenever there is a model to chat with; it gives the accent up to
  /// the download button otherwise.
  let chatCarriesAccent: Bool
  /// The engine serves Observatory's page, so once it has stopped answering the page cannot
  /// open and Restart takes its place: the one action that helps.
  let offersRestartInPlaceOfObservatory: Bool

  init(statusDocument: DaemonStatusDocument, hasDiscoveredModels: Bool) {
    let engineIsAnswering = statusDocument.status != "unavailable"
    offersModelDownload = engineIsAnswering && !hasDiscoveredModels
    chatCarriesAccent = hasDiscoveredModels
    offersRestartInPlaceOfObservatory = !engineIsAnswering
  }
}
