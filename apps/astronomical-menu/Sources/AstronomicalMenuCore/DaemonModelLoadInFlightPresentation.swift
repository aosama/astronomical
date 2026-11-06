// Owns the words the menu uses for a model load, which stand between the user and a menu bar
// that looks idle while the previous model is already gone.

import Foundation

extension DaemonStatusDocument.ModelLoadInFlight {
  var headlineTitle: String {
    guard let modelIdentifier else { return "Loading a model" }
    return "Loading \(modelIdentifier)"
  }

  var menuBarTitle: String { "Loading model" }
  var phaseTitle: String { "Loading" }
  var flightTitle: String { "Loading the model" }
  var progressTitle: String { "Loading the model" }
  var elapsedTimeMetricTitle: String { "Elapsed" }
  // A load reports no clock, and a frozen "0.0 s" would read as a stuck load.
  var elapsedTimeTitle: String { "In progress" }
}
