// Presents the bounded status polling failure separately from ordinary unavailable telemetry, so
// users can distinguish a stopped server from an incompatible local wire contract.

import SwiftUI

struct StatusRefreshErrorView: View {
  let message: String

  var body: some View {
    Text(message)
      .font(.caption)
      .foregroundStyle(.secondary)
      .lineLimit(3)
      .accessibilityLabel("Server status error: \(message)")
  }
}
