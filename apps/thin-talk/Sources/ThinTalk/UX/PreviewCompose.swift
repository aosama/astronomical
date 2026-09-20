import SwiftUI
import ThinTalkUI

/// Compose dock: large input area with send button.
///
/// Mirrors Copilot's composer: very tall input area, large controls,
/// large placeholder text.
struct PreviewComposeDock: View {
  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      composeInput
    }
  }

  private var composeInput: some View {
    // The reference composer is a single rounded container: the text-field row
    // and the left/bottom-side control pills all live inside one card, instead
    // of two stacked boxes. Keep the placeholder text identical to the reference.
    VStack(alignment: .leading, spacing: 14) {
      TextField("Message Copilot", text: .constant(""))
        .font(PreviewTheme.composerInput)
        .foregroundColor(PreviewTheme.textPrimary)
        .textFieldStyle(.plain)
        .padding(.horizontal, 20)
        .padding(.top, 18)
        .padding(.bottom, 12)

      HStack(spacing: 10) {
        Button { /* attachment */ } label: {
          Image(systemName: "plus")
            .font(.title2)
            .foregroundColor(PreviewTheme.textSecondary)
            .frame(width: 44, height: 44)
            .background(PreviewTheme.panelAlt, in: Circle())
        }
        .buttonStyle(.plain)

        Button { /* smart mode */ } label: {
          HStack(spacing: 6) {
            Text("Smart")
              .font(PreviewTheme.body)
            Image(systemName: "chevron.down")
              .font(.system(size: 16, weight: .medium, design: .default))
          }
          .foregroundColor(PreviewTheme.textPrimary)
          .padding(.horizontal, 14)
          .padding(.vertical, 10)
          .background(PreviewTheme.panelAlt, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
        .buttonStyle(.plain)

        Button { /* tools */ } label: {
          HStack(spacing: 6) {
            Image(systemName: "globe")
              .font(.title3)
            Image(systemName: "chevron.down")
              .font(.system(size: 14, weight: .medium, design: .default))
          }
          .foregroundColor(PreviewTheme.textPrimary)
          .padding(.horizontal, 12)
          .padding(.vertical, 10)
          .background(PreviewTheme.panelAlt, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
        .buttonStyle(.plain)

        Spacer()

        // Right-side tools (share-node, mic) — same glyphs as the reference, right-aligned
        Button { /* share */ } label: {
          Image(systemName: "point.topleft.down.curvedto.point.bottomright.up")
            .font(.title3)
            .foregroundColor(PreviewTheme.textSecondary)
        }
        .buttonStyle(.plain)

        Button { /* audio */ } label: {
          Image(systemName: "waveform")
            .font(.title3)
            .foregroundColor(PreviewTheme.textSecondary)
        }
        .buttonStyle(.plain)
      }
      .padding(.horizontal, 20)
    }
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(PreviewTheme.panel)
    .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
  }
}
