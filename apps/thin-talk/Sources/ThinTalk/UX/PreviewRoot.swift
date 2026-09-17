import SwiftUI

/// Root layout that owns the full Thin Talk UX: sidebar + main pane
/// (header → conversation → compose dock).
struct PreviewRootView: View {
  var body: some View {
    HStack(spacing: 0) {
      PreviewSidebar(viewModel: PreviewSidebarViewModel(conversations: PreviewData.conversations,
                                                        identity: PreviewData.identity))
      // A hairline seam instead of a Divider: the system divider renders a
      // thick grabbable-looking wedge that reads as a resize handle.
      Rectangle()
        .fill(PreviewTheme.border)
        .frame(width: 1)
      mainPane
    }
    .background(PreviewTheme.windowBackground)
  }

  // MARK: - Main pane

  private var mainPane: some View {
    VStack(spacing: 0) {
      headerRow
      Divider()
        .background(PreviewTheme.border)
        .padding(.horizontal, 24)
      PreviewConversation(viewModel: PreviewConversationViewModel(
        conversation: PreviewData.conversations.first!
      ))
      Divider()
        .background(PreviewTheme.border)
        .padding(.horizontal, 24)
      PreviewComposeDock()
    }
    .frame(minWidth: 1100, minHeight: 700)
  }

  // MARK: - Header

  private var headerRow: some View {
    HStack(spacing: 0) {
      // Top-left toolbar icons: sidebar-toggle and open-in-new, matching the
      // reference's utility bar next to the traffic lights.
      HStack(spacing: 12) {
        Image(systemName: "sidebar.left")
          .font(.body)
          .foregroundColor(PreviewTheme.textSecondary)
        Image(systemName: "square.and.pencil")
          .font(.body)
          .foregroundColor(PreviewTheme.textSecondary)
      }
      .padding(.leading, 24)

      Spacer()

      // Title centered in the remaining space
      Text("MATMUL in Apple MLX")
        .font(PreviewTheme.h2)
        .foregroundColor(PreviewTheme.textPrimary)
        .frame(maxWidth: .infinity)

      // Top-right icons: share/invite + clone/pane, matching the reference
      HStack(spacing: 14) {
        Image(systemName: "person.crop.circle.badge.plus")
          .font(.body)
          .foregroundColor(PreviewTheme.textSecondary)
        Image(systemName: "rectangle.on.rectangle")
          .font(.body)
          .foregroundColor(PreviewTheme.textSecondary)
      }
      .padding(.trailing, 24)
    }
    .padding(.vertical, 14)
  }
}
