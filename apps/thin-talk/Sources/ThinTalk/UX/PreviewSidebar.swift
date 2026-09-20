import SwiftUI
import ThinTalkUI

/// Sidebar that mirrors the reference Copilot layout exactly:
/// brand row → three nav items (Discover/Imagine/Library) → a
/// "Conversations" label heading → a long flat single-line list
/// (no dividers, no "New conversation" button) → a circular avatar
/// with the user's first name anchored to the bottom.
///
/// The previous pass added dividers, a large "New conversation" button,
/// two-line title+subtitle rows, and a "Local on this Mac" footer,
/// none of which appear in the reference, so those are removed.
struct PreviewSidebar: View {
  @StateObject private var viewModel: PreviewSidebarViewModel

  init(viewModel: PreviewSidebarViewModel) {
    _viewModel = StateObject(wrappedValue: viewModel)
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      brandRow
      Spacer().frame(height: 18)
      iconRail
      Spacer().frame(height: 20)
      conversationsLabel
      Spacer().frame(height: 10)
      conversationList
      Spacer()
      bottomRow
    }
    .padding(20)
    .frame(width: 300, alignment: .leading)
    .background(PreviewTheme.panel)
  }

  // MARK: - Brand row

  private var brandRow: some View {
    HStack(spacing: 10) {
      // A small multicolor rounded mark stands in for the Copilot logo glyph.
      RoundedRectangle(cornerRadius: 6, style: .continuous)
        .fill(
          LinearGradient(
            colors: [Color(red: 0.35, green: 0.45, blue: 0.95),
                     Color(red: 0.95, green: 0.45, blue: 0.75),
                     Color(red: 0.45, green: 0.85, blue: 0.95)],
            startPoint: .topLeading, endPoint: .bottomTrailing))
        .frame(width: 26, height: 26)
      Text("Copilot")
        .font(PreviewTheme.sidebarHeading)
        .foregroundColor(PreviewTheme.textPrimary)
    }
  }

  // MARK: - Icon rail (Discover / Imagine / Library only)

  private var iconRail: some View {
    VStack(spacing: 4) {
      ForEach(Array(PreviewIconRailItem.allCases.enumerated()), id: \.offset) { _, item in
        Button {
          viewModel.selectRail(item)
        } label: {
          HStack(spacing: 12) {
            Image(systemName: item.iconName)
              .font(.system(size: 18))
              .foregroundColor(PreviewTheme.textPrimary)
              .frame(width: 24, height: 24)
            Text(item.label)
              .font(PreviewTheme.sidebarLabel)
              .foregroundColor(PreviewTheme.textPrimary)
            Spacer()
          }
          .padding(.horizontal, 10)
          .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .frame(height: PreviewTheme.sidebarNavHeight)
      }
    }
  }

  // MARK: - Conversations label

  private var conversationsLabel: some View {
    Text("Conversations")
      .font(PreviewTheme.sidebarHeading)
      .foregroundColor(PreviewTheme.textSecondary)
      .kerning(0.5)
  }

  // MARK: - Flat single-line conversation list

  private var conversationList: some View {
    VStack(alignment: .leading, spacing: 2) {
      ForEach(viewModel.conversations) { conv in
        Button {
          viewModel.selectConversation(conv)
        } label: {
          Text(conv.title)
            .font(PreviewTheme.sidebarConversation)
            .foregroundColor(conv.isSelected ? PreviewTheme.textPrimary : PreviewTheme.textSecondary)
            .lineLimit(1)
            .padding(.vertical, 8)
            .padding(.horizontal, 10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
              conv.isSelected
                ? PreviewTheme.panelAlt
                : Color.clear
            )
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        .buttonStyle(.plain)
      }
    }
  }

  // MARK: - Bottom avatar row

  private var bottomRow: some View {
    HStack(spacing: 10) {
      ZStack {
        Circle()
          .fill(PreviewTheme.panelAlt)
          .frame(width: 34, height: 34)
        Text("A")
          .font(.system(size: 15, weight: .semibold))
          .foregroundColor(PreviewTheme.textPrimary)
      }
      Text("Ahmed")
        .font(PreviewTheme.sidebarLabel)
        .foregroundColor(PreviewTheme.textPrimary)
      Spacer()
    }
  }
}