import SwiftUI
import ThinTalkCore
import ThinTalkUI

/// Main conversation canvas. Mirrors the reference layout exactly:
/// intro bullets → divider → "Visual:" heading + carousel → divider
/// → "If you want" section with blue links → source cards → reasoning line.
/// No rounded bubble — content flows flush on the dark background.
struct PreviewConversation: View {
  @StateObject private var viewModel: PreviewConversationViewModel

  init(viewModel: PreviewConversationViewModel) {
    _viewModel = StateObject(wrappedValue: viewModel)
  }

  var body: some View {
    VStack(spacing: 0) {
      messageScrollView
      Spacer().frame(height: 16)
      Divider()
        .background(PreviewTheme.border)
        .padding(.horizontal, 4)
      Spacer().frame(height: 10)
      actionBar
      Spacer().frame(height: 12)
    }
    .padding(.horizontal, 28)
    .padding(.top, 24)
  }

  // MARK: - Scroll area

  private var messageScrollView: some View {
    ScrollView {
      LazyVStack(alignment: .leading, spacing: 32) {
        ForEach(viewModel.messages) { message in
          if message.role == .user {
            userBubble(message)
          } else {
            assistantBlock(message)
          }
        }
      }
      .padding(.vertical, 8)
      .frame(maxWidth: .infinity, alignment: .leading)
    }
    .defaultScrollAnchor(.top)
  }

  // MARK: - User bubble

  private func userBubble(_ message: PreviewMessage) -> some View {
    VStack(alignment: .trailing, spacing: 10) {
      Text("You")
        .font(.system(size: 14, weight: .medium))
        .foregroundColor(PreviewTheme.textSecondary)
      Text(message.content)
        .font(PreviewTheme.body)
        .foregroundColor(PreviewTheme.textPrimary)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
  }

  // MARK: - Assistant block (structured to match reference)

  @ViewBuilder
  private func assistantBlock(_ message: PreviewMessage) -> some View {
    VStack(alignment: .leading, spacing: 20) {
      // Intro content rendered from markdown
      MarkdownText(markdown: message.content)
        .foregroundColor(PreviewTheme.textPrimary)

      // Divider
      Divider().background(PreviewTheme.border)

      // "Visual:" heading
      Text("🖊 Visual: Matrix Multiplication Concept")
        .font(PreviewTheme.h1)
        .foregroundColor(PreviewTheme.textPrimary)

      // Image carousel
      VisualCarouselCard(imageCount: 4)
        .frame(maxWidth: .infinity, alignment: .leading)

      // Divider
      Divider().background(PreviewTheme.border)

      // "If you want" section
      Text("If you want, I can break down:")
        .font(PreviewTheme.h2)
        .foregroundColor(PreviewTheme.textPrimary)

      VStack(alignment: .leading, spacing: 10) {
        ForEach(PreviewData.followUpLinks, id: \.self) { link in
          HStack(alignment: .top, spacing: 8) {
            Text("•")
              .font(PreviewTheme.body)
              .foregroundColor(PreviewTheme.linkBlue)
              .padding(.top, 2)
            Text(link)
              .font(PreviewTheme.body)
              .foregroundColor(PreviewTheme.linkBlue)
          }
        }
      }

      // Source cards
      sourceCards

      // Reasoning (italic, lighter)
      if !message.reasoning.isEmpty {
        Text(message.reasoning)
          .font(PreviewTheme.reasoning)
          .foregroundColor(PreviewTheme.textSecondary)
          .padding(.top, 4)
      }
    }
    .frame(maxWidth: .infinity, alignment: .leading)
  }

  // MARK: - Source cards

  private var sourceCards: some View {
    HStack(spacing: 12) {
      SourceCard(domain: "github.io",
                 icon: "chevron.left.forwardslash.chevron.right",
                 title: "Unified Memory — MLX 0.32.1 docume...")
      SourceCard(domain: "Emergent Mind",
                 icon: "paintpalette.fill",
                 title: "MLX: Apple Silicon ML Framework")
      Button { } label: {
        VStack(alignment: .leading, spacing: 8) {
          Image(systemName: "globe")
            .font(.body)
            .foregroundColor(PreviewTheme.textSecondary)
          Text("Show all")
            .font(PreviewTheme.body)
            .foregroundColor(PreviewTheme.textPrimary)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(PreviewTheme.panelAlt)
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
      }
      .buttonStyle(.plain)
    }
  }

  // MARK: - Action bar

  private var actionBar: some View {
    HStack(spacing: 20) {
      actionIcon("hand.thumbsup")
      actionIcon("hand.thumbsdown")
      actionIcon("arrowshape.turn.up.right")
      actionIcon("square.on.square")
      actionIcon("arrow.clockwise")
      actionIcon("speaker.wave.2.fill")
      actionIcon("pencil")
      Spacer()
    }
  }

  private func actionIcon(_ systemName: String) -> some View {
    Button { } label: {
      Image(systemName: systemName)
        .font(.system(size: 16))
        .foregroundColor(PreviewTheme.textSecondary)
    }
    .buttonStyle(.plain)
  }
}

// MARK: - Markdown text

/// Parses the small markdown subset (bullets `-`, H1/H2 headings, `**bold**`)
/// into properly styled SwiftUI views so `#` and `-` never leak into the UI.
struct MarkdownText: View {
  let markdown: String

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      ForEach(Array(markdownLines.enumerated()), id: \.offset) { _, line in
        renderLine(line)
      }
    }
  }

  @ViewBuilder
  private func renderLine(_ line: String) -> some View {
    let trimmed = line.trimmingCharacters(in: .whitespaces)
    if trimmed.hasPrefix("- ") {
      HStack(alignment: .top, spacing: 8) {
        Text("•").font(PreviewTheme.body).padding(.top, 1)
        styledText(String(trimmed.dropFirst(2)))
          .font(PreviewTheme.body)
      }
    } else if trimmed.hasPrefix("## ") {
      styledText(String(trimmed.dropFirst(3)))
        .font(PreviewTheme.h2)
        .padding(.top, 4)
    } else if trimmed.hasPrefix("# ") {
      styledText(String(trimmed.dropFirst(2)))
        .font(PreviewTheme.h1)
        .padding(.top, 4)
    } else if trimmed.isEmpty {
      Spacer().frame(height: 4)
    } else {
      styledText(trimmed)
        .font(PreviewTheme.body)
    }
  }

  private var markdownLines: [String] {
    markdown.components(separatedBy: "\n")
  }
}

/// Builds a `Text` with inline `**bold**` emphasis, so bold formatting is
/// preserved through the per-line markdown rendering pass.
@MainActor func styledText(_ text: String) -> Text {
  var result = Text("")
  var remaining = Substring(text)
  while let open = remaining.range(of: "**") {
    let before = remaining[..<open.lowerBound]
    let afterOpen = remaining[open.upperBound...]
    if let close = afterOpen.range(of: "**") {
      result = result + Text(String(before))
      result = result + Text(String(afterOpen[..<close.lowerBound])).fontWeight(.semibold)
      remaining = afterOpen[close.upperBound...]
    } else {
      break
    }
  }
  return result + Text(String(remaining))
}

// MARK: - Visual carousel card

/// A rounded strip of image placeholders with a "N images" badge, matching
/// the reference's matrix-multiplication diagram carousel.
struct VisualCarouselCard: View {
  let imageCount: Int

  var body: some View {
    let count = min(imageCount, 4)
    HStack(spacing: 4) {
      ForEach(0..<count, id: \.self) { index in
        CarouselThumb(index: index)
      }
    }
    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    .overlay(alignment: .bottomTrailing) {
      Text("\(imageCount) images")
        .font(.system(size: 12, weight: .semibold))
        .foregroundColor(.white)
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(RoundedRectangle(cornerRadius: 6).fill(.black.opacity(0.7)))
        .offset(x: -4, y: -4)
    }
    .frame(maxWidth: 560)
  }
}

private struct CarouselThumb: View {
  let index: Int

  // Light placeholders — the reference shows white cards with matrix math.
  private static let palette: [Color] = [
    .init(red: 0.92, green: 0.92, blue: 0.93),
    .init(red: 0.90, green: 0.91, blue: 0.94),
    .init(red: 0.88, green: 0.90, blue: 0.93),
    .init(red: 0.91, green: 0.91, blue: 0.93),
  ]

  var body: some View {
    RoundedRectangle(cornerRadius: 4)
      .fill(Self.palette[index % Self.palette.count])
      .frame(height: 88)
      .frame(maxWidth: .infinity)
  }
}

// MARK: - Source card

struct SourceCard: View {
  let domain: String
  var icon: String = "globe"
  let title: String

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack(spacing: 6) {
        Image(systemName: icon)
          .font(.system(size: 13))
          .foregroundColor(icon == "paintpalette.fill" ? .yellow : PreviewTheme.textSecondary)
        Text(domain)
          .font(.system(size: 13, weight: .medium))
          .foregroundColor(PreviewTheme.textSecondary)
      }
      Text(title)
        .font(PreviewTheme.body)
        .foregroundColor(PreviewTheme.textPrimary)
        .lineLimit(1)
        .truncationMode(.tail)
    }
    .padding(14)
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(PreviewTheme.panelAlt)
    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
  }
}

// MARK: - Failure banner

struct FailureBanner: View {
  let failure: ChatFailure

  var body: some View {
    HStack {
      Image(systemName: "exclamationmark.triangle.fill")
        .foregroundColor(PreviewTheme.danger)
      Text(failure.message ?? failure.kind.rawValue)
        .font(PreviewTheme.body)
        .foregroundColor(PreviewTheme.danger)
    }
    .padding(14)
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(PreviewTheme.danger.opacity(0.1))
    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
  }
}