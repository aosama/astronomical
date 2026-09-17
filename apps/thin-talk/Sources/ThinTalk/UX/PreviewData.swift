import Foundation
import SwiftUI
import ThinTalkCore

/// Chunked tokens of streamed content. Splitting preserves spaces and newlines as
/// their own tokens so char-by-char streaming keeps the original markdown
/// formatting instead of collapsing it into one flat paragraph.
func tokenizeStream(_ raw: String) -> [String] {
  var tokens: [String] = []
  var buffer = ""
  for character in raw {
    if character == " " || character == "\n" {
      if !buffer.isEmpty { tokens.append(buffer); buffer = "" }
      tokens.append(String(character))
    } else {
      buffer.append(character)
    }
  }
  if !buffer.isEmpty { tokens.append(buffer) }
  return tokens
}

/// A scripted assistant answer used by the preview so the agreed layout can be
/// reviewed without a supervisor. It carries a quiet reasoning lead-in, visible
/// markdown, and a rich HTML card rendered through WebKit.
struct ScriptedAnswer: Sendable {
  let reasoning: String
  let contentMarkdown: String
  let richHTML: String
  let chunked: [String]

  init(reasoning: String, contentMarkdown: String, richHTML: String) {
    self.reasoning = reasoning
    self.contentMarkdown = contentMarkdown
    self.richHTML = richHTML
    self.chunked = tokenizeStream(contentMarkdown)
  }
}

/// Static preview content: a small Library model list, mock local sessions, a
/// sample conversation, and one scripted rich answer. This is demo material only;
/// the production surface reads these sources from the backend.
enum PreviewData {
  /// Stable reference id for the pre-seeded rich conversation.
  static let sampleSessionID = UUID()

  static let models = [
    ThinTalkModel(id: "Astronomical-3B-chat", name: "Astronomical 3B Chat", inputModalities: ["text"]),
    ThinTalkModel(id: "Astronomical-7B-vision", name: "Astronomical 7B Vision", inputModalities: ["text", "image"]),
    ThinTalkModel(id: "Ornith-1.5-9B-vision-OptiQ-static-4.5bpw",
                  name: "Ornith 1.5 · 9B · vision", inputModalities: ["text", "image"]),
  ]

  static let scripted = ScriptedAnswer(
    reasoning: "I'll explain matrix multiplication simply, show the shape rule, give a compact example, then relate it to why it stays fast and private on-device.",
    contentMarkdown:
      "- MLX-LM uses MATMUL for attention, MLP layers, and KV-cache projections.\n"
      + "- OMLX layers additional optimizations (persistent KV cache, batching), but still relies on MLX's MATMUL kernels underneath.\n",
    richHTML: richHTMLContent
  )

  /// The "If you want" follow-up prompts shown as blue links.
  static let followUpLinks: [String] = [
    "MLX's linear-algebra kernel design",
    "How MATMUL affects LLM speed on Apple Silicon",
    "How MLX chooses CPU vs GPU for MATMUL",
  ]

  static let sampleSession: [PreviewMessage] = [
    PreviewMessage(role: .user, content: "Tell me how matrix multiplication works in a local model."),
    PreviewMessage(role: .assistant,
      content: scripted.contentMarkdown,
      reasoning: scripted.reasoning,
      richHTML: scripted.richHTML),
  ]

  static let sessions: [PreviewSession] = [
    PreviewSession(id: sampleSessionID,
      title: "How matrix multiplication works",
      lastText: "Why does it feel instant locally?",
      updatedAt: Date(),
      hasFailure: false),
    PreviewSession(title: "Packaging a model into a wheel",
      lastText: "The weights ship separately, so the package points on disk.",
      updatedAt: Date().addingTimeInterval(-3600)),
    PreviewSession(title: "Meaning of 'dare_ties' in Rust",
      lastText: "A trait bound that ties a lifetime to the caller.",
      updatedAt: Date().addingTimeInterval(-86_400)),
  ]

  /// Sidebar conversation items derived from mock sessions. The reference shows
  /// a long, flat, single-line list of titles (no subtitles, no separators), so
  /// the preview supplies a realistic-length list of distinct topics.
  static let conversations: [PreviewConversationItem] = [
    PreviewConversationItem(id: sampleSessionID,
      title: "MATMUL in Apple MLX",
      subtitle: nil,
      isSelected: true),
    PreviewConversationItem(title: "Packaging and Shipping Ru...", subtitle: nil, isSelected: false),
    PreviewConversationItem(title: "Rust Libraries for Machine...", subtitle: nil, isSelected: false),
    PreviewConversationItem(title: "Meaning of \"dare_ties\" in...", subtitle: nil, isSelected: false),
    PreviewConversationItem(title: "Checking Spelling in Profes...", subtitle: nil, isSelected: false),
    PreviewConversationItem(title: "O(n²) vs Quadratic Equations", subtitle: nil, isSelected: false),
    PreviewConversationItem(title: "OpenAI Responses and Co...", subtitle: nil, isSelected: false),
    PreviewConversationItem(title: "DeepSeek V4 Flash Active...", subtitle: nil, isSelected: false),
    PreviewConversationItem(title: "OpenAI Coding Plan Pricin...", subtitle: nil, isSelected: false),
    PreviewConversationItem(title: "Fast Fourier Transform Com...", subtitle: nil, isSelected: false),
    PreviewConversationItem(title: "LM Studio Open-Source St...", subtitle: nil, isSelected: false),
    PreviewConversationItem(title: "Calculating iogpu.wired_...", subtitle: nil, isSelected: false),
    PreviewConversationItem(title: "Codex Reset Monitor Webs...", subtitle: nil, isSelected: false),
    PreviewConversationItem(title: "Company Behind Ling-3.0-...", subtitle: nil, isSelected: false),
    PreviewConversationItem(title: "BF16 in MLX Model Names", subtitle: nil, isSelected: false),
  ]

  /// Development identity used by the UX preview.
  static let identity = ThinTalkApplicationIdentity(
    channel: .development,
    supervisorPort: 6733
  )

  static func messages(for sessionID: UUID?) -> [PreviewMessage] {
    guard sessionID == sampleSessionID else {
      return [
        PreviewMessage(role: .user, content: "How do I package a model into a wheel?"),
        PreviewMessage(role: .assistant, content:
          "Weights ship separately, so the package points to the model\n"
          + "on disk. The runner then loads them on demand."),
      ]
    }
    return sampleSession
  }
}

/// A self-contained HTML answer card rendered inside a bounded WebKit view. It
/// demonstrates the rich-content path (title, CSS image gallery, source chips)
/// without loading any remote resource, so it stays fast and fully local.
private let richHTMLContent: String = """
<html>
<head>
<style>
  body { font-family: -apple-system, "SF Pro Text", system-ui, sans-serif;
    margin: 0; background: transparent; color: #e8ebf0; }
  .card { background: #161a24; border: 1px solid #2a2f3a; border-radius: 16px;
    padding: 16px; max-width: 520px; }
  .title { font-size: 15px; font-weight: 600; margin-bottom: 10px; color: #ffffff; }
  .gallery { display: flex; gap: 8px; margin-bottom: 12px; }
  .thumb { width: 96px; height: 72px; border-radius: 12px;
    background: linear-gradient(135deg, #343b4d 0%, #2a2f3a 100%);
    display: flex; align-items: flex-end; padding: 6px; font-size: 11px; color: #9aa2b2; }
  .thumb.a { background: linear-gradient(135deg, #2f6bb0 0%, #23283a 100%); }
  .thumb.b { background: linear-gradient(135deg, #5b7bd6 0%, #23283a 100%); }
  .thumb.c { background: linear-gradient(135deg, #3a8a7d 0%, #23283a 100%); }
  .sources { display: flex; gap: 8px; flex-wrap: wrap; }
  .chip { display: inline-flex; align-items: center; gap: 6px; font-size: 12px;
    color: #cfd6e2; background: #202634; border: 1px solid #2f3646;
    border-radius: 999px; padding: 5px 10px; text-decoration: none; }
</style>
</head>
<body>
  <div class="card">
    <div class="title">Matrix multiplication</div>
    <div class="gallery">
      <div class="thumb a">1</div>
      <div class="thumb b">2</div>
      <div class="thumb c">3</div>
    </div>
    <div class="sources">
      <span class="chip">📄 Unified Memory — MLX docs</span>
      <span class="chip">🔗 Matrix Multiplication</span>
    </div>
  </div>
</body>
</html>
"""
