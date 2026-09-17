import Foundation
import SwiftUI
import ThinTalkCore

/// A sidebar conversation item: the compact row shown in the conversation
/// list with title, optional subtitle, and selection state.
struct PreviewConversationItem: Identifiable, Equatable {
  let id: UUID
  let title: String
  let subtitle: String?
  let isSelected: Bool

  init(id: UUID = UUID(), title: String, subtitle: String?, isSelected: Bool = false) {
    self.id = id
    self.title = title
    self.subtitle = subtitle
    self.isSelected = isSelected
  }
}

/// Icon-rail row items shown between brand and conversation list.
/// The reference shows exactly three items (Discover/Imagine/Library); the
/// previous "Tools" case is removed so the rail matches the agreed layout.
enum PreviewIconRailItem: String, CaseIterable, Identifiable {
  case discover = "stopwatch"
  case imagine = "video.fill"
  case library = "books.vertical.fill"

  var id: String { rawValue }
  var iconName: String { rawValue }
  var label: String {
    switch self {
    case .discover: return "Discover"
    case .imagine: return "Imagine"
    case .library: return "Library"
    }
  }
}

/// One rendered conversation message in the UX preview. It mirrors the core
/// `ChatMessage` shape but adds an optional WebKit-rendered answer card, so the
/// surface can demonstrate both native markdown and rich HTML rendering without
/// changing the production storage model.
struct PreviewMessage: Identifiable, Sendable, Equatable {
  let id: UUID
  let role: ChatRole
  var content: String
  var reasoning: String
  /// Optional HTML rendered inside an embedded WebKit view. An empty value means
  /// native markdown only.
  var richHTML: String?

  init(id: UUID = UUID(), role: ChatRole, content: String, reasoning: String = "", richHTML: String? = nil) {
    self.id = id
    self.role = role
    self.content = content
    self.reasoning = reasoning
    self.richHTML = richHTML
  }
}

/// A persisted local session shown in the sidebar. In production these live under
/// the active channel's state directory; here they are mock rows for the layout
/// and interactions being reviewed.
struct PreviewSession: Identifiable, Sendable, Equatable {
  let id: UUID
  var title: String
  var lastText: String
  var updatedAt: Date
  var hasFailure: Bool

  init(id: UUID = UUID(), title: String, lastText: String, updatedAt: Date = Date(), hasFailure: Bool = false) {
    self.id = id
    self.title = title
    self.lastText = lastText
    self.updatedAt = updatedAt
    self.hasFailure = hasFailure
  }
}
