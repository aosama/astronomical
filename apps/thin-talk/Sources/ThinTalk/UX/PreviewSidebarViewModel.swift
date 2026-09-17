import Foundation
import SwiftUI
import ThinTalkCore

/// Owns sidebar UX state: session management, conversation selection,
/// and icon-rail navigation.
@MainActor
final class PreviewSidebarViewModel: ObservableObject {
  @Published var conversations: [PreviewConversationItem]
  let identity: ThinTalkApplicationIdentity

  init(conversations: [PreviewConversationItem], identity: ThinTalkApplicationIdentity) {
    self.conversations = conversations
    self.identity = identity
  }

  var selectedConversation: PreviewConversationItem? {
    conversations.first { $0.isSelected }
  }

  func selectConversation(_ conv: PreviewConversationItem) {
    conversations = conversations.map {
      $0.id == conv.id ? PreviewConversationItem(id: $0.id, title: $0.title,
                                                subtitle: $0.subtitle, isSelected: true)
                        : PreviewConversationItem(id: $0.id, title: $0.title,
                                                subtitle: $0.subtitle, isSelected: false)
    }
  }

  func newConversation() {
    let item = PreviewConversationItem(id: UUID(), title: "New conversation",
                                       subtitle: "", isSelected: true)
    conversations.insert(item, at: 0)
  }

  func selectRail(_ item: PreviewIconRailItem) {
    _ = item
  }
}
