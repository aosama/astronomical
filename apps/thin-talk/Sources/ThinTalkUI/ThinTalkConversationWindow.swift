import AppKit
import SwiftUI
import ThinTalkCanvas
import ThinTalkCore

/// The one conversation window both hosts open: the menu application's chat
/// window and the standalone development app. Building it in one place keeps
/// the shipped window and the one developers look at from drifting apart, which
/// is how the standalone app came to have a hidden title bar the shipped one
/// never had.
///
/// The frame follows Apple's own chat apps: the content runs under a
/// transparent title bar, the title text is hidden (the window keeps its title
/// for the Window menu, Mission Control and VoiceOver), and a unified toolbar
/// holds the sidebar toggle and New Chat. Window tabs are disabled because a
/// conversation window has nothing to tab.
@MainActor
public final class ThinTalkConversationWindow {
  public static let defaultContentSize = NSSize(width: 1080, height: 720)
  /// Below this the sidebar and the composer pills start overlapping.
  public static let minimumContentSize = NSSize(width: 640, height: 420)

  public let window: NSWindow
  public let conversationActions: ThinTalkConversationActions
  // NSToolbar holds its delegate weakly, so the window object owns it.
  private let toolbarController: ThinTalkToolbarController
  private let commandController: CanvasCommandController
  private var fullScreenObservers: [NSObjectProtocol] = []

  /// `restartEngine` is the host's way to restart the local engine, offered
  /// by Astro's welcome when the engine never answers; nil where the host
  /// has none (the standalone development app).
  public init(applicationIdentity: ThinTalkApplicationIdentity, restartEngine: (@MainActor () -> Void)? = nil) {
    let zoomController = CanvasZoomController()
    let commandController = CanvasCommandController()
    self.commandController = commandController
    conversationActions = ThinTalkConversationActions(
      commandController: commandController, zoomController: zoomController)
    toolbarController = ThinTalkToolbarController(conversationActions: conversationActions)
    window = Self.makeFramedWindow(
      title: applicationIdentity.windowTitle, toolbar: toolbarController.makeToolbar())
    conversationActions.hostWindow = window
    window.contentView = NSHostingView(
      rootView: RootView(
        applicationIdentity: applicationIdentity,
        zoomController: zoomController,
        commandController: commandController,
        titlebarInset: Self.titlebarInset(of: window),
        restartEngine: restartEngine))
    window.center()
    observeTitlebarHeightChanges()
  }

  /// Full screen changes how far the title bar and toolbar reach into the
  /// content, so the page is told the new height on the way in and out.
  private func observeTitlebarHeightChanges() {
    for titlebarChange in [NSWindow.didEnterFullScreenNotification, NSWindow.didExitFullScreenNotification] {
      let observer = NotificationCenter.default.addObserver(
        forName: titlebarChange, object: window, queue: .main
      ) { [weak self] _ in
        MainActor.assumeIsolated { self?.remeasureTitlebarInset() }
      }
      fullScreenObservers.append(observer)
    }
  }

  private func remeasureTitlebarInset() {
    commandController.applyTitlebarInset(points: Self.titlebarInset(of: window))
  }

  static func makeFramedWindow(title: String, toolbar: NSToolbar) -> NSWindow {
    let window = NSWindow(
      contentRect: NSRect(origin: .zero, size: defaultContentSize),
      styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
      backing: .buffered,
      defer: false)
    window.title = title
    window.titleVisibility = .hidden
    window.titlebarAppearsTransparent = true
    window.tabbingMode = .disallowed
    window.isReleasedWhenClosed = false
    window.contentMinSize = minimumContentSize
    window.toolbar = toolbar
    window.toolbarStyle = .unified
    return window
  }

  /// How far the title bar and toolbar reach into the content, in points. The
  /// page starts below this strip because the strip, not the page, receives the
  /// clicks there; that is what lets the window be dragged by its top edge.
  static func titlebarInset(of window: NSWindow) -> Int {
    let titlebarHeight = window.frame.height - window.contentLayoutRect.height
    return max(0, Int(titlebarHeight.rounded(.up)))
  }
}
