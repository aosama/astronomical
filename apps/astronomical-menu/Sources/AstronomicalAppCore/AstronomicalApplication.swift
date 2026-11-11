import AppKit
import ThinTalkUI

/// The main app: the Thin Talk conversation window and the menu bar of a
/// regular Mac application.
///
/// It is a regular application from launch to quit. Its window and menu bar
/// are built once, at launch, and never swapped, which is what keeps the
/// menus reachable: a process that changes its activation policy while
/// running does not reliably get the menu bar back.
///
/// The engine is not this app's: the menu-bar agent bundled inside it owns
/// the engine, so quitting this app leaves the engine serving other clients.
/// This app starts the agent when it is not already running and asks it,
/// through `ProcessCommandChannel`, for anything that touches the engine.
@MainActor
public final class AstronomicalApplication: NSObject, NSApplicationDelegate {
  /// How long after launch the app keeps making sure its own agent runs. An
  /// agent left over from another build quits when it hears the build check,
  /// and its quit first stops the engine, so the replacement cannot start in
  /// the same instant.
  private static let agentReplacementCheckDelays: [Duration] = [.seconds(1), .seconds(3), .seconds(6), .seconds(12)]

  public override init() {}

  private let applicationIdentity = ApplicationIdentity.current()
  private lazy var commandChannel = ProcessCommandChannel(channel: applicationIdentity.channel)
  private lazy var agentLauncher = AgentLauncher(applicationChannel: applicationIdentity.channel)
  private lazy var applicationUpdateController: any ApplicationUpdateChecking =
    ApplicationUpdateControllerInstaller.makeController(applicationChannel: applicationIdentity.channel)
  private lazy var loginItemRegistrar = SystemLoginItemRegistrar(
    agentBundleIdentifier: applicationIdentity.channel.agentBundleIdentifier)
  private lazy var settingsMenu = ApplicationSettingsMenu(
    applicationUpdateController: applicationUpdateController,
    loginItemRegistrar: loginItemRegistrar,
    launchAtLoginSupported: launchAtLoginSupported)
  private lazy var chatWindowController = ChatWindowController(
    thinTalkIdentity: ChatLaunchIdentity.chatIdentity(from: applicationIdentity),
    restartEngine: { [weak self] in self?.restartEngine() })
  private var quitCommandObservation: NSObjectProtocol?
  private var agentReplacementTask: Task<Void, Never>?
  /// Set once the agent is meant to stay stopped: this app told it to quit
  /// for an update or for "Quit and Stop Engine", or the agent's own Quit
  /// told this app to follow. Nothing may start the agent again after that;
  /// during an update it would be the build being replaced.
  private var isAgentStopIntended = false
  private var hasFinishedLaunching = false
  private lazy var updateCheckFromLink = LaunchDeferredAction { [weak self] in
    guard let self else { return }
    NSApp.activate(ignoringOtherApps: true)
    requestManualApplicationUpdateCheck(using: applicationUpdateController)
  }

  private var launchAtLoginSupported: Bool {
    launchAtLoginControlsSupported(
      applicationChannel: applicationIdentity.channel,
      hostIsPackagedApplication: Bundle.main.bundleIdentifier != nil)
  }

  // A link that launched the app arrives before launching finishes, so the
  // handler is in place first.
  public func applicationWillFinishLaunching(_ notification: Notification) {
    NSAppleEventManager.shared().setEventHandler(
      self,
      andSelector: #selector(handleGetURLEvent(_:withReplyEvent:)),
      forEventClass: AEEventClass(kInternetEventClass),
      andEventID: AEEventID(kAEGetURL))
  }

  public func applicationDidFinishLaunching(_ notification: Notification) {
    // A packaged app is regular through its bundle. A bare executable run has
    // no bundle to say so; the policy is set once here and never changed.
    if Bundle.main.bundleIdentifier == nil { NSApp.setActivationPolicy(.regular) }
    ThinTalkMainMenu.install(
      ThinTalkMainMenu.make(
        hostActions: hostMenuActions(), conversationActions: chatWindowController.conversationActions),
      in: NSApp)
    applicationUpdateController.prepareForRelaunch = { [weak self] in self?.stopAgentBeforeUpdate() }
    applicationUpdateController.start()
    updateCheckFromLink.becomeReady()
    quitCommandObservation = commandChannel.observe(.quitApplication) { [weak self] _ in
      self?.isAgentStopIntended = true
      NSApp.terminate(nil)
    }
    // Before anything can start the agent: its engine must find the state an
    // earlier store build left in this app's own container, which the agent
    // cannot open. Registering the agent at login is such a start, because
    // macOS launches a login item the moment it is registered.
    _ = moveContainerStateIntoGroupContainerIfNeeded(applicationIdentity: applicationIdentity)
    if launchAtLoginSupported { moveWholeApplicationLoginItemToAgent(using: loginItemRegistrar) }
    startAgentOfThisBuild()
    chatWindowController.showChatWindowAtLaunch(ChatLaunchPlacement.fromLaunchArguments())
    hasFinishedLaunching = true
  }

  public func applicationWillTerminate(_ notification: Notification) {
    agentReplacementTask?.cancel()
    // An update that installs at quit replaces the bundle with no relaunch to
    // follow, so this is the last moment to stop the agent and its engine.
    if applicationUpdateController.installsUpdateOnQuit { stopAgentBeforeUpdate() }
    if let quitCommandObservation {
      DistributedNotificationCenter.default().removeObserver(quitCommandObservation)
    }
  }

  /// Closing the window leaves the app in the Dock, as Messages does: a reply
  /// may still be arriving, and the window comes back with one click.
  public func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
    false
  }

  /// The agent owns the engine, and nothing else restarts an agent that
  /// crashed or was quit from Activity Monitor. Coming back to the chat app
  /// is when its absence matters, so that is when it is started again; the
  /// chat page then finds the engine by itself.
  public func applicationDidBecomeActive(_ notification: Notification) {
    guard hasFinishedLaunching else { return }
    ensureAgentOfThisBuild()
  }

  public func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows: Bool) -> Bool {
    chatWindowController.showChatWindow()
    return false
  }

  @objc private func handleGetURLEvent(_ event: NSAppleEventDescriptor, withReplyEvent: NSAppleEventDescriptor) {
    guard let openedURLString = event.paramDescriptor(forKeyword: AEKeyword(keyDirectObject))?.stringValue,
      let openedURL = URL(string: openedURLString)
    else { return }
    // A draft link fills the message field and nothing else: the window is
    // neither shown nor brought forward, which is the point of the link.
    if let draftText = DraftLink.draftText(from: openedURL, channel: applicationIdentity.channel) {
      chatWindowController.conversationActions.fillDraft(draftText)
      return
    }
    if let exportFolder = ExportFolderLink.folderURL(from: openedURL, channel: applicationIdentity.channel) {
      chatWindowController.conversationActions.saveNextExportWithoutAsking(into: exportFolder)
      return
    }
    guard let applicationLink = ApplicationLink.parse(openedURL, channel: applicationIdentity.channel) else { return }
    switch applicationLink {
    case .chat: chatWindowController.showChatWindow()
    case .yourAstro: chatWindowController.showYourAstro()
    case .checkForUpdates: updateCheckFromLink.request()
    }
  }

  private func hostMenuActions() -> ThinTalkHostMenuActions {
    let checkForUpdatesAction: (@MainActor () -> Void)?
    if applicationUpdateController.supportsUserUpdateControls {
      checkForUpdatesAction = { [weak self] in
        guard let self else { return }
        requestManualApplicationUpdateCheck(using: applicationUpdateController)
      }
    } else {
      checkForUpdatesAction = nil
    }
    // The bundle's own name, so the Development build's menus say
    // "Quit Astronomical Development" like its Dock icon does.
    let bundleApplicationName =
      Bundle.main.object(forInfoDictionaryKey: "CFBundleName") as? String ?? "Astronomical"
    return ThinTalkHostMenuActions(
      applicationName: bundleApplicationName,
      checkForUpdates: checkForUpdatesAction,
      openObservatory: { [weak self] in self?.openObservatory() },
      helpURL: URL(string: "https://aosama.github.io/astronomical/"),
      hostSettingsMenuItems: settingsMenu.makeMenuItems(),
      quitEverything: { [weak self] in self?.quitAndStopEngine() })
  }

  /// Starts the agent, and replaces one left running by another build: the
  /// build check makes such an agent quit, and the delayed passes start this
  /// build's agent once it has gone. Each pass asks again, because an agent
  /// that was still launching at the first ask had no observer to hear it and
  /// would otherwise keep serving the old engine. A pass while the right
  /// agent runs does nothing.
  private func startAgentOfThisBuild() {
    ensureAgentOfThisBuild()
    agentReplacementTask = Task { [weak self] in
      for replacementCheckDelay in Self.agentReplacementCheckDelays {
        try? await Task.sleep(for: replacementCheckDelay)
        guard !Task.isCancelled, let self else { return }
        ensureAgentOfThisBuild()
      }
    }
  }

  private func ensureAgentOfThisBuild() {
    guard !isAgentStopIntended else { return }
    agentLauncher.launchAgentIfNeeded()
    commandChannel.post(.checkAgentBuild, detail: applicationIdentity.buildFingerprint)
  }

  /// Astro's welcome asks for this when the engine never answers. An agent
  /// that is not running starts the engine as it launches, so starting it is
  /// the whole restart in that case.
  private func restartEngine() {
    guard agentLauncher.isAgentRunning else {
      agentLauncher.launchAgentIfNeeded()
      return
    }
    commandChannel.post(.restartEngine)
  }

  /// An update replaces the whole bundle. The agent and the engine it owns
  /// must stop first, or they would carry on running the old build; the
  /// relaunched app then starts the new agent.
  private func stopAgentBeforeUpdate() {
    isAgentStopIntended = true
    commandChannel.post(.quitAgent)
    _ = agentLauncher.waitForAgentExit()
  }

  private func quitAndStopEngine() {
    isAgentStopIntended = true
    commandChannel.post(.quitAgent)
    NSApp.terminate(nil)
  }

  private func openObservatory() {
    ObservatoryLauncher.open(path: "/", applicationIdentity: applicationIdentity)
  }
}

/// Runs the main app. Both main-app executables call this after choosing their
/// update controller, so the only difference between them stays that one line.
@MainActor
public func runAstronomicalApplication() {
  let application = NSApplication.shared
  let applicationDelegate = AstronomicalApplication()
  application.delegate = applicationDelegate
  // NSApplication holds its delegate weakly, and Swift may release a local
  // after its last use, so the delegate is kept alive for the whole run.
  withExtendedLifetime(applicationDelegate) {
    application.run()
  }
}
