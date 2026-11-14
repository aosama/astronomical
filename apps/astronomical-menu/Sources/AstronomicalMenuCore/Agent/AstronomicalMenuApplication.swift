import AppKit
import SwiftUI

/// The menu-bar agent: the status item, its popover, and ownership of the
/// engine. It is an accessory application for its whole life (its bundle sets
/// LSUIElement) and never shows a window of its own. Chat belongs to the main
/// app, which the agent asks for by opening one of its links.
@MainActor
public final class AstronomicalMenuApplication: NSObject, NSApplicationDelegate, NSPopoverDelegate {
  // The adaptor instantiates the delegate from the executable module, so the
  // default initializer must be publicly reachable.
  public override init() {}

  private let applicationIdentity = ApplicationIdentity.current()
  private lazy var daemonClient = LocalDaemonClient(applicationIdentity: applicationIdentity)
  private lazy var telemetryStore = TelemetryStore(daemonClient: daemonClient)
  private lazy var daemonLifecycleController = DaemonLifecycleController(
    daemonClient: daemonClient, applicationIdentity: applicationIdentity)
  private let neuralEngineBannerPresenter = NeuralEngineBannerPresenter()
  private let notificationRouter = AgentNotificationRouter()
  private lazy var updateNoticeCoordinator = UpdateNoticeCoordinator(
    store: UpdateNoticeStore(
      installedBuildNumber: Int(applicationIdentity.buildNumber) ?? 0,
      settingsReader: BundledMainApplicationUpdateSettingsReader(
        applicationChannel: applicationIdentity.channel, agentBundleURL: Bundle.main.bundleURL)),
    presenter: UpdateNoticePresenter(),
    announcementRecord: UpdateAnnouncementRecord(userDefaults: .standard),
    mainApplicationPresence: RunningMainApplicationPresence(
      mainApplicationBundleIdentifier: applicationIdentity.channel.applicationBundleIdentifier),
    activityIdentifier: "\(applicationIdentity.channel.agentBundleIdentifier).update-notice")
  private lazy var cliLinkInstaller = CliLinkInstaller(channel: applicationIdentity.channel)
  private lazy var commandChannel = ProcessCommandChannel(channel: applicationIdentity.channel)
  private var commandObservations: [NSObjectProtocol] = []
  private var statusItem: NSStatusItem?
  private var telemetryPopover: NSPopover?
  private var latestMenuBarTitle = ""
  private var latestMenuBarGlyphState = MenuBarGlyphState.idle
  private var daemonMaintenanceTask: Task<Void, Never>?

  public func applicationDidFinishLaunching(_ notification: Notification) {
    // A packaged agent is an accessory through its bundle's LSUIElement. A
    // bare executable run has no bundle to say so, and would otherwise take a
    // Dock icon; the policy is set once here and never changed again.
    if Bundle.main.bundleIdentifier == nil { NSApp.setActivationPolicy(.accessory) }
    answerMainApplicationCommands()
    let menuBarStatusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
    menuBarStatusItem.button?.target = self
    menuBarStatusItem.button?.action = #selector(toggleTelemetryPopover)
    menuBarStatusItem.button?.image = makeMenuBarGlyphImage(state: latestMenuBarGlyphState)
    menuBarStatusItem.button?.setAccessibilityLabel(
      "Astronomical \(applicationIdentity.channel.displayName) telemetry")
    menuBarStatusItem.button?.toolTip = applicationIdentity.buildTitle
    statusItem = menuBarStatusItem

    let popover = NSPopover()
    popover.behavior = .transient
    popover.delegate = self
    popover.contentViewController = NSHostingController(
      rootView: OrbitalTelemetryPopover(
        telemetryStore: telemetryStore,
        updateNoticeStore: updateNoticeCoordinator.store,
        applicationIdentity: applicationIdentity,
        openObservatory: { [weak self] in self?.openObservatory() },
        openDownloadModels: { [weak self] in self?.openDownloadModels() },
        openChat: { [weak self] in self?.showChatWindow() },
        reloadConfiguration: { [weak self] in self?.telemetryStore.reloadConfiguration() },
        restartServer: { [weak self] in self?.restartServer() },
        updateCheckSupported: updateCheckSupported(applicationChannel: applicationIdentity.channel),
        checkForUpdates: { [weak self] in self?.openInMainApplication(.checkForUpdates) },
        terminalCommandSupported: cliLinkInstaller.isSupportedForChannel,
        installTerminalCommand: { [weak self] in self?.installTerminalCommand() },
        removeTerminalCommand: { [weak self] in self?.removeTerminalCommand() },
        revealConfiguration: revealConfiguration,
        showSettings: { [weak self] in self?.showSettings() },
        quitApplication: { [weak self] in self?.quitEverything() }
      )
    )
    telemetryPopover = popover
    telemetryStore.onNeuralEngineBannerDue = { [weak self] neuralEngineBanner in
      self?.neuralEngineBannerPresenter.present(neuralEngineBanner)
    }
    telemetryStore.onMenuBarTitleChanged = { [weak self] menuBarTitle in
      guard let self else { return }
      let channelAwareMenuBarTitle =
        applicationIdentity.channel == .development
        ? " DEV\(menuBarTitle.isEmpty ? "" : " · \(menuBarTitle)")" : menuBarTitle
      latestMenuBarTitle = channelAwareMenuBarTitle
      let currentTitle: String
      if let statusButton = statusItem?.button {
        currentTitle = statusButton.title
      } else {
        currentTitle = ""
      }
      statusItem?.button?.title = menuBarTitleToDisplay(
        currentTitle: currentTitle,
        latestTitle: channelAwareMenuBarTitle,
        popoverIsShown: telemetryPopover?.isShown == true
      )
    }
    telemetryStore.onMenuBarGlyphStateChanged = { [weak self] menuBarGlyphState in
      self?.showMenuBarGlyph(menuBarGlyphState)
    }
    daemonMaintenanceTask = Task { [weak self] in
      guard let self else { return }
      await maintainDaemonForApplication(
        daemonLifecycleController: daemonLifecycleController,
        telemetryStore: telemetryStore)
    }
    telemetryStore.startPolling()
    startUpdateNotices()
    // Full auto-repair (owner decision on issue #825): a terminal-command
    // link this app owns dangles whenever the bundle moves; recreating it on
    // every launch keeps the command current without user action.
    Task { [weak self] in await self?.cliLinkInstaller.repairOnLaunchIfNeeded() }
  }

  public func applicationWillTerminate(_ notification: Notification) {
    for commandObservation in commandObservations {
      DistributedNotificationCenter.default().removeObserver(commandObservation)
    }
    commandObservations = []
    daemonMaintenanceTask?.cancel()
    daemonMaintenanceTask = nil
    updateNoticeCoordinator.stop()
    telemetryStore.stopPolling()
    daemonLifecycleController.stopChannelDaemonOnApplicationQuit()
  }

  /// Starts the daily look at the update feed on builds that update
  /// themselves. Clicking the notification leads to the same place as the
  /// popover's row: the main app's updater.
  private func startUpdateNotices() {
    guard updateCheckSupported(applicationChannel: applicationIdentity.channel) else { return }
    notificationRouter.onUpdateNoticeOpened = { [weak self] in
      self?.openInMainApplication(.checkForUpdates)
    }
    updateNoticeCoordinator.start()
  }

  // Status polls arrive every few seconds; redrawing only on a state change
  // keeps the status item from re-laying out while the title text updates.
  private func showMenuBarGlyph(_ menuBarGlyphState: MenuBarGlyphState) {
    guard menuBarGlyphState != latestMenuBarGlyphState else { return }
    latestMenuBarGlyphState = menuBarGlyphState
    statusItem?.button?.image = makeMenuBarGlyphImage(state: menuBarGlyphState)
  }

  @objc private func toggleTelemetryPopover() {
    guard let telemetryPopover else { return }
    telemetryPopover.isShown ? telemetryPopover.performClose(nil) : showTelemetryPopover()
  }

  private func showTelemetryPopover() {
    guard let statusButton = statusItem?.button, let telemetryPopover else { return }
    // Without this the very first click dismisses the transient popover before
    // it is visible: macOS treats the click gesture as outside the popover the
    // moment this app is not the active app, which it is when you click the
    // menu-bar icon from another foreground app. Activating first keeps the
    // tap-to-dismiss behaviour intact while making the first click reliably
    // stick (issue: menu vanishes on first open).
    NSApp.activate(ignoringOtherApps: true)
    statusItem?.length = menuBarStatusItemLength(
      popoverIsShown: true, currentButtonWidth: statusButton.bounds.width)
    telemetryPopover.show(
      relativeTo: popoverAnchorRect(
        statusButtonBounds: statusButton.bounds,
        statusItemImageRect: statusButton.cell?.imageRect(forBounds: statusButton.bounds)
      ), of: statusButton,
      preferredEdge: .minY)
    telemetryStore.setPopoverVisible(true)
    // A version skipped, or automatic checks turned off, in the main app
    // since the last look should not still be offered here.
    updateNoticeCoordinator.store.applyCurrentSettings()
  }

  public func popoverDidClose(_ notification: Notification) {
    telemetryStore.setPopoverVisible(false)
    let currentButtonWidth: CGFloat
    if let statusButton = statusItem?.button {
      currentButtonWidth = statusButton.bounds.width
    } else {
      currentButtonWidth = NSStatusItem.squareLength
    }
    statusItem?.length = menuBarStatusItemLength(
      popoverIsShown: false,
      currentButtonWidth: currentButtonWidth
    )
    statusItem?.button?.title = latestMenuBarTitle
  }

  private func restartServer() {
    telemetryStore.beginServerRestart()
    Task { [weak self] in
      guard let self else { return }
      do {
        let restartMessage = try await daemonLifecycleController.restartDaemon()
        telemetryStore.completeServerRestart(restartMessage: restartMessage)
      } catch {
        telemetryStore.failServerRestart(error)
      }
      telemetryStore.refreshNow()
    }
  }

  /// The main app asks through these when it needs the engine restarted or
  /// stopped. A build check from a main app of another build means this agent
  /// was left running across an update, with the engine of the old build: it
  /// quits, and the main app starts the agent of its own build.
  private func answerMainApplicationCommands() {
    commandObservations = [
      commandChannel.observe(.restartEngine) { [weak self] _ in self?.restartServer() },
      commandChannel.observe(.quitAgent) { _ in NSApp.terminate(nil) },
      commandChannel.observe(.checkAgentBuild) { [weak self] mainApplicationBuildFingerprint in
        guard let self, let mainApplicationBuildFingerprint,
          mainApplicationBuildFingerprint != applicationIdentity.buildFingerprint
        else { return }
        NSApp.terminate(nil)
      },
    ]
  }

  /// The popover's Quit stops everything a person thinks of as Astronomical:
  /// the chat app, this agent, and the engine the agent owns.
  private func quitEverything() {
    commandChannel.post(.quitApplication)
    NSApp.terminate(nil)
  }

  private func showChatWindow() { openInMainApplication(.chat) }

  // The popover's "Settings…" opens the chat window on its Settings page,
  // so the chat settings are always one click from the menu bar.
  private func showSettings() { openInMainApplication(.settings) }

  /// Opens a link in the main app that contains this agent. Naming the bundle
  /// keeps the link from reaching another installed copy that registers the
  /// same scheme; a bare executable run has no enclosing bundle and lets
  /// macOS choose.
  private func openInMainApplication(_ applicationLink: ApplicationLink) {
    guard let linkURL = applicationLink.url(channel: applicationIdentity.channel) else { return }
    telemetryPopover?.performClose(nil)
    guard
      let mainApplicationBundleURL = ApplicationBundleLayout.mainApplicationBundleURL(
        forAgentBundleAt: Bundle.main.bundleURL)
    else {
      NSWorkspace.shared.open(linkURL)
      return
    }
    NSWorkspace.shared.open(
      [linkURL], withApplicationAt: mainApplicationBundleURL,
      configuration: NSWorkspace.OpenConfiguration())
  }

  private func openObservatory() {
    ObservatoryLauncher.open(path: "/", applicationIdentity: applicationIdentity)
  }

  private func openDownloadModels() {
    ObservatoryLauncher.open(path: "/download-models", applicationIdentity: applicationIdentity)
  }

  private func revealConfiguration() {
    let configurationURL = applicationIdentity.configFileURL()
    NSWorkspace.shared.activateFileViewerSelecting([configurationURL])
  }

  private func installTerminalCommand() {
    Task { [weak self] in
      guard let self else { return }
      do {
        try await cliLinkInstaller.installTerminalCommand()
      } catch {
        NSApp.presentError(error)
      }
    }
  }

  private func removeTerminalCommand() {
    Task { [weak self] in
      guard let self else { return }
      do {
        try await cliLinkInstaller.removeTerminalCommand()
      } catch {
        NSApp.presentError(error)
      }
    }
  }

}

@MainActor
func startDaemonForApplication(
  daemonLifecycleController: DaemonLifecycleController,
  telemetryStore: TelemetryStore
) async {
  do {
    try await daemonLifecycleController.startDaemonIfNeeded()
    telemetryStore.completeServerStartup()
  } catch {
    telemetryStore.failServerStartup(error)
  }
}

@MainActor
func maintainDaemonForApplication(
  daemonLifecycleController: DaemonLifecycleController,
  telemetryStore: TelemetryStore,
  retryDelay: Duration = .seconds(5)
) async {
  // Keep trying after a bad config or missing folder so the menu does not sit on a dead port.
  while !Task.isCancelled {
    do {
      try await daemonLifecycleController.startDaemonIfNeeded()
      telemetryStore.completeServerStartup()
      return
    } catch {
      telemetryStore.failServerStartup(error)
      try? await Task.sleep(for: retryDelay)
    }
  }
}

func menuBarStatusItemLength(popoverIsShown: Bool, currentButtonWidth: CGFloat) -> CGFloat {
  popoverIsShown ? currentButtonWidth : NSStatusItem.variableLength
}

func menuBarTitleToDisplay(
  currentTitle: String,
  latestTitle: String,
  popoverIsShown: Bool
) -> String {
  popoverIsShown ? currentTitle : latestTitle
}

func popoverAnchorRect(
  statusButtonBounds: CGRect,
  statusItemImageRect: CGRect?
) -> CGRect {
  if let statusItemImageRect {
    return statusItemImageRect
  }
  return statusButtonBounds
}
