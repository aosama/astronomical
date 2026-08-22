import AppKit
import SwiftUI

@MainActor
final class AstronomicalMenuApplication: NSObject, NSApplicationDelegate, NSPopoverDelegate {
  private let applicationIdentity = ApplicationIdentity.current()
  private lazy var supervisorClient = LocalSupervisorClient(applicationIdentity: applicationIdentity)
  private lazy var telemetryStore = TelemetryStore(supervisorClient: supervisorClient)
  private lazy var applicationUpdateController = ApplicationUpdateController(
    applicationChannel: applicationIdentity.channel)
  private lazy var daemonLifecycleController = DaemonLifecycleController(
    supervisorClient: supervisorClient, applicationIdentity: applicationIdentity)
  private var statusItem: NSStatusItem?
  private var telemetryPopover: NSPopover?
  private var latestMenuBarTitle = ""

  nonisolated func applicationWillFinishLaunching(_ notification: Notification) {
    DispatchQueue.main.async { NSApp.setActivationPolicy(.regular) }
  }

  func applicationDidFinishLaunching(_ notification: Notification) {
    let menuBarStatusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
    menuBarStatusItem.button?.target = self
    menuBarStatusItem.button?.action = #selector(toggleTelemetryPopover)
    menuBarStatusItem.button?.image = NSImage(
      systemSymbolName: "sparkles", accessibilityDescription: "Astronomical")
    menuBarStatusItem.button?.image?.size = NSSize(width: 18, height: 18)
    menuBarStatusItem.button?.image?.isTemplate = true
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
        applicationIdentity: applicationIdentity,
        openObservatory: { [weak self] in self?.openObservatory() },
        openLibrary: { [weak self] in self?.openLibrary() },
        reloadConfiguration: { [weak self] in self?.telemetryStore.reloadConfiguration() },
        restartServer: { [weak self] in self?.restartServer() },
        checkForUpdates: { [weak self] in self?.checkForUpdates() },
        automaticallyChecksForUpdates: Binding(
          get: { [weak self] in self?.applicationUpdateController.automaticallyChecksForUpdates ?? true },
          set: { [weak self] shouldCheckAutomatically in
            guard let updateController = self?.applicationUpdateController else { return }
            setAutomaticApplicationUpdateChecks(shouldCheckAutomatically, using: updateController)
          }
        ),
        selectedUpdateChannel: Binding(
          get: { [weak self] in self?.applicationUpdateController.selectedChannel ?? .stable },
          set: { [weak self] updateChannel in
            self?.applicationUpdateController.selectUpdateChannel(updateChannel)
          }
        ),
        revealConfiguration: revealConfiguration,
        quitApplication: { NSApp.terminate(nil) }
      )
    )
    telemetryPopover = popover
    applicationUpdateController.start()
    telemetryStore.onMenuBarTitleChanged = { [weak self] menuBarTitle in
      guard let self else { return }
      let channelAwareMenuBarTitle =
        applicationIdentity.channel == .development
        ? " DEV\(menuBarTitle.isEmpty ? "" : " · \(menuBarTitle)")" : menuBarTitle
      latestMenuBarTitle = channelAwareMenuBarTitle
      let currentTitle = statusItem?.button?.title ?? ""
      statusItem?.button?.title = menuBarTitleToDisplay(
        currentTitle: currentTitle,
        latestTitle: channelAwareMenuBarTitle,
        popoverIsShown: telemetryPopover?.isShown == true
      )
    }
    Task { [weak self] in
      guard let self else { return }
      await startDaemonForApplication(
        daemonLifecycleController: daemonLifecycleController,
        telemetryStore: telemetryStore)
    }
    telemetryStore.startPolling()
    DispatchQueue.main.async { NSApp.setActivationPolicy(.accessory) }
  }

  func applicationWillTerminate(_ notification: Notification) {
    telemetryStore.stopPolling()
    daemonLifecycleController.stopOwnedDaemon()
  }

  @objc private func toggleTelemetryPopover() {
    guard let telemetryPopover else { return }
    telemetryPopover.isShown ? telemetryPopover.performClose(nil) : showTelemetryPopover()
  }

  private func showTelemetryPopover() {
    guard let statusButton = statusItem?.button, let telemetryPopover else { return }
    statusItem?.length = menuBarStatusItemLength(
      popoverIsShown: true, currentButtonWidth: statusButton.bounds.width)
    telemetryPopover.show(
      relativeTo: popoverAnchorRect(
        statusButtonBounds: statusButton.bounds,
        statusItemImageRect: statusButton.cell?.imageRect(forBounds: statusButton.bounds)
      ), of: statusButton,
      preferredEdge: .minY)
    telemetryStore.setPopoverVisible(true)
  }

  func popoverDidClose(_ notification: Notification) {
    telemetryStore.setPopoverVisible(false)
    statusItem?.length = menuBarStatusItemLength(
      popoverIsShown: false,
      currentButtonWidth: statusItem?.button?.bounds.width ?? NSStatusItem.squareLength
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

  private func openObservatory() {
    do {
      let observatoryURL = try applicationIdentity.endpointURL(path: "/")
      guard NSWorkspace.shared.open(observatoryURL) else {
        throw ObservatoryLaunchError.defaultBrowserUnavailable
      }
    } catch {
      NSApp.presentError(error)
    }
  }

  private func openLibrary() {
    do {
      let libraryURL = try applicationIdentity.endpointURL(path: "/library")
      guard NSWorkspace.shared.open(libraryURL) else {
        throw ObservatoryLaunchError.defaultBrowserUnavailable
      }
    } catch {
      NSApp.presentError(error)
    }
  }

  private func revealConfiguration() {
    let configurationURL = applicationIdentity.configFileURL()
    NSWorkspace.shared.activateFileViewerSelecting([configurationURL])
  }

  private func checkForUpdates() {
    requestManualApplicationUpdateCheck(using: applicationUpdateController)
  }
}

@MainActor
func startDaemonForApplication(
  daemonLifecycleController: DaemonLifecycleController,
  telemetryStore: TelemetryStore
) async {
  do {
    try await daemonLifecycleController.startDaemonIfNeeded()
  } catch {
    telemetryStore.failServerStartup(error)
  }
}

private enum ObservatoryLaunchError: LocalizedError {
  case defaultBrowserUnavailable

  var errorDescription: String? {
    "The Observatory could not be opened in the default browser"
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
  statusItemImageRect ?? statusButtonBounds
}
