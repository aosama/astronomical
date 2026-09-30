import { ChatController } from "./ChatController";
import { TranscriptView } from "./TranscriptView";
import { ComposerView } from "./ComposerView";
import { SidebarView } from "./SidebarView";
import { ChatSurfaceState } from "./ChatSurfaceState";
import { HostChannel } from "./HostChannel";

const COMPOSER_FONT_SIZE_VARIABLE = "--composer-font-size";

/**
 * Wires the views to the controller and the page to its host.
 *
 * This class owns no behaviour: sending, stopping, effort choice, and session
 * switching all run in the controller; copying and opening a link leave the page
 * because the host owns the pasteboard and the system link opener. It applies
 * the published surface state to every view and exposes the test entry point.
 */
export class AppShell {
  private readonly controller: ChatController;
  private readonly transcript: TranscriptView;
  private readonly composer: ComposerView;
  private readonly sidebar: SidebarView;
  private readonly host: HostChannel;

  public constructor(init: {
    controller: ChatController;
    transcript: TranscriptView;
    composer: ComposerView;
    sidebar: SidebarView;
    host: HostChannel;
  }) {
    this.controller = init.controller;
    this.transcript = init.transcript;
    this.composer = init.composer;
    this.sidebar = init.sidebar;
    this.host = init.host;
  }

  public start(): void {
    this.controller.subscribe((state) => this.render(state));
    window.addEventListener("error", (event) => {
      this.host.postError(`${event.message} (${event.filename}:${event.lineno})`);
    });
    window.addEventListener("unhandledrejection", (event) => {
      this.host.postError(`unhandled rejection: ${String(event.reason)}`);
    });
    window.__thintalk = {
      messageCount: () => this.transcript.messageCount(),
      renderedText: (messageID: string) => this.transcript.renderedText(messageID),
      diagnostics: () => this.transcript.diagnostics(),
    };
  }

  private render(state: ChatSurfaceState): void {
    document.documentElement.style.setProperty(COMPOSER_FONT_SIZE_VARIABLE, `${state.composerFontSize}px`);
    this.transcript.render(state.transcript);
    this.composer.render(state.composer);
    this.sidebar.render({ sessions: state.sessions, activeSessionID: state.activeSessionID });
  }
}
