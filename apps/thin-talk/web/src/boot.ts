import { AppConfig } from "./AppConfig";
import { HostChannel } from "./HostChannel";
import { SessionBridge } from "./SessionBridge";
import { SessionStore } from "./SessionStore";
import { SupervisorClient } from "./SupervisorClient";
import { TrustBoundary } from "./TrustBoundary";
import { MathPipeline } from "./MathPipeline";
import { DiagramPipeline } from "./DiagramPipeline";
import { Renderer } from "./Renderer";
import { TranscriptView } from "./TranscriptView";
import { ComposerView } from "./ComposerView";
import { SidebarView } from "./SidebarView";
import { ChatController } from "./ChatController";
import { AppShell } from "./AppShell";
import { ThinkingEffort } from "./ThinkingEffort";

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`the page is missing its "${selector}" element`);
  }
  return element;
}

/**
 * Wires every client component to the page's elements and starts the shell.
 * Both the production entry (main.ts) and the test harness call this, so there
 * is exactly one wiring path and a test boot is the production boot.
 */
export function bootChatClient(): AppShell {
  const config = AppConfig.fromWindow(window.__thintalkConfig);
  const host = new HostChannel();
  const bridge = new SessionBridge((request) => host.postSessionCall(request));
  const store = new SessionStore(bridge);
  const client = new SupervisorClient(config);

  const trust = new TrustBoundary();
  const math = new MathPipeline(trust);
  const diagrams = new DiagramPipeline();
  const renderer = new Renderer({ trust, math, diagrams });

  const transcript = new TranscriptView({
    transcript: requireElement<HTMLElement>("#transcript"),
    emptyState: requireElement<HTMLElement>("#empty-state"),
    renderer,
    onAction: (action) => {
      if (action.kind === "copy") {
        // The host owns the pasteboard, so the text travels in the action detail.
        host.postAction("copy", action.messageID, transcript.renderedText(action.messageID));
        return;
      }
      host.postAction("openExternal", "", action.detail);
    },
    onError: (detail) => host.postError(detail),
  });

  const composer = new ComposerView({
    field: requireElement<HTMLTextAreaElement>("#composer-input"),
    sendButton: requireElement<HTMLButtonElement>('[data-testid="composer-send"]'),
    stopButton: requireElement<HTMLButtonElement>('[data-testid="composer-stop"]'),
    retryButton: requireElement<HTMLButtonElement>('[data-testid="composer-retry"]'),
    effortPill: requireElement<HTMLButtonElement>('[data-testid="composer-effort"]'),
    effortLabel: requireElement<HTMLElement>('[data-testid="composer-effort-label"]'),
    effortMenu: requireElement<HTMLElement>('[data-testid="composer-effort-menu"]'),
    banner: requireElement<HTMLElement>('[data-testid="composer-banner"]'),
    bannerMessage: requireElement<HTMLElement>('[data-testid="banner-message"]'),
    bannerNextAction: requireElement<HTMLElement>('[data-testid="banner-next-action"]'),
    onAction: (action) => {
      switch (action.kind) {
        case "send":
          controller.send(action.detail);
          return;
        case "stop":
          void controller.stop();
          return;
        case "retry":
          controller.retry();
          return;
        case "setEffort":
          controller.setEffort(ThinkingEffort.parse(action.detail));
          return;
      }
    },
  });

  const sidebar = new SidebarView({
    newChatButton: requireElement<HTMLButtonElement>("#new-chat"),
    sessionList: requireElement<HTMLElement>("#session-list"),
    onSelect: (id) => void controller.openSession(id),
    onNewChat: () => void controller.newChat(),
  });

  const controller = new ChatController({ client, bridge, store });
  const shell = new AppShell({ controller, transcript, composer, sidebar, host });
  shell.start();
  void controller.load();
  return shell;
}
