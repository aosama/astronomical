import { HostCommand, parseHostCommand } from "./HostCommand";
import { validTitlebarInset } from "./WindowChrome";

/** What each host command does; boot.ts supplies these from the controller and
 * views, so the router owns dispatch and nothing else. */
export interface HostCommandHandlers {
  newChat(): void;
  toggleSidebar(): void;
  stopGenerating(): void;
  copyLastAnswer(): void;
  deleteConversation(): void;
  exportConversation(): void;
  focusComposer(): void;
  openYourAstro(): void;
  toggleAstro(): void;
  setTitlebarInset(titlebarInsetPixels: number): void;
}

/**
 * Receives commands from the native menu bar and toolbar.
 *
 * The host calls `window.__thintalkCommands.run({name})`; `setTitlebarInset`
 * also carries `points`. The return value says whether the command was
 * understood, which the tests check; the host does not read it.
 */
export class HostCommandRouter {
  private readonly handlers: HostCommandHandlers;

  public constructor(handlers: HostCommandHandlers) {
    this.handlers = handlers;
  }

  public install(target: Window): void {
    target.__thintalkCommands = { run: (payload: unknown) => this.run(payload) };
  }

  public run(payload: unknown): boolean {
    const hostCommand = parseHostCommand(payload);
    if (hostCommand === null) {
      return false;
    }
    if (hostCommand === HostCommand.SET_TITLEBAR_INSET) {
      const titlebarInsetPixels = validTitlebarInset((payload as Record<string, unknown>)["points"]);
      if (titlebarInsetPixels === null) {
        return false;
      }
      this.handlers.setTitlebarInset(titlebarInsetPixels);
      return true;
    }
    this.dispatch(hostCommand);
    return true;
  }

  private dispatch(hostCommand: Exclude<HostCommand, HostCommand.SET_TITLEBAR_INSET>): void {
    switch (hostCommand) {
      case HostCommand.NEW_CHAT:
        this.handlers.newChat();
        return;
      case HostCommand.TOGGLE_SIDEBAR:
        this.handlers.toggleSidebar();
        return;
      case HostCommand.STOP_GENERATING:
        this.handlers.stopGenerating();
        return;
      case HostCommand.COPY_LAST_ANSWER:
        this.handlers.copyLastAnswer();
        return;
      case HostCommand.DELETE_CONVERSATION:
        this.handlers.deleteConversation();
        return;
      case HostCommand.EXPORT_CONVERSATION:
        this.handlers.exportConversation();
        return;
      case HostCommand.FOCUS_COMPOSER:
        this.handlers.focusComposer();
        return;
      case HostCommand.OPEN_YOUR_ASTRO:
        this.handlers.openYourAstro();
        return;
      case HostCommand.TOGGLE_ASTRO:
        this.handlers.toggleAstro();
        return;
    }
  }
}
