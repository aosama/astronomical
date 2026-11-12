import { SessionSummary } from "./SessionSummary";

const UNTITLED_SESSION_LABEL = "New chat";

/**
 * Sidebar layer: the new-chat button and the session list.
 *
 * This class renders the session list the controller publishes and reports
 * intents back (select, open the right-click menu, rename); it holds no
 * conversation state. The one thing it owns is an edit in progress: the row
 * being renamed keeps its field across re-renders, so a reply streaming in
 * elsewhere never wipes what the user is typing.
 */
export class SidebarView {
  private readonly newChatButton: HTMLButtonElement;
  private readonly sessionList: HTMLElement;
  private readonly onSelect: (id: string) => void;
  private readonly onNewChat: () => void;
  private readonly onConversationMenu: (id: string) => void;
  private readonly onRename: (id: string, title: string) => void;
  private renameField: HTMLInputElement | null = null;
  private renamingSessionID: string | null = null;
  /** Set while the list is rebuilt: moving the rename field to its new row
   * can blur it, and that blur is not the user leaving the field. */
  private isRendering = false;
  private latestState: { sessions: readonly SessionSummary[]; activeSessionID: string | null } = { sessions: [], activeSessionID: null };

  public constructor(init: {
    newChatButton: HTMLButtonElement;
    sessionList: HTMLElement;
    onSelect: (id: string) => void;
    onNewChat: () => void;
    onConversationMenu: (id: string) => void;
    onRename: (id: string, title: string) => void;
  }) {
    this.newChatButton = init.newChatButton;
    this.sessionList = init.sessionList;
    this.onSelect = init.onSelect;
    this.onNewChat = init.onNewChat;
    this.onConversationMenu = init.onConversationMenu;
    this.onRename = init.onRename;
    this.newChatButton.addEventListener("click", () => this.onNewChat());
    this.sessionList.addEventListener("click", (event) => {
      const id = SidebarView.sessionIDAt(event.target);
      if (id != null && id !== this.renamingSessionID) {
        this.onSelect(id);
      }
    });
    this.sessionList.addEventListener("contextmenu", (event) => {
      const id = SidebarView.sessionIDAt(event.target);
      if (id == null || id === this.renamingSessionID) {
        return;
      }
      // The page's own menu replaces WebKit's text menu (Look Up, Translate)
      // on a conversation row; elsewhere the default menu stays.
      event.preventDefault();
      this.onConversationMenu(id);
    });
  }

  public render(state: { sessions: readonly SessionSummary[]; activeSessionID: string | null }): void {
    this.latestState = state;
    if (this.renamingSessionID != null && !state.sessions.some((session) => session.id === this.renamingSessionID)) {
      this.endRename();
    }
    this.isRendering = true;
    this.sessionList.textContent = "";
    state.sessions.forEach((session) => {
      const item = document.createElement("li");
      item.setAttribute("role", "none");
      item.appendChild(session.id === this.renamingSessionID && this.renameField != null
        ? this.renameField
        : SidebarView.sessionButton(session, session.id === state.activeSessionID));
      this.sessionList.appendChild(item);
    });
    this.renameField?.focus();
    this.isRendering = false;
  }

  /** Turns one row into a title field, its text selected for replacing. */
  public beginRename(id: string): void {
    const renamedSession = this.latestState.sessions.find((session) => session.id === id);
    if (renamedSession == null) {
      return;
    }
    const renameField = document.createElement("input");
    renameField.type = "text";
    renameField.className = "sidebar__session-rename";
    renameField.dataset.testid = "session-rename";
    renameField.dataset.sessionId = id;
    renameField.setAttribute("aria-label", "Conversation name");
    renameField.value = renamedSession.title;
    renameField.addEventListener("keydown", (event) => this.handleRenameKey(event));
    renameField.addEventListener("blur", () => {
      if (!this.isRendering) {
        this.commitRename();
      }
    });
    this.renameField = renameField;
    this.renamingSessionID = id;
    this.render(this.latestState);
    renameField.select();
  }

  private handleRenameKey(event: KeyboardEvent): void {
    if (event.key === "Enter") {
      event.preventDefault();
      this.commitRename();
    } else if (event.key === "Escape") {
      event.preventDefault();
      this.endRenameInPlace();
    }
  }

  private commitRename(): void {
    const renamedSessionID = this.renamingSessionID;
    const requestedTitle = this.renameField?.value ?? "";
    if (renamedSessionID == null) {
      return;
    }
    // Ended before reporting, so a second blur finds no edit in progress and
    // cannot report the same rename twice.
    this.endRenameInPlace();
    this.onRename(renamedSessionID, requestedTitle);
  }

  /** Puts the renamed row's button back in place of the field, touching no
   * other row. Rebuilding the whole list here would replace the row the user
   * just pressed (pressing another chat is what blurs the field), and the
   * click on it would then never arrive. */
  private endRenameInPlace(): void {
    const renameField = this.renameField;
    const renamedSessionID = this.renamingSessionID;
    this.endRename();
    const renamedSession = this.latestState.sessions.find((session) => session.id === renamedSessionID);
    if (renameField == null || renamedSession == null || renameField.parentElement == null) {
      this.render(this.latestState);
      return;
    }
    renameField.replaceWith(SidebarView.sessionButton(renamedSession, renamedSession.id === this.latestState.activeSessionID));
  }

  private endRename(): void {
    this.renamingSessionID = null;
    this.renameField = null;
  }

  private static sessionIDAt(target: EventTarget | null): string | null {
    if (!(target instanceof Element)) {
      return null;
    }
    const item = target.closest<HTMLElement>("[data-session-id]");
    const id = item?.dataset.sessionId;
    return id != null && id.length > 0 ? id : null;
  }

  private static sessionButton(session: SessionSummary, isActive: boolean): HTMLButtonElement {
    const displayedTitle = session.title.length > 0 ? session.title : UNTITLED_SESSION_LABEL;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sidebar__session-item";
    button.dataset.testid = "session-item";
    button.dataset.sessionId = session.id;
    button.title = displayedTitle;
    if (isActive) {
      button.classList.add("sidebar__session-item--active");
      button.setAttribute("aria-current", "true");
    }
    const title = document.createElement("span");
    title.className = "sidebar__session-title";
    title.textContent = displayedTitle;
    button.appendChild(title);
    return button;
  }
}
