import { SessionSummary } from "./SessionSummary";

/**
 * Sidebar layer: the new-chat button and the session list.
 *
 * This class renders the session list the controller publishes and reports
 * selection intents back; it holds no conversation state.
 */
export class SidebarView {
  private readonly newChatButton: HTMLButtonElement;
  private readonly sessionList: HTMLElement;
  private readonly onSelect: (id: string) => void;
  private readonly onNewChat: () => void;

  public constructor(init: {
    newChatButton: HTMLButtonElement;
    sessionList: HTMLElement;
    onSelect: (id: string) => void;
    onNewChat: () => void;
  }) {
    this.newChatButton = init.newChatButton;
    this.sessionList = init.sessionList;
    this.onSelect = init.onSelect;
    this.onNewChat = init.onNewChat;
    this.newChatButton.addEventListener("click", () => this.onNewChat());
    this.sessionList.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof Element)) {
        return;
      }
      const item = target.closest("[data-session-id]");
      if (!item) {
        return;
      }
      const id = (item as HTMLElement).dataset.sessionId;
      if (id) {
        this.onSelect(id);
      }
    });
  }

  public render(state: { sessions: readonly SessionSummary[]; activeSessionID: string | null }): void {
    this.sessionList.textContent = "";
    state.sessions.forEach((session) => {
      const item = document.createElement("li");
      item.setAttribute("role", "none");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "sidebar__session-item";
      button.dataset.testid = "session-item";
      button.dataset.sessionId = session.id;
      button.title = session.title.length > 0 ? session.title : "New chat";
      if (session.id === state.activeSessionID) {
        button.classList.add("sidebar__session-item--active");
        button.setAttribute("aria-current", "true");
      }
      const title = document.createElement("span");
      title.className = "sidebar__session-title";
      title.textContent = session.title.length > 0 ? session.title : "New chat";
      button.appendChild(title);
      item.appendChild(button);
      this.sessionList.appendChild(item);
    });
  }
}
