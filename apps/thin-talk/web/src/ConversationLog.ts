import { ChatMessage } from "./ChatMessage";
import { ChatRole } from "./ChatRole";
import { ChatMessageState } from "./ChatMessageState";

/**
 * Owns the message history and folds stream deltas into it. Keeping the fold
 * free of network and view code keeps the conversation behaviour unit-testable:
 * on a delta it accumulates into the current assistant message or opens one.
 */
export class ConversationLog {
  public messages: ChatMessage[] = [];
  public assistantMessageID: string | null = null;

  public appendUserTurn(content: string): ChatMessage {
    const userMessage = new ChatMessage({
      id: ConversationLog.newIdentifier(),
      role: ChatRole.USER,
      content,
    });
    this.messages.push(userMessage);
    return userMessage;
  }

  /** Opens the assistant turn up-front and hands its id back. Without this the
   * assistant id leaks from the previous turn and a second answer keeps
   * appending inside the first assistant bubble; a visible placeholder also
   * prevents stacked user turns from looking unanswered. */
  public openAssistantTurn(): string {
    const placeholderID = ConversationLog.newIdentifier();
    this.assistantMessageID = placeholderID;
    this.messages.push(
      new ChatMessage({ id: placeholderID, role: ChatRole.ASSISTANT, content: "" }),
    );
    return placeholderID;
  }

  public applyTextDelta(text: string): void {
    const assistantTurn = this.currentAssistantTurn();
    if (assistantTurn) {
      assistantTurn.content += text;
    } else {
      const openedID = this.openAssistantTurn();
      this.currentAssistantTurnByID(openedID).content += text;
    }
  }

  public applyReasoningDelta(text: string): void {
    const assistantTurn = this.currentAssistantTurn();
    if (assistantTurn) {
      assistantTurn.reasoning += text;
    } else {
      // Reasoning can arrive before any visible text: open the message now.
      const openedID = this.openAssistantTurn();
      this.currentAssistantTurnByID(openedID).reasoning += text;
    }
  }

  /** Clears the log for a fresh conversation, as when switching sessions. */
  public reset(): void {
    this.messages = [];
    this.assistantMessageID = null;
  }

  /** Drops a turn the model never filled, so a stopped or failed request leaves
   * no empty bubble behind the user's question. */
  public removeEmptyAssistantTurn(): void {
    const id = this.assistantMessageID;
    if (!id) {
      return;
    }
    const index = this.messages.findIndex((message) => message.id === id);
    if (index < 0) {
      return;
    }
    const turn = this.messages[index];
    if (!turn) {
      return;
    }
    if (turn.content.length === 0 && turn.reasoning.length === 0) {
      this.messages.splice(index, 1);
    }
  }

  public markAssistantTurn(state: ChatMessageState): void {
    const id = this.assistantMessageID;
    if (!id) {
      return;
    }
    const turn = this.messages.find((message) => message.id === id);
    if (turn) {
      turn.state = state;
    }
  }

  public lastUserTurn(): ChatMessage | null {
    for (let index = this.messages.length - 1; index >= 0; index -= 1) {
      const candidate = this.messages[index];
      if (candidate && candidate.role === ChatRole.USER) {
        return candidate;
      }
    }
    return null;
  }

  public requestMessages(): ChatMessage[] {
    return this.messages.filter((message) => message.state !== ChatMessageState.FAILED);
  }

  private currentAssistantTurn(): ChatMessage | null {
    return this.assistantMessageID
      ? this.messages.find((message) => message.id === this.assistantMessageID) ?? null
      : null;
  }

  private currentAssistantTurnByID(id: string): ChatMessage {
    const turn = this.messages.find((message) => message.id === id);
    if (!turn) {
      throw new Error("the assistant turn disappeared mid-stream");
    }
    return turn;
  }

  private static newIdentifier(): string {
    return crypto.randomUUID();
  }
}
