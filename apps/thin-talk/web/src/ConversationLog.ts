import { ChatMessage } from "./ChatMessage";
import { ChatRole } from "./ChatRole";
import { ChatMessageState } from "./ChatMessageState";
import { ToolStep } from "./ToolStep";

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

  /** Adds a message the user sent while an answer was still coming. */
  public appendQueuedUserTurn(content: string): ChatMessage {
    const queuedMessage = new ChatMessage({
      id: ConversationLog.newIdentifier(),
      role: ChatRole.USER,
      content,
      state: ChatMessageState.QUEUED,
    });
    this.messages.push(queuedMessage);
    return queuedMessage;
  }

  /** The oldest queued message still waiting for an answer, or null. */
  public firstQueuedUserTurn(): ChatMessage | null {
    const queuedMessage = this.messages.find((message) => message.state === ChatMessageState.QUEUED);
    return queuedMessage != null ? queuedMessage : null;
  }

  /** Starts answering a queued message: the answer in progress is finished
   * (or dropped if it is still empty), the queued message moves to just after
   * the last message that is not waiting, and its answer opens right after it.
   * The log then reads in the order the model saw the conversation, and a
   * later request always ends with this question. Returns false when the id
   * is not a queued message. */
  public beginAnsweringQueuedTurn(queuedMessageID: string): boolean {
    const queuedIndex = this.messages.findIndex((message) => message.id === queuedMessageID && message.state === ChatMessageState.QUEUED);
    if (queuedIndex < 0) {
      return false;
    }
    this.markAssistantTurn(ChatMessageState.COMPLETE);
    this.removeEmptyAssistantTurn();
    const [queuedMessage] = this.messages.splice(this.messages.findIndex((message) => message.id === queuedMessageID), 1);
    if (queuedMessage == null) {
      return false;
    }
    queuedMessage.state = ChatMessageState.COMPLETE;
    const insertionIndex = this.indexAfterLastSettledMessage();
    const answerID = ConversationLog.newIdentifier();
    this.assistantMessageID = answerID;
    this.messages.splice(insertionIndex, 0, queuedMessage, new ChatMessage({ id: answerID, role: ChatRole.ASSISTANT, content: "" }));
    return true;
  }

  /** Where a message that is being answered now belongs: after every
   * message that was sent and answered (or failed), ahead of messages still
   * queued or never sent. */
  private indexAfterLastSettledMessage(): number {
    for (let messageIndex = this.messages.length - 1; messageIndex >= 0; messageIndex -= 1) {
      const candidate = this.messages[messageIndex];
      if (candidate != null && !ConversationLog.isWaitingUserMessage(candidate)) {
        return messageIndex + 1;
      }
    }
    return 0;
  }

  /** A user message that was queued, or queued and never sent. */
  private static isWaitingUserMessage(message: ChatMessage): boolean {
    return message.role === ChatRole.USER && (message.state === ChatMessageState.QUEUED || message.state === ChatMessageState.FAILED);
  }

  /** Marks every message still queued as not sent, as when the user stops
   * or the answer fails; such messages stay visible but are left out of
   * later requests. */
  public markQueuedTurnsNotSent(): void {
    for (const message of this.messages) {
      if (message.state === ChatMessageState.QUEUED) {
        message.state = ChatMessageState.FAILED;
      }
    }
  }

  /** Opens the assistant turn up-front and hands its id back. Without this the
   * assistant id leaks from the previous turn and a second answer keeps
   * appending inside the first assistant bubble; a visible placeholder also
   * prevents stacked user turns from looking unanswered. */
  public openAssistantTurn(): string {
    const placeholderID = ConversationLog.newIdentifier();
    this.assistantMessageID = placeholderID;
    // Ahead of any message still queued or never sent, so a retry's answer
    // sits under the question it answers.
    this.messages.splice(
      this.indexAfterLastSettledMessage(),
      0,
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

  /** Records a tool step on the turn being answered: a new call is appended,
   * and a later update for the same call (its result) replaces it in place,
   * so steps stay in the order the model made them. */
  public applyToolStep(toolStep: ToolStep): void {
    const assistantTurn = this.currentAssistantTurn();
    const turn = assistantTurn != null ? assistantTurn : this.currentAssistantTurnByID(this.openAssistantTurn());
    const existingIndex = turn.toolSteps.findIndex((recordedStep) => recordedStep.toolCallID === toolStep.toolCallID);
    if (existingIndex >= 0) {
      turn.toolSteps[existingIndex] = toolStep;
    } else {
      turn.toolSteps.push(toolStep);
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
    if (turn.content.length === 0 && turn.reasoning.length === 0 && turn.toolSteps.length === 0) {
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

  /** The latest question that was actually sent; queued and never-sent
   * messages are not something a retry can answer. */
  public lastUserTurn(): ChatMessage | null {
    for (let index = this.messages.length - 1; index >= 0; index -= 1) {
      const candidate = this.messages[index];
      if (candidate && candidate.role === ChatRole.USER && !ConversationLog.isWaitingUserMessage(candidate)) {
        return candidate;
      }
    }
    return null;
  }

  /** The history a new turn sends: every message except failed ones, queued
   * ones not yet being answered, and the assistant turn being answered right now. That turn is opened as a
   * placeholder before the request goes out, and sending it made the daemon
   * render an empty assistant reply into the prompt ahead of the real one. */
  public requestMessages(): ChatMessage[] {
    return this.messages.filter(
      (message) =>
        message.state !== ChatMessageState.FAILED &&
        message.state !== ChatMessageState.QUEUED &&
        message.id !== this.assistantMessageID,
    );
  }

  private currentAssistantTurn(): ChatMessage | null {
    if (!this.assistantMessageID) {
      return null;
    }
    const assistantTurn: ChatMessage | undefined = this.messages.find(
      (message) => message.id === this.assistantMessageID,
    );
    return assistantTurn != null ? assistantTurn : null;
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
