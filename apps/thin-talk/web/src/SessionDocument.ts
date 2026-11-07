import { ChatMessage } from "./ChatMessage";
import { ChatRole } from "./ChatRole";
import { ChatMessageState } from "./ChatMessageState";
import { ToolStep } from "./ToolStep";

/** The schema version this client writes. Bump when the document shape changes.
 * Version 2 adds `toolSteps` to assistant messages; a version 1 file has none
 * and loads unchanged. */
export const SESSION_SCHEMA_VERSION = 2;

/** The longest title derived from a first user message. */
const MAX_TITLE_LENGTH = 48;

/**
 * One persisted conversation: the document the session bridge stores on disk.
 * Dates are ISO 8601 strings so the JSON round trip stays lossless across the
 * bridge and the Swift file store.
 */
export class SessionDocument {
  public schemaVersion: number;
  public id: string;
  public title: string;
  public createdAt: string;
  public updatedAt: string;
  public messages: ChatMessage[];

  public constructor(init: {
    id: string;
    title: string;
    createdAt: string;
    updatedAt: string;
    messages: ChatMessage[];
    schemaVersion?: number;
  }) {
    const configuredSchemaVersion: number | undefined = init.schemaVersion;
    this.schemaVersion =
      configuredSchemaVersion != null ? configuredSchemaVersion : SESSION_SCHEMA_VERSION;
    this.id = init.id;
    this.title = init.title;
    this.createdAt = init.createdAt;
    this.updatedAt = init.updatedAt;
    this.messages = init.messages;
  }

  /** Prepares messages for writing: an empty streaming placeholder is dropped
   * and a partial answer caught mid-stream is downgraded to stopped, so a
   * reloaded session never shows a blank bubble or a phantom spinner. */
  public static persistableMessages(messages: readonly ChatMessage[]): ChatMessage[] {
    const persistable: ChatMessage[] = [];
    messages.forEach((message) => {
      const hasContent = message.content.length > 0 || message.reasoning.length > 0 || message.toolSteps.length > 0;
      if (message.state === ChatMessageState.STREAMING) {
        if (!hasContent) {
          return;
        }
        persistable.push(
          new ChatMessage({
            id: message.id,
            role: message.role,
            content: message.content,
            reasoning: message.reasoning,
            state: ChatMessageState.STOPPED,
            toolSteps: message.toolSteps,
          }),
        );
        return;
      }
      if (!hasContent && message.state === ChatMessageState.COMPLETE) {
        return;
      }
      persistable.push(message);
    });
    return persistable;
  }

  /** Derives the session title from the first user message, trimmed to a
   * readable length so the sidebar stays scannable. */
  public static titleFromFirstUserMessage(messages: ChatMessage[]): string {
    const firstUserMessage = messages.find((message) => message.role === ChatRole.USER);
    const source = firstUserMessage ? firstUserMessage.content : "";
    const collapsed = source.replace(/\s+/g, " ").trim();
    if (collapsed.length <= MAX_TITLE_LENGTH) {
      return collapsed;
    }
    return `${collapsed.slice(0, MAX_TITLE_LENGTH)}…`;
  }

  public toPlain(): unknown {
    return {
      schemaVersion: this.schemaVersion,
      id: this.id,
      title: this.title,
      createdAt: this.createdAt,
      updatedAt: this.updatedAt,
      messages: this.messages.map((message) => ({
        id: message.id,
        role: message.role,
        content: message.content,
        reasoning: message.reasoning,
        state: message.state,
        toolSteps: message.toolSteps.map((toolStep) => toolStep.toPlain()),
      })),
    };
  }

  /** Rebuilds a document from bridge JSON, validating every field so a corrupt
   * or foreign file fails loudly instead of poisoning the conversation. */
  public static fromPlain(plain: unknown): SessionDocument {
    if (typeof plain !== "object" || plain === null) {
      throw new Error("session document is not an object");
    }
    const record = plain as Record<string, unknown>;
    const id = record["id"];
    if (typeof id !== "string" || id.length === 0) {
      throw new Error("session document has no id");
    }
    const title = typeof record["title"] === "string" ? record["title"] : "";
    const createdAt = typeof record["createdAt"] === "string" ? record["createdAt"] : new Date(0).toISOString();
    const updatedAt = typeof record["updatedAt"] === "string" ? record["updatedAt"] : createdAt;
    const rawMessages = Array.isArray(record["messages"]) ? record["messages"] : [];
    const messages = rawMessages.map((rawMessage) => SessionDocument.messageFromPlain(rawMessage));
    return new SessionDocument({ id, title, createdAt, updatedAt, messages });
  }

  private static messageFromPlain(rawMessage: unknown): ChatMessage {
    if (typeof rawMessage !== "object" || rawMessage === null) {
      throw new Error("session message is not an object");
    }
    const record = rawMessage as Record<string, unknown>;
    const id = typeof record["id"] === "string" ? record["id"] : "";
    if (id.length === 0) {
      throw new Error("session message has no id");
    }
    const role = record["role"] === ChatRole.USER ? ChatRole.USER : ChatRole.ASSISTANT;
    const content = typeof record["content"] === "string" ? record["content"] : "";
    const reasoning = typeof record["reasoning"] === "string" ? record["reasoning"] : "";
    const state = SessionDocument.stateFromPlain(record["state"]);
    const toolSteps = SessionDocument.toolStepsFromPlain(record["toolSteps"]);
    return new ChatMessage({ id, role, content, reasoning, state, toolSteps });
  }

  /** Tool steps saved with a message; absent in version 1 files. A damaged
   * step is dropped on its own so the rest of the conversation still opens. */
  private static toolStepsFromPlain(rawToolSteps: unknown): ToolStep[] {
    if (!Array.isArray(rawToolSteps)) {
      return [];
    }
    return rawToolSteps
      .map((rawToolStep) => ToolStep.fromPlain(rawToolStep))
      .filter((toolStep): toolStep is ToolStep => toolStep !== null);
  }

  private static stateFromPlain(rawState: unknown): ChatMessageState {
    if (rawState === ChatMessageState.STREAMING) return ChatMessageState.STREAMING;
    if (rawState === ChatMessageState.STOPPED) return ChatMessageState.STOPPED;
    if (rawState === ChatMessageState.FAILED) return ChatMessageState.FAILED;
    return ChatMessageState.COMPLETE;
  }
}
