import { ChatRole } from "./ChatRole";
import { ChatMessageState } from "./ChatMessageState";
import { TranscriptAttachment } from "./TranscriptAttachment";

/**
 * The view-model of one transcript message: exactly what the renderer needs to
 * draw one bubble, decoupled from the controller's ChatMessage so the rendering
 * contract stays stable while conversation state evolves.
 */
export class TranscriptMessage {
  public readonly id: string;
  public readonly role: ChatRole;
  public readonly markdown: string;
  public readonly reasoning: string;
  public readonly state: ChatMessageState;
  public readonly attachments: readonly TranscriptAttachment[];

  public constructor(init: {
    id: string;
    role: ChatRole;
    markdown: string;
    reasoning: string;
    state: ChatMessageState;
    attachments?: readonly TranscriptAttachment[];
  }) {
    this.id = init.id;
    this.role = init.role;
    this.markdown = init.markdown;
    this.reasoning = init.reasoning;
    this.state = init.state;
    this.attachments = init.attachments ?? [];
  }
}
