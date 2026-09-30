import { ChatRole } from "./ChatRole";
import { ChatMessageState } from "./ChatMessageState";

/**
 * One message the user typed or the assistant produced. Storage stays simple
 * (plain text plus separate reasoning) so richer modalities can be added later
 * without rewriting the core model.
 */
export class ChatMessage {
  public readonly id: string;
  public role: ChatRole;
  public content: string;
  /** Model reasoning kept separate from the visible answer, so the surface can
   * render the thinking quietly instead of mixing it into the reply. */
  public reasoning: string;
  public state: ChatMessageState;

  public constructor(init: {
    id: string;
    role: ChatRole;
    content: string;
    reasoning?: string;
    state?: ChatMessageState;
  }) {
    this.id = init.id;
    this.role = init.role;
    this.content = init.content;
    this.reasoning = init.reasoning ?? "";
    this.state = init.state ?? ChatMessageState.COMPLETE;
  }
}
