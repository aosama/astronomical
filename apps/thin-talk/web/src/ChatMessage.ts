import { ChatRole } from "./ChatRole";
import { ChatMessageState } from "./ChatMessageState";
import { ToolStep } from "./ToolStep";

/**
 * One message the user typed or the assistant produced. Storage stays simple
 * (plain text, separate reasoning, and the tool steps of an assistant turn) so
 * richer modalities can be added later without rewriting the core model.
 */
export class ChatMessage {
  public readonly id: string;
  public role: ChatRole;
  public content: string;
  /** Model reasoning kept separate from the visible answer, so the surface can
   * render the thinking quietly instead of mixing it into the reply. */
  public reasoning: string;
  public state: ChatMessageState;
  /** The tools the assistant called while producing this answer, in call
   * order. Always empty for a user message. */
  public toolSteps: ToolStep[];

  public constructor(init: {
    id: string;
    role: ChatRole;
    content: string;
    reasoning?: string;
    state?: ChatMessageState;
    toolSteps?: ToolStep[];
  }) {
    this.id = init.id;
    this.role = init.role;
    this.content = init.content;
    const configuredReasoning: string | undefined = init.reasoning;
    this.reasoning = configuredReasoning != null ? configuredReasoning : "";
    const configuredState: ChatMessageState | undefined = init.state;
    this.state = configuredState != null ? configuredState : ChatMessageState.COMPLETE;
    const configuredToolSteps: ToolStep[] | undefined = init.toolSteps;
    this.toolSteps = configuredToolSteps != null ? configuredToolSteps : [];
  }
}
