import { ChatMessage } from "./ChatMessage";
import { ThinkingEffort } from "./ThinkingEffort";
import { ToolStep } from "./ToolStep";

/** One streamed assistant turn the conversation asks for. */
export interface ChatStreamRequest {
  modelID: string;
  /** The conversation being answered, for tools that remember approvals;
   * null when none is open. */
  conversationID: string | null;
  messages: ChatMessage[];
  thinkingEffort: ThinkingEffort;
  /** Whether the user's Tools switch is on. */
  toolsEnabled: boolean;
  /** Whether the chosen model declares that it can call tools. */
  modelSupportsToolCalls: boolean;
  /** Whether the chosen model can take images; images it generated earlier
   * in the conversation are then shown to it again. */
  modelAcceptsImages: boolean;
  /** The chosen model's context window in tokens; long conversations are
   * trimmed to fit it. Null when the daemon does not say. */
  contextWindowTokens: number | null;
  onText: (delta: string) => void;
  onReasoning: (delta: string) => void;
  /** Called when a tool call starts and again when it finishes, with the
   * same tool call id, so the conversation can show the step as it runs. */
  onToolStep: (toolStep: ToolStep) => void;
  /** Called when the turn starts answering a message queued while it ran,
   * so the conversation can close the current answer and open the next. */
  onQueuedMessageAnswered: (queuedMessageID: string) => void;
  signal: AbortSignal;
}

/**
 * The seam between the conversation and whatever produces the assistant turn.
 *
 * The controller only needs text, reasoning and tool-step updates plus a stop
 * signal, so the agent loop behind a turn (AgentTurnRunner today) can run
 * tools and several model requests per turn without the controller changing.
 * A user stop ends the stream normally; every other failure throws a
 * classified ChatFailure. `messages` ends with the user's question and never
 * includes the assistant turn being answered.
 */
export interface ChatStreamer {
  readonly stallTimeoutMilliseconds: number;
  streamChat(request: ChatStreamRequest): Promise<void>;
  /** Hands the running turn a message the user sent while it was working;
   * returns whether a running turn accepted it. A message the turn never
   * starts stays queued in the conversation, which sends it next itself. */
  queueUserMessage(queuedMessageID: string, content: string): boolean;
}
