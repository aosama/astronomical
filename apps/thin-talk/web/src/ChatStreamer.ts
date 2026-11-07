import { ChatMessage } from "./ChatMessage";
import { ThinkingEffort } from "./ThinkingEffort";
import { ToolStep } from "./ToolStep";

/** One streamed assistant turn the conversation asks for. */
export interface ChatStreamRequest {
  modelID: string;
  messages: ChatMessage[];
  thinkingEffort: ThinkingEffort;
  /** Whether the user's Tools switch is on. */
  toolsEnabled: boolean;
  /** Whether the chosen model declares that it can call tools. */
  modelSupportsToolCalls: boolean;
  onText: (delta: string) => void;
  onReasoning: (delta: string) => void;
  /** Called when a tool call starts and again when it finishes, with the
   * same tool call id, so the conversation can show the step as it runs. */
  onToolStep: (toolStep: ToolStep) => void;
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
}
