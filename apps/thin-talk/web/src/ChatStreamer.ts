import { ChatMessage } from "./ChatMessage";
import { ThinkingEffort } from "./ThinkingEffort";

/** One streamed assistant turn the conversation asks for. */
export interface ChatStreamRequest {
  modelID: string;
  messages: ChatMessage[];
  thinkingEffort: ThinkingEffort;
  onText: (delta: string) => void;
  onReasoning: (delta: string) => void;
  signal: AbortSignal;
}

/**
 * The seam between the conversation and whatever produces the assistant turn.
 *
 * The controller only needs text and reasoning deltas plus a stop signal, so
 * the agent loop behind a turn (AgentTurnRunner today) can grow tools and
 * several model requests per turn without the controller changing. A user stop
 * ends the stream normally; every other failure throws a classified
 * ChatFailure. `messages` ends with the user's question and never includes
 * the assistant turn being answered.
 */
export interface ChatStreamer {
  readonly stallTimeoutMilliseconds: number;
  streamChat(request: ChatStreamRequest): Promise<void>;
}
